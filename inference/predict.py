import torch
import json
import re
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TicketPredictor:
    def __init__(self, base_model_id, adapter_path):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._load_model()

    def _load_model(self):
        """Loads the base model and merges the LoRA adapter."""
        try:
            logger.info(f"⏳ Loading base model: {self.base_model_id} on {self.device.upper()}...")
            
            # Load Base Model
            # using float32 on CPU for stability, float16 on GPU for speed
            torch_type = torch.float16 if self.device == "cuda" else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch_type,
                device_map=self.device,
                low_cpu_mem_usage=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)

            # Load and Merge Adapter
            logger.info(f"⏳ Loading adapter from: {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model = self.model.merge_and_unload() # Optimizes inference speed
            self.model.eval()
            
            logger.info("✅ Model loaded and merged successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def _extract_json(self, text):
        """
        Robustly extracts a JSON object from a string using Regex.
        """
        try:
            # Look for content between the first { and the last }
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in model output.")
                return None
        except json.JSONDecodeError:
            logger.warning("Failed to decode JSON from model output.")
            return None

    def predict(self, ticket_text):
        """
        Runs inference and returns a Python Dictionary.
        """
        # This PROMPT FORMAT must match your training code exactly
        prompt = f"""### Instruction:
Analyze the incoming support ticket. Extract the category, reasoning, priority, and suggested action. Output in JSON format.

### Input:
{ticket_text}

### Response:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.3, # Low temp = consistent structure
                top_p=0.9
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Get everything after "### Response:" 
        response_text = full_text.split("### Response:")[-1].strip()
        
        # Robust parsing
        parsed_json = self._extract_json(response_text)
        
        if parsed_json:
            return parsed_json
        else:
            # Fallback if JSON fails (prevents API crash)
            return {
                "error": "Model parsing failed",
                "raw_output": response_text
            }

# ==========================================
# SINGLETON INSTANCE (CRITICAL FOR APP.PY)
# ==========================================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "models/Network-Support-Llama"

# This variable 'predictor' is what api/app.py is importing!
predictor = TicketPredictor(BASE_MODEL, ADAPTER_PATH)

if __name__ == "__main__":
    # Test Run
    test_ticket = "I'm extremely angry! I was charged $49.99 for a renewal I cancelled 2 weeks ago. Refund me now!"
    result = predictor.predict(test_ticket)
    
    print("\n--- FINAL OUTPUT (Dictionary) ---")
    print(json.dumps(result, indent=2))