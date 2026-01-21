import json
import os
import random
import time
from pathlib import Path
from dotenv import load_dotenv  
from groq import Groq

# 1. Load Environment Variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# 2. Configuration
OUTPUT_FILE = Path("data/train_synthetic.jsonl")
NUM_SAMPLES_TO_GENERATE = 50  

# Ensure data directory exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# 3. The "Teacher" Prompt
SYSTEM_PROMPT = """
You are a Senior IT Support Manager creating training data for a fine-tuning task.
Generate 5 UNIQUE, REALISTIC support tickets.

CRITICAL RULES:
1. VARY THE PERSONA:
   - Frustrated Executive (rude, demanding, short)
   - Confused Non-Technical User (vague, "it's broken", typos)
   - DevOps Engineer (technical, mentions logs, Docker, K8s)
   - Panicked Intern (urgent, scared of getting fired)

2. VARY THE PROBLEM:
   - Network (VPN, Slow WiFi, Packet Loss)
   - Access (MFA failure, SSO drift, Locked out)
   - Hardware (Flickering screen, Overheating, Printer jam)
   - Cloud (S3 bucket access, EC2 instance stopped, Billing spike)

3. OUTPUT FORMAT:
   Return strictly a JSON object containing a list under the key "tickets".
   Structure:
   {
     "tickets": [
       {
         "user_query": "The raw complaint text from the user",
         "classification": "Category > SubCategory",
         "reasoning": "A chain-of-thought explanation",
         "priority": "High/Medium/Low",
         "suggested_action": "Actionable resolution step"
       }
     ]
   }
"""

def generate_batch(client):
    """Calls Groq to get a batch of 5 unique tickets."""
    try:
        completion = client.chat.completions.create(
            # UPDATED MODEL NAME HERE:
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate 5 new unique tickets. Random seed: {random.randint(1, 100000)}"}
            ],
            temperature=0.85, # High temperature = more creativity
            response_format={"type": "json_object"}
        )
        # Parse the JSON response
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None

def main():
    if not API_KEY:
        print("❌ ERROR: GROQ_API_KEY not found.")
        print("Please create a .env file in your root folder and add: GROQ_API_KEY=gsk_...")
        return

    client = Groq(api_key=API_KEY)
    
    all_formatted_samples = []
    print(f"🚀 Starting generation of {NUM_SAMPLES_TO_GENERATE} synthetic samples...")

    while len(all_formatted_samples) < NUM_SAMPLES_TO_GENERATE:
        print(f"   ...Generating batch (Current total: {len(all_formatted_samples)})")
        
        batch_data = generate_batch(client)
        
        if batch_data and "tickets" in batch_data:
            current_batch = batch_data["tickets"]
            
            # Convert to Training Format (Instruction Tuning)
            for t in current_batch:
                formatted_entry = {
                    "instruction": "Analyze the support ticket. Output structured JSON classification.",
                    "input": t["user_query"],
                    "output": {
                        "classification": t["classification"],
                        "reasoning": t["reasoning"],
                        "priority": t["priority"],
                        "suggested_action": t["suggested_action"]
                    }
                }
                all_formatted_samples.append(formatted_entry)
        
        time.sleep(1) # Rate limit protection

    # Save to JSONL
    print(f"💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
        for entry in all_formatted_samples[:NUM_SAMPLES_TO_GENERATE]:
            json.dump(entry, f)
            f.write("\n")

    print(f"✅ Success! Generated {len(all_formatted_samples)} samples.")

if __name__ == "__main__":
    main()