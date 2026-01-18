import json
import random


ISSUES = [
    {"category": "Billing", "sub_category": "Overcharge", "root_cause": "Auto-renewal enabled without notification"},
    {"category": "Technical", "sub_category": "Login Failure", "root_cause": "MFA token desynchronization"},
    {"category": "Technical", "sub_category": "Slow Upload", "root_cause": "ISP throttling or region mismatch"},
    {"category": "Access", "sub_category": "Permission Denied", "root_cause": "User role downgrade after subscription lapse"}
]

TEMPLATES = [
    "I'm so frustrated! I checked my bank statement and saw a charge for {amount} but I cancelled last week! Fix this.",
    "Hey support, I can't get into my account. It keeps asking for a code but my authenticator app says invalid. Help.",
    "My uploads to the {region} bucket are crawling. Is the server down? I have a deadline.",
    "Why can't I edit the shared folder anymore? It says read-only access."
]

def generate_sample():
    issue = random.choice(ISSUES)
    # Simulate messy user input
    user_text = random.choice(TEMPLATES).format(amount="$49.99", region="us-east-1")
    
    # The "Gold Standard" output we want the model to learn
    # We force a JSON format including 'reasoning' (Chain of Thought)
    structured_output = {
        "classification": f"{issue['category']} > {issue['sub_category']}",
        "reasoning": f"User mentions '{user_text[:20]}...'. This indicates a {issue['sub_category']} issue likely caused by {issue['root_cause']}.",
        "priority": "High" if issue['category'] == "Technical" else "Medium",
        "suggested_action": "Check logs" if issue['category'] == "Technical" else "Review billing history"
    }

    # Alpaca Prompt Format (Standard for instruction tuning)
    prompt = f"""### Instruction:
Analyze the incoming support ticket. Extract the category, reasoning, priority, and suggested action. Output in JSON format.

### Input:
{user_text}

### Response:
{json.dumps(structured_output)}
"""
    return prompt

# Generate 500 samples
print("Generating dataset...")
data = [{"text": generate_sample()} for _ in range(500)]

# Save to JSONL (standard format for LLM training)
with open("support_tickets.jsonl", "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")

print("✅ generated 'support_tickets.jsonl' with 500 samples.")