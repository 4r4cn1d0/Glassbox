import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("=== DEBUGGING GPT-2 LARGE STEP BY STEP ===")

# Load model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
model.eval()

prompt = "Hello"
print(f"Input prompt: '{prompt}'")

# Step 1: Tokenize
input_ids = tokenizer.encode(prompt, return_tensors='pt')
print(f"Input tokens: {input_ids.tolist()[0]}")
print(f"Input tokens decoded: {[tokenizer.decode([t]) for t in input_ids[0]]}")

# Step 2: Forward pass
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]  # Last position logits
    
    # Get top 10 most likely next tokens
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 10)
    
    print(f"\nTOP 10 MOST LIKELY NEXT TOKENS:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token = tokenizer.decode([idx.item()])
        print(f"  {i+1}. '{token}' (ID: {idx.item()}) - {prob.item():.4f}")
    
    # What does greedy decoding pick?
    next_token_id = torch.argmax(logits).item()
    next_token = tokenizer.decode([next_token_id]) 
    print(f"\nGREEDY CHOICE: '{next_token}' (ID: {next_token_id})")
    
    # What should "Hello" + next_token look like?
    new_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)
    full_text = tokenizer.decode(new_ids[0].tolist())
    print(f"FULL TEXT: '{full_text}'")

print("\n=== This is what we SHOULD be generating ===") 