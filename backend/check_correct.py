import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Quick check
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
model.eval()

input_ids = tokenizer.encode('Hello', return_tensors='pt')
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]
    top_token_id = torch.argmax(logits).item()
    top_token = tokenizer.decode([top_token_id])
    print(f'✅ CORRECT: "Hello" should be followed by: "{top_token}" (ID: {top_token_id})')
    
    # Check our API vs this
    print(f'❌ OUR API: "Hello" is generating: "." (ID: 14)')
    print(f'🔥 BUG CONFIRMED: We should get ID {top_token_id} but we\'re getting ID 14') 