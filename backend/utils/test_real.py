from transformers import pipeline
import torch

print("=== TESTING REAL GPT-2 LARGE ===")

# Set deterministic seed
torch.manual_seed(42)

# Create HF pipeline
generator = pipeline('text-generation', model='gpt2-large', device=-1)

# Test what it should generate
test_prompts = ["Hello world", "The quick brown fox"]

for prompt in test_prompts:
    print(f"\nPrompt: '{prompt}'")
    result = generator(
        prompt, 
        max_length=len(prompt.split()) + 3,
        do_sample=False,  # Greedy
        num_return_sequences=1
    )
    generated = result[0]['generated_text']
    new_text = generated[len(prompt):].strip()
    print(f"CORRECT: '{prompt}' → '{new_text}'")
    print(f"Full: '{generated}'")

print("\n=== This is what our API should return ===") 