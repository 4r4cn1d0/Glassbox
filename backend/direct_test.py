from simple_gpt2 import gpt2_tracer

print("=== DIRECT TEST OF GPT2LargeTracer ===")

# Test directly
result = gpt2_tracer.trace_generation("Hello", max_new_tokens=2)

print(f"\n🎯 DIRECT RESULT:")
for i, token_data in enumerate(result):
    token = token_data["token"]
    token_id = token_data["token_id"]
    print(f"  {i+1}. '{token}' (ID: {token_id})")

tokens = [t["token"] for t in result]
generated_text = "".join(tokens)
print(f"\n📝 FULL: 'Hello' → '{generated_text}'") 