import requests
import json
import time

print("=== Testing Fixed GPT-2 API ===")

# Wait for server to start
time.sleep(5)

try:
    # Test health check
    response = requests.get("http://localhost:8000/api/health")
    print(f"Health check: {response.status_code}")

    # Test generation
    test_data = {
        "prompt": "Hello",
        "max_new_tokens": 3
    }
    
    response = requests.post(
        "http://localhost:8000/api/trace",
        headers={"Content-Type": "application/json"},
        json=test_data
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ SUCCESS! Generated {len(data)} tokens:")
        for i, token_data in enumerate(data):
            token = token_data["token"]
            token_id = token_data["token_id"]
            print(f"  {i+1}. '{token}' (ID: {token_id})")
        
        # Show full reconstruction
        tokens = [t["token"] for t in data]
        generated_text = "".join(tokens)
        print(f"\n🎯 RESULT: 'Hello' → '{generated_text}'")
        
    else:
        print(f"❌ ERROR: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ EXCEPTION: {e}") 