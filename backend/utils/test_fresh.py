import requests
import json
import time

print("=== Testing Fresh Server on Port 8001 ===")

# Wait for server to start
time.sleep(10)

try:
    # Test generation
    test_data = {
        "prompt": "Hello",
        "max_new_tokens": 2
    }
    
    response = requests.post(
        "http://localhost:8001/api/trace",
        headers={"Content-Type": "application/json"},
        json=test_data
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nFRESH SERVER SUCCESS! Generated {len(data)} tokens:")
        for i, token_data in enumerate(data):
            token = token_data["token"]
            token_id = token_data["token_id"]
            print(f"  {i+1}. '{token}' (ID: {token_id})")
        
        # Show full reconstruction
        tokens = [t["token"] for t in data]
        generated_text = "".join(tokens)
        print(f"\nFRESH SERVER RESULT: 'Hello' → '{generated_text}'")
        
    else:
        print(f"ERROR: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"EXCEPTION: {e}") 