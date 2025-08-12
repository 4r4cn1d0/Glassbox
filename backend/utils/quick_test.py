import requests

try:
    response = requests.post(
        "http://localhost:8001/api/trace",
        headers={"Content-Type": "application/json"},
        json={"prompt": "Hello", "max_new_tokens": 1},
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        token = data[0]["token"]
        token_id = data[0]["token_id"]
        print(f"FRESH SERVER: 'Hello' → '{token}' (ID: {token_id})")
    else:
        print(f"ERROR: {response.status_code}")
        
except Exception as e:
    print(f"EXCEPTION: {e}") 