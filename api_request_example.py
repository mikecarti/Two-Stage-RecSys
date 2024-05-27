import requests

url = "http://localhost:80/predict"

payload = {
    "user_ids": [1, 2, 3, 4]
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    predictions = response.json()
    print(predictions)
else:
    print(f"Request failed with status code: {response.status_code}")
