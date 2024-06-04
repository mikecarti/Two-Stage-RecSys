import requests

train_url = "http://localhost:80/train"
predict_url = "http://localhost:80/predict"

payload = {
    "user_ids": [1, 2, 3, 4]
}

a = input('run prediction? (Y/n)\n')
if a not in ("Y", 'y'):
    exit(0)

#  make the prediction
response = requests.post(url=predict_url, json=payload)

if response.status_code == 200:
    predictions = response.json()
    print(predictions)
else:
    print(f"Prediction request failed with status code: {response.status_code}. Reason: {response.content}")


a = input('run train? (Y/n)\n')
if a not in ("Y", 'y'):
    exit(0)
# Train the model
response = requests.post(url=train_url, json={})

if response.status_code == 200:
    status = response.json()
    print(status)
    if status.get('success') == 1:
        print("Training successful, proceeding with prediction.")
    else:
        print("Training failed, cannot proceed with prediction.")
else:
    print(f"Training request failed with status code: {response.status_code}")
