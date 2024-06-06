import requests

train_url = "http://localhost:80/train"
predict_url = "http://localhost:80/predict"
get_url = "http://localhost:80/get_items"

payload = {
    "user_ids": [1, 2, 3, 4]
}


try:
    # Sending a GET request
    response = requests.get(get_url)

    # Checking if the request was successful (status code 200)
    if response.status_code == 200:
        # Printing the response content
        print("Response Content:")
        print(response.json())  # Assuming the response contains JSON data
    else:
        print(f"Request failed with status code: {response.status_code}")

except requests.exceptions.RequestException as e:
    print("Error making the request:", e)

a = input('run prediction? (Y/n)\n')
if a not in ("Y", 'y'):
    exit(0)

# Make the prediction
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
