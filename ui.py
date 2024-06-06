import pandas as pd
import requests
import streamlit as st

train_url = "http://localhost:80/train"
predict_url = "http://localhost:80/predict"
get_url = "http://localhost:80/get_items"

# Fetch items
response = requests.get(get_url)

if response.status_code == 200:
    items = response.json().get('items')
    print(items[:10], "...")
else:
    print(f"Prediction request failed with status code: {response.status_code}. Reason: {response.content}")

st.write("Hello!")
user_ids_input = st.text_input("Which users to predict? (Enter user IDs separated by spaces)")

if user_ids_input:
    try:
        user_ids = list(map(int, user_ids_input.split()))
    except ValueError:
        st.write("Please enter valid integer user IDs separated by spaces.")
        user_ids = []

    if user_ids and st.button("Send"):
        # Make the prediction
        response = requests.post(url=predict_url, json={"user_ids": user_ids})

        if response.status_code == 200:
            predictions = response.json().get('predictions')
            print(predictions)
        else:
            print(f"Prediction request failed with status code: {response.status_code}. Reason: {response.content}")

        predictions_df = pd.DataFrame(predictions, columns=items, index=user_ids)
        st.write(predictions_df)
        top5_predictions = predictions_df.apply(lambda x: pd.Series(x.nlargest(5).index, name='Top 5'), axis=1)
        st.write("Top 5 Predictions for Each User:")
        st.write(top5_predictions)
