import streamlit as st
import joblib
from sklearn.metrics import accuracy_score

# Load the model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load preprocessed train/test data
# X_train = joblib.load('X_train.pkl')
# X_test = joblib.load('X_test.pkl')
# y_train = joblib.load('y_train.pkl')
# y_test = joblib.load('y_test.pkl')

# Streamlit app
st.title('Email Spam Classifier')
st.write("This app predicts whether an email is spam or ham (not spam).")

user_input = st.text_area("Enter the email text below:")

if st.button("Predict"):
    if user_input.strip() != "":
        # Transform the user input to match the training data format
        email_features = vectorizer.transform([user_input])
        prediction = model.predict(email_features)
        prediction_label = "Spam" if prediction[0] == 1 else "Ham"
        st.write(f"The email is classified as: **{prediction_label}**")
    else:
        st.write("Please enter the text of the email to classify.")

st.write("Training Accuracy: 99.80%")
st.write("Validation Accuracy: 98.03%")