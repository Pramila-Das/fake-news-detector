import streamlit as st
import pandas as pd
import pickle
import smtplib
from email.message import EmailMessage
import os

# ---------------------------
# Sidebar Info
# ---------------------------
st.sidebar.title("About Project")
st.sidebar.write("Dataset: `data.csv`")
st.sidebar.write("Model Accuracy: 97%")
st.sidebar.write("Built by Pramila Das")

# ---------------------------
# Load Model
# ---------------------------
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Main App
# ---------------------------
st.title("Fake News Detector")

# Input field for prediction
st.header("Check if a News Article is Fake or Real")
news_text = st.text_area("Enter the news headline + body here:")

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        prediction = model.predict([news_text])[0]
        result = "Real" if prediction == 1 else "Fake"
        st.success(f"The news is predicted as: **{result}**")

# Contribution section
st.header("Contribute a News Article")

contrib_text = st.text_area("News Headline + Body", key="contrib_text")
contrib_label = st.radio("Label", ["Real", "Fake"], key="contrib_label")
contrib_email = st.text_input("Your Email (optional)", key="contrib_email")

if st.button("Submit Contribution"):
    if contrib_text.strip() == "":
        st.warning("Please enter the news content.")
    elif contrib_email.strip() == "":
        st.warning("Please enter your email to receive a confirmation.")
    else:
        # 1. Save contribution to CSV
        new_data = pd.DataFrame({
            "content": [contrib_text],
            "label": [1 if contrib_label == "Real" else 0],
            "email": [contrib_email]
        })

        if not os.path.exists("user_contributions.csv"):
            new_data.to_csv("user_contributions.csv", index=False)
        else:
            new_data.to_csv("user_contributions.csv", mode='a', header=False, index=False)

        st.success("Thank you for contributing!")

        # 2. Send thank-you email
        try:
            EMAIL_USER = os.getenv("EMAIL_USER")  # Your Gmail
            EMAIL_PASS = os.getenv("EMAIL_PASS")  # Your App Password

            msg = EmailMessage()
            msg['Subject'] = "Thank you for contributing to Fake News Detector!"
            msg['From'] = EMAIL_USER
            msg['To'] = contrib_email
            msg.set_content(f"""
Hi!

Thank you for contributing a news article labeled as '{contrib_label}' to our dataset.

We really appreciate your help in improving our Fake News Detector project.

Best regards,
Pramila Das
""")

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(EMAIL_USER, EMAIL_PASS)
                smtp.send_message(msg)

            st.info("A confirmation email has been sent to your email!")
        except Exception as e:
            st.error(f"Error sending email: {e}")
