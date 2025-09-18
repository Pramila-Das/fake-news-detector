import streamlit as st
import pandas as pd
import pickle
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ------------------ App Header ------------------
st.title("📰 Fake News Detector")
st.sidebar.title("About Project")
st.sidebar.write("Dataset: `data.csv`")
st.sidebar.write("Model Accuracy: 97%")
st.sidebar.write("Built by Pramila Das")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    with open("fake_news_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ------------------ News Prediction ------------------
st.header("Check News Article")
news_text = st.text_area("Enter news text here:")

if st.button("Predict"):
    if not news_text.strip():
        st.warning("Please enter a news article to predict.")
    else:
        prediction = model.predict([news_text])[0]
        st.write("Prediction:", "Fake ❌" if prediction == 0 else "Real ✅")

# ------------------ Contribution Section ------------------
st.header("Contribute a News Article")

user_email = st.text_input("Your Email")
contrib_text = st.text_area("News Article Text")
label = st.selectbox("Label", ["Fake", "Real"])

csv_file = "user_contributions.csv"

# Ensure CSV exists
if not os.path.exists(csv_file):
    pd.DataFrame(columns=["Email", "News", "Label"]).to_csv(csv_file, index=False)

if st.button("Submit Contribution"):
    if not user_email or not contrib_text.strip():
        st.warning("Please enter both email and news text.")
    else:
        # Append contribution
        new_entry = pd.DataFrame([[user_email, contrib_text, label]], columns=["Email", "News", "Label"])
        new_entry.to_csv(csv_file, mode='a', header=False, index=False)
        st.success("Thank you for your contribution!")

        # Send email notification
        EMAIL_USER = os.getenv("EMAIL_USER")  # Or st.secrets["EMAIL_USER"]
        EMAIL_PASS = os.getenv("EMAIL_PASS")  # Or st.secrets["EMAIL_PASS"]

        if not EMAIL_USER or not EMAIL_PASS:
            st.warning("Email not sent. EMAIL_USER or EMAIL_PASS not set.")
        else:
            try:
                msg = MIMEMultipart()
                msg['From'] = EMAIL_USER
                msg['To'] = user_email
                msg['Subject'] = "Thank you for contributing to Fake News Detector"
                body = f"Hi,\n\nThank you for contributing a news article.\n\nYour submission:\n{contrib_text}\nLabel: {label}\n\nBest,\nFake News Detector Team"
                msg.attach(MIMEText(body, 'plain'))

                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(EMAIL_USER, EMAIL_PASS)
                server.send_message(msg)
                server.quit()

                st.success("Confirmation email sent!")
            except Exception as e:
                st.error(f"Error sending email: {e}")
