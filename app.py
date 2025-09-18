import streamlit as st
import pandas as pd
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
# Main App
# ---------------------------
st.title("Fake News Detector")

st.header("Contribute a News Article")

# Input fields
contrib_text = st.text_area("News Headline + Body")
contrib_label = st.radio("Label", ["Real", "Fake"])
contrib_email = st.text_input("Your Email (optional)")

if st.button("Submit"):
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

# ---------------------------
# Display Contributions
# ---------------------------
st.header("All Contributions")
if os.path.exists("user_contributions.csv"):
    contrib_df = pd.read_csv("user_contributions.csv")
    st.dataframe(contrib_df)
else:
    st.info("No contributions yet.")
