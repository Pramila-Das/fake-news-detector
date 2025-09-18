import streamlit as st
import pickle
import pandas as pd
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# =========================
# 1. App Configuration
# =========================
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# =========================
# 2. Title & Description
# =========================
st.markdown("<h1 style='text-align: center; color: #4B0082;'>📰 AI-Based Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Classify news as Real or Fake. Enter a headline/article or upload multiple headlines. Contribute to expand the dataset and optionally receive updates via email.</p>", unsafe_allow_html=True)
st.markdown("---")

# =========================
# 3. Sidebar Info
# =========================
st.sidebar.header("About Project")
st.sidebar.write("**Dataset:** `data.csv`")
st.sidebar.write("**Model Accuracy:** 97%")
st.sidebar.write("**Built by:** Pramila Das")
st.sidebar.write("**Instructions:** Enter at least 5 characters for reliable results.")

# =========================
# 4. Load Model
# =========================
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# 5. Helper function to send email
# =========================
def send_email(to_email, subject, body):
    sender_email = os.environ.get("EMAIL_USER")  # Set this in your environment variables
    password = os.environ.get("EMAIL_PASS")      # Gmail App password recommended
    if not sender_email or not password:
        st.warning("⚠️ Email not sent. EMAIL_USER or EMAIL_PASS not set in environment variables.")
        return

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")

# =========================
# 6. Tabs: Prediction and Contribution
# =========================
tab = st.tabs(["Predict", "Contribute"])

# -------------------------
# Prediction Tab
# -------------------------
with tab[0]:
    input_mode = st.radio("Choose input type:", ["Single News", "Batch Upload (.csv)"])

    if input_mode == "Single News":
        user_input = st.text_area("Enter news text here:", height=150)
        if st.button("Predict"):
            if not user_input.strip():
                st.warning("⚠️ Please enter some text to classify.")
            elif len(user_input.strip()) < 5:
                st.info("ℹ️ Input too short to classify reliably.")
            else:
                prediction = model.predict([user_input])[0]
                prediction_prob = model.predict_proba([user_input])[0]
                confidence = max(prediction_prob) * 100
                if prediction == 1:
                    st.success(f"✅ Prediction: Real News ({confidence:.2f}% confident)")
                    st.balloons()
                else:
                    st.error(f"❌ Prediction: Fake News ({confidence:.2f}% confident)")

    else:  # Batch mode
        uploaded_file = st.file_uploader("Upload a CSV file with a 'content' column:", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'content' not in df.columns:
                st.error("CSV must have a 'content' column.")
            else:
                predictions = model.predict(df['content'])
                probs = model.predict_proba(df['content'])
                confidence = [max(p)*100 for p in probs]
                df['Prediction'] = ["Real" if p==1 else "Fake" for p in predictions]
                df['Confidence (%)'] = confidence
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv'
                )

# -------------------------
# Contribution Tab
# -------------------------
with tab[1]:
    st.markdown("## 📝 Contribute to Dataset")
    with st.form("contribute_form"):
        news_text = st.text_area("Enter news article or headline:", height=150)
        label = st.radio("Label this news as:", ["Real", "Fake"])
        email = st.text_input("Enter your email (optional, for updates):")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if not news_text.strip():
                st.warning("⚠️ Please enter some text!")
            else:
                # Save to CSV
                contrib_file = "user_contributions.csv"
                new_entry = pd.DataFrame({
                    "content": [news_text],
                    "label": [1 if label == "Real" else 0],
                    "email": [email if email else "N/A"]
                })
                if os.path.exists(contrib_file):
                    new_entry.to_csv(contrib_file, mode='a', header=False, index=False)
                else:
                    new_entry.to_csv(contrib_file, mode='w', header=True, index=False)

                st.success("Thank you! Your news article has been added to the contributions dataset.")

                # Send email if provided
                if email.strip():
                    send_email(
                        to_email=email,
                        subject="Thank you for contributing!",
                        body=f"Hi,\n\nThank you for contributing to the Fake News dataset.\n\nYour submission: {news_text}\nLabel: {label}\n\n– Pramila"
                    )

    # Display stats
    try:
        contrib_df = pd.read_csv("user_contributions.csv")
        st.info(f"Total contributions: {len(contrib_df)} | Real: {sum(contrib_df['label']==1)} | Fake: {sum(contrib_df['label']==0)}")
    except FileNotFoundError:
        st.info("No contributions yet. Be the first to add news!")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ by <b>Pramila Das</b></p>", unsafe_allow_html=True)
