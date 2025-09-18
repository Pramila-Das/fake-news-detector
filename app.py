import streamlit as st
import pickle
import pandas as pd

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
st.markdown("<p style='text-align: center; font-size:18px;'>Classify news as Real or Fake. Enter a headline/article or upload multiple headlines.</p>", unsafe_allow_html=True)
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
# 5. Input Options
# =========================
input_mode = st.radio("Choose input type:", ["Single News", "Batch Upload (.csv)"])

# -------------------------
# 5A. Single News Input
# -------------------------
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

# -------------------------
# 5B. Batch Upload
# -------------------------
else:
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

            # Download option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )

# =========================
# 6. Footer / Credits
# =========================
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ by <b>Pramila Das</b></p>", unsafe_allow_html=True)
