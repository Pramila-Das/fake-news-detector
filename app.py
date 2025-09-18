import streamlit as st
import pickle

# =========================
# 1. App Configuration
# =========================
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# =========================
# 2. App Title & Description
# =========================
st.markdown("<h1 style='text-align: center; color: #4B0082;'>📰 AI-Based Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Paste a news headline or article below to check if it's Real or Fake.</p>", unsafe_allow_html=True)
st.markdown("---")

# =========================
# 3. Sidebar Info
# =========================
st.sidebar.header("About Project")
st.sidebar.write("**Dataset:** `data.csv`")
st.sidebar.write("**Model Accuracy:** 97%")
st.sidebar.write("**Built by:** Pramila Das")
st.sidebar.write("**Instructions:** Enter at least 5 characters for a reliable prediction.")

# =========================
# 4. Load Trained Model
# =========================
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# 5. User Input
# =========================
user_input = st.text_area("Enter news text here:", height=150)

# =========================
# 6. Prediction Button & Logic
# =========================
if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to classify.")
    elif len(user_input.strip()) < 5:
        st.info("ℹ️ Input too short to classify reliably.")
    else:
        prediction = model.predict([user_input])[0]
        prediction_prob = model.predict_proba([user_input])[0]
        confidence = max(prediction_prob) * 100  # confidence %

        if prediction == 1:
            st.success(f"✅ Prediction: Real News ({confidence:.2f}% confident)")
            st.balloons()  # Fun effect for real news
        else:
            st.error(f"❌ Prediction: Fake News ({confidence:.2f}% confident)")

# =========================
# 7. Footer / Credits
# =========================
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ by <b>Pramila Das</b></p>", unsafe_allow_html=True)
