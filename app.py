import streamlit as st
import pickle

# 1. App title
st.title("📰 AI-Based Fake News Detector")
st.write("Paste a news headline or article below to check if it's Real or Fake.")

# 2. Sidebar info
st.sidebar.title("About Project")
st.sidebar.write("**Dataset:** `data.csv`")
st.sidebar.write("**Model Accuracy:** 97%")
st.sidebar.write("**Built by:** Pramila Das")

# 3. Load trained model
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

# 4. User input
user_input = st.text_area("Enter news text:")

# 5. Input validation & prediction
if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to classify.")
    elif len(user_input.strip()) < 5:
        st.info("ℹ️ Input too short to classify reliably.")
    else:
        prediction = model.predict([user_input])[0]
        if prediction == 1:
            st.success("✅ Prediction: Real News")
        else:
            st.error("❌ Prediction: Fake News")

# 6. Footer / credits
st.markdown("---")
st.markdown("Made with ❤️ by Pramila Das")
