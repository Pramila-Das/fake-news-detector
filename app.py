import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📰 Fake News Detection App")

# Sidebar
st.sidebar.title("About Project")
st.sidebar.write("Dataset: `data.csv`")
st.sidebar.write("Model Accuracy: 97%")
st.sidebar.write("Built by **Pramila Das**")

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_area("Enter a news headline or article:", "")

if st.button("Check"):
    if user_input.strip() != "":
        # Predict
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]
        confidence = round(max(probabilities) * 100, 2)
        result = "REAL" if prediction == 1 else "FAKE"

        # Display result
        if prediction == 1:
            st.success(f"✅ This news looks REAL (Confidence: {confidence}%)")
        else:
            st.error(f"🚨 This news looks FAKE (Confidence: {confidence}%)")

        # Save to history
        st.session_state.history.append({
            "News": user_input,
            "Prediction": result,
            "Confidence (%)": confidence
        })
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Show history
if st.session_state.history:
    st.subheader("📝 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.table(history_df)

    # Download button
    csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download History as CSV",
        data=csv,
        file_name='prediction_history.csv',
        mime='text/csv'
    )
