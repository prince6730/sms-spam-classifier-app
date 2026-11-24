import streamlit as st
import pickle

# Load trained model and vectorizer
@st.cache_resource
def load_objects():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_objects()

# Page configuration
st.set_page_config(
    page_title="SMS & Email Spam Detector",
    page_icon="📩",
    layout="centered"
)

# UI
st.title("📩 SMS & Email Spam Detector")
st.write(
    "Detect whether an SMS or Email message is Spam or Not Spam using Machine Learning."
)

st.divider()

user_input = st.text_area(
    "Enter your SMS or Email message:",
    placeholder="Paste your SMS or email message here...",
    height=180
)

if st.button("Check Message", use_container_width=True):

    if user_input.strip():

        # Transform input using existing vectorizer
        input_vector = vectorizer.transform([user_input])

        # Prediction
        prediction = model.predict(input_vector)

        # Display result
        if prediction[0] == 1:
            st.error("🚨 SPAM MESSAGE DETECTED!")
            st.write("This message is classified as Spam.")

        elif prediction[0] == 0:
            st.success("✅ NOT SPAM")
            st.write("This message is classified as legitimate.")

        # Prediction confidence
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_vector)[0]

            st.subheader("Prediction Confidence")

            for label, probability in zip(model.classes_, probabilities):
                class_name = "Spam" if label == 1 else "Not Spam"
                st.write(f"**{class_name}:** {probability:.2%}")

    else:
        st.warning("Please enter an SMS or Email message first.")
