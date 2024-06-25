import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd

# --- Load Model and Vectorizer with Version Check ---
try:
    with open('Random Forest Hate Spech Classification', 'rb') as f:
        clf = pickle.load(f)
except ValueError as e:
    if "incompatible dtype" in str(e):
        st.error("Model version mismatch. Retraining might be required.")
    else:
        st.error(f"Error loading the model: {e}")
    st.stop()  # Stop execution if there's an error

with open('RandomForest_vectorizer', 'rb') as f:
    _, vectorizer = pickle.load(f)

# --- Helper Function ---
def classify_text(text):
    text_vec = vectorizer.transform([text])
    prediction = clf.predict(text_vec)[0]

    labels = ['Race', 'Religion', 'Sexuality', 'Age']
    result = {label: int(prediction[i]) for i, label in enumerate(labels)}
    return pd.DataFrame([result])


# --- Streamlit App ---
st.title('Hate Speech Classification')
st.write("Enter a text to classify its potential for containing hate speech related to Race, Religion, Sexuality, or Age:")

input_text = st.text_area(label="Text Input", height=200)
if st.button('Classify'):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        result_df = classify_text(input_text)

        st.subheader('Classification Results')

        # Display results with color-coded styling
        for col in result_df.columns:
            value = result_df[col][0]
            if value == 1:
                color = "red"  # Highlight if predicted as hate speech
            else:
                color = "green"
            st.markdown(f'<span style="color:{color};">{col}: {value}</span>', unsafe_allow_html=True)
