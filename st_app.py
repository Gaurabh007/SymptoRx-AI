import streamlit as st
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.preprocessing import LabelEncoder
import re

# ----------- Load model and data ------------
@st.cache_resource
def load_resources():
    model = pickle.load(open('randomforest.pkl','rb'))
    train_df = pd.read_csv('data/disease_prediction_dataset/Training.csv')
    train_df.columns = train_df.columns.str.replace('_',' ')
    all_symptoms = train_df.drop(columns=['prognosis']).columns.tolist()
    le = LabelEncoder()
    le.fit(train_df['prognosis'])
    med_df = pd.read_csv('disease2med.csv')
    return model, all_symptoms, le, med_df

model, all_symptoms, le, med_df = load_resources()

# Normalize/clean med_df columns for case-insensitive matching (if not done already)
for col in ['prognosis', 'disease_name']:
    med_df[col] = med_df[col].astype(str).str.strip().str.lower()

# ------------------- Utility Functions -------------------

def predict_disease(user_symptoms, all_symptoms, model, le, min_symptoms=4):
    """
    Returns: (predicted_disease:str or None, message:str)
    """
    if len(user_symptoms) < min_symptoms:
        return None, "Insufficient symptoms provided for a reliable prediction. Please select at least 4 symptoms."
    user_input = np.zeros((1, len(all_symptoms)), dtype=int)
    for idx, s in enumerate(all_symptoms):
        if s in user_symptoms:
            user_input[0, idx] = 1
    best_class = model.predict(user_input)[0]
    disease = le.inverse_transform([best_class])[0]
    return disease, f"Predicted disease: {disease}"

def fetch_medicines_for_disease(med_df, disease_query):
    """
    Matches disease_query to both columns, returns unique med_names.
    """
    disease_query = disease_query.strip().lower()
    mask = (med_df['prognosis'] == disease_query) | (med_df['disease_name'] == disease_query)
    meds = med_df.loc[mask, 'med_name'].dropna().unique().tolist()
    return meds

# ---------------------- Streamlit UI -----------------------
st.title("Disease Prediction & Medicine Recommendation")

st.markdown("""
Select your symptoms from the dropdown below, one by one (type to filter).<br>
You must select at least 4 symptoms before you can get a prediction.<br>
After prediction, you will see 3 sample medicines for the predicted disease.
""", unsafe_allow_html=True)

if 'user_symptom_list' not in st.session_state:
    st.session_state.user_symptom_list = []

# Only offer symptoms that haven't already been selected
available_symptoms = [s for s in all_symptoms if s not in st.session_state.user_symptom_list]

single_symptom = st.selectbox(
    "Add a symptom (type to search):",
    [""] + sorted(available_symptoms),
    key="symptom_input"
)

if st.button("Add Symptom"):
    if single_symptom and single_symptom not in st.session_state.user_symptom_list:
        st.session_state.user_symptom_list.append(single_symptom)

if st.session_state.user_symptom_list:
    st.write(f"**Current symptoms ({len(st.session_state.user_symptom_list)}):**")
    st.write(", ".join(st.session_state.user_symptom_list))
    if st.button("Clear all symptoms"):
        st.session_state.user_symptom_list = []

if st.button("Predict Disease"):
    if len(st.session_state.user_symptom_list) < 4:
        st.error("Please select at least 4 symptoms for a reliable prediction.")
    else:
        # Predict disease
        disease, msg = predict_disease(
            st.session_state.user_symptom_list,
            all_symptoms,
            model,
            le
        )
        st.success(msg)
        # Medicine recommendation
        if disease:
            # Also try a loose, lowercased match
            disease_query = disease.strip().lower()
            meds_list = fetch_medicines_for_disease(med_df, disease_query)
            if meds_list:
                display_meds = random.sample(meds_list, min(3, len(meds_list)))
                st.info("Sample recommended medicines:")
                for med in display_meds:
                    st.write(f"- {med}")
            else:
                st.warning(f"No medicines found for '{disease}'.")

