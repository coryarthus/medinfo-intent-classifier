import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import json
import os
from datetime import datetime

# Load the Excel file
df = pd.read_excel("MedInfo Agent -List of Intents.xlsx", engine="openpyxl", header=None)

# Extract intents
medical_intents = []
for i in range(13, 21):
    row = df.iloc[i]
    intent = {
        "intent_id": f"medical_{row[1]}",
        "category": "Medical",
        "subtype": row[2],
        "definition": row[3],
        "keywords": row[4].split(",") if pd.notna(row[4]) else [],
        "response": None
    }
    medical_intents.append(intent)

nonmedical_intents = []
for i in range(26, 39):
    row = df.iloc[i]
    intent = {
        "intent_id": f"nonmedical_{row[1]}",
        "category": "Non-Medical",
        "subtype": row[2],
        "definition": row[5],
        "keywords": row[6].split(",") if pd.notna(row[6]) else [],
        "response": row[7] if pd.notna(row[7]) else ""
    }
    nonmedical_intents.append(intent)

all_intents = medical_intents + nonmedical_intents

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Precompute intent embeddings
intent_embeddings = []
for intent in all_intents:
    text = f"{intent['subtype']}. {intent['definition']}. {' '.join(intent['keywords'])}"
    embedding = model.encode(text, convert_to_tensor=True)
    intent_embeddings.append((intent, embedding))

# Streamlit UI
st.title("MedInfo Intent Classifier")


# Sidebar for feedback review
if st.sidebar.checkbox("Show Feedback Review Dashboard"):
    st.sidebar.write("### Feedback Review")
    if os.path.exists("feedback_log.json"):
        with open("feedback_log.json", "r") as f:
            feedback_data = json.load(f)
        feedback_df = pd.DataFrame(feedback_data)
        st.sidebar.dataframe(feedback_df)
    else:
        st.sidebar.write("No feedback submitted yet.")


user_query = st.text_input("Enter your medical or non-medical query:")

if user_query:
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    best_intent = None
    best_score = -1

    for intent, emb in intent_embeddings:
        score = util.pytorch_cos_sim(query_embedding, emb).item()
        if score > best_score:
            best_score = score
            best_intent = intent

    st.subheader("Classification Result")
    st.write(f"**Category:** {best_intent['category']}")
    st.write(f"**Subtype:** {best_intent['subtype']}")
    st.write(f"**Confidence:** {best_score:.2f}")
    if best_intent['category'] == "Non-Medical":
        st.write(f"**Response:** {best_intent['response']}")
    else:
        st.write("**Response:** This is a medical inquiry. Response will be generated based on product labels.")

# Feedback option
    if st.checkbox("Flag this classification as incorrect"):
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "predicted_intent_id": best_intent["intent_id"],
            "category": best_intent["category"],
            "subtype": best_intent["subtype"],
            "confidence": round(confidence, 2)
        }
        feedback_log = []
        if os.path.exists("feedback_log.json"):
            with open("feedback_log.json", "r") as f:
                feedback_log = json.load(f)
        feedback_log.append(feedback_entry)
        with open("feedback_log.json", "w") as f:
            json.dump(feedback_log, f, indent=2)
        st.success("Thank you! Your feedback has been recorded.")
