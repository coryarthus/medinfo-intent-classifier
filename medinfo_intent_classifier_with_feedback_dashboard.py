import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd

# Hardcoded sample intents
sample_intents = [
    {
        "intent_id": "medical_1",
        "category": "Medical",
        "subtype": "Safety",
        "definition": "Questions related to the potential risks and adverse effects associated with the drug.",
        "keywords": ["fatigue", "nausea", "dizziness", "side effect", "contraindication"],
        "response": "This drug may cause side effects such as fatigue, nausea, or dizziness. Please consult your healthcare provider."
    },
    {
        "intent_id": "nonmedical_1",
        "category": "Non-Medical",
        "subtype": "Product Complaint",
        "definition": "Issues related to product defects or complaints.",
        "keywords": ["complaint", "defective", "broken", "issue", "problem"],
        "response": "To report a product complaint, please contact our support team or use the online complaint form."
    },
    {
        "intent_id": "no_match",
        "category": "No Match",
        "subtype": "Unknown",
        "definition": "Fallback intent when no match is found.",
        "keywords": [],
        "response": "We could not identify a matching intent. Please rephrase your question or contact support."
    }
]

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

# Main input and classification
user_query = st.text_input("Enter your medical or non-medical query:")

if user_query:
    best_intent = None
    best_score = 0

    for intent in sample_intents:
        match_score = sum(1 for kw in intent["keywords"] if kw.lower() in user_query.lower())
        if match_score > best_score:
            best_score = match_score
            best_intent = intent

    if best_intent and best_intent["intent_id"] != "no_match":
        confidence = min(1.0, 0.5 + 0.1 * best_score)
    else:
        best_intent = sample_intents[-1]
        confidence = 0.5

    st.subheader("Classification Result")
    st.write(f"**Category:** {best_intent['category']}")
    st.write(f"**Subtype:** {best_intent['subtype']}")
    st.write(f"**Confidence:** {confidence:.2f}")
    st.write(f"**Response:** {best_intent['response']}")

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

