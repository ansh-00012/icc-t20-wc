import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# App settings
st.set_page_config(page_title="T20 Match Winner Prediction", page_icon="🏏", layout="wide")

# Sidebar menu
menu = st.sidebar.radio("Menu", ["Train Model", "Predict Winner"])

MODEL_FILE = "t20_model.pkl"

# ============================
# TRAINING SECTION
# ============================
if menu == "Train Model":
    st.title("🏏 T20 Match Winner Prediction - Train Model")
    st.subheader("Upload training data (CSV)")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview", df.head())

        if "winner" not in df.columns:
            st.error("CSV must have a 'winner' column.")
        else:
            if st.button("Train Model"):
                # Remove missing rows
                df = df.dropna()

                # Encode features
                X = pd.get_dummies(df.drop("winner", axis=1))
                y = df["winner"]

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Train Random Forest model
                model = RandomForestClassifier()
                model.fit(X_train, y_train)

                

                # Save model and feature names
                with open(MODEL_FILE, "wb") as f:
                    pickle.dump((model, X.columns.tolist()), f)

                st.success(f"✅ Model trained successfully")
    else:
        st.info("Please upload a training CSV file.")

# ============================
# PREDICTION SECTION
# ============================
elif menu == "Predict Winner":
    st.title("🏏 Predict T20 Match Winner")

    if not os.path.exists(MODEL_FILE):
        st.error("❌ Please train the model first in 'Train Model' menu.")
    else:
        with open(MODEL_FILE, "rb") as f:
            model, feature_names = pickle.load(f)

        # User inputs
        team1 = st.text_input("Enter Team 1")
        team2 = st.text_input("Enter Team 2")
        venue = st.text_input("Enter Venue")
        stage = st.text_input("Enter Stage (e.g., Group Stage, Semi Final, Final)")

        if st.button("Predict Winner"):
            if team1 and team2 and venue and stage:
                # Create dataframe from user input
                data = pd.DataFrame(
                    [[team1, team2, venue, stage]],
                    columns=["team1", "team2", "venue", "stage"]
                )

                # One-hot encode
                data = pd.get_dummies(data)

                # Align columns with training features
                for col in feature_names:
                    if col not in data.columns:
                        data[col] = 0
                data = data[feature_names]

                # Predict winner
                winner = model.predict(data)[0]
                st.success(f"🏆 Predicted Winner: {winner}")
            else:
                st.warning("⚠ Please fill in all fields before predicting.")
