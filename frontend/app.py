"""
==========================================================
STREAMLIT FRONTEND — app.py (Standalone Version)
==========================================================

WHAT THIS FILE DOES:
--------------------
This is the COMPLETE frontend for the Customer Churn Prediction System.
It runs STANDALONE — no separate backend server needed!

It:
1. Loads the ML model directly (from models/ folder)
2. Handles Supabase authentication (signup/login)
3. Lets users enter data manually or upload CSV
4. Makes predictions and shows visualizations
5. Stores predictions in Supabase database

DEPLOYMENT:
-----------
This is designed for Streamlit Cloud deployment.
The model is loaded directly from the repo — no FastAPI backend needed.

HOW TO RUN LOCALLY:
-------------------
    cd frontend
    pip install -r requirements.txt
    streamlit run app.py
==========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
import json
from pathlib import Path

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────
# Supabase credentials — read from Streamlit secrets (cloud) or .env (local)
def get_config(key, default=""):
    """Read config from Streamlit secrets first, then env vars, then default."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

SUPABASE_URL = get_config("SUPABASE_URL")
SUPABASE_KEY = get_config("SUPABASE_KEY")


# ──────────────────────────────────────────────────────────
# ML MODEL — Load directly (no backend needed)
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Load the trained ML model, feature names, and label encoders.
    Uses @st.cache_resource so the model is loaded ONCE and reused.
    """
    # Find the models directory (works both locally and on Streamlit Cloud)
    # Locally: frontend/ -> ../models/
    # Streamlit Cloud: the repo root has models/
    current_dir = Path(__file__).parent
    models_dir = current_dir.parent / "models"

    if not models_dir.exists():
        return None, None, None

    try:
        model = joblib.load(models_dir / "model.pkl")
        feature_names = joblib.load(models_dir / "feature_names.pkl")
        label_encoders = joblib.load(models_dir / "label_encoders.pkl")
        return model, feature_names, label_encoders
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None, None


def predict_single(customer_data: dict, model, feature_names, label_encoders):
    """
    Make a churn prediction for a single customer.
    This replaces the backend API call — runs the model directly.
    """
    try:
        df = pd.DataFrame([customer_data])

        # Encode categorical columns
        for col, encoder in label_encoders.items():
            if col in df.columns:
                known_classes = set(encoder.classes_)
                df[col] = df[col].apply(
                    lambda x: encoder.transform([x])[0] if x in known_classes else -1
                )

        # Reorder columns to match training order
        df = df[feature_names]

        # Predict
        prediction = int(model.predict(df)[0])
        probabilities = model.predict_proba(df)[0]
        confidence = float(max(probabilities))

        prediction_label = "Will Churn" if prediction == 1 else "Will Not Churn"

        if prediction == 1:
            message = (
                f"⚠️ This customer is likely to churn "
                f"(confidence: {confidence:.1%}). "
                f"Consider offering retention incentives."
            )
        else:
            message = (
                f"✅ This customer is likely to stay "
                f"(confidence: {confidence:.1%}). "
                f"Keep up the good service!"
            )

        return {
            "prediction": prediction,
            "prediction_label": prediction_label,
            "confidence": round(confidence, 4),
            "message": message
        }
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────
# SUPABASE — Authentication & Database
# ──────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    """Get or create a Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None


def store_prediction_to_db(supabase_client, user_id, customer_data, prediction, prediction_label, confidence):
    """Store a prediction in Supabase."""
    if supabase_client is None:
        return
    try:
        supabase_client.table("predictions").insert({
            "user_id": user_id,
            "customer_data": customer_data,
            "prediction": prediction,
            "prediction_label": prediction_label,
            "confidence": confidence
        }).execute()
    except Exception as e:
        st.warning(f"Could not save prediction: {e}")


# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { color: white !important; margin-bottom: 0.3rem; }
    .main-header p { color: rgba(255,255,255,0.85); font-size: 1.1rem; }

    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        border: 1px solid #3d3d5c;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h3 { color: #a0a0cc; font-size: 0.85rem; margin-bottom: 0.3rem; }
    .metric-card .value { font-size: 1.8rem; font-weight: bold; color: #fafafa; }

    .prediction-churn {
        background: linear-gradient(135deg, #ff6b6b22, #ff535322);
        border: 2px solid #ff6b6b;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .prediction-no-churn {
        background: linear-gradient(135deg, #51cf6622, #2ed57322);
        border: 2px solid #51cf66;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .sidebar-info {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        border: 1px solid #3d3d5c;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "predictions" not in st.session_state:
    st.session_state.predictions = []


# ──────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────
model, feature_names, label_encoders = load_model()


# ──────────────────────────────────────────────────────────
# SIDEBAR — Authentication
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 Authentication")

    supabase_client = get_supabase()

    if supabase_client is None:
        st.warning(
            "⚠️ Supabase not configured.\n\n"
            "The app works without login — predictions just won't be saved.\n\n"
            "See PROJECT_GUIDE.md Section 5 to set up Supabase."
        )
        st.session_state.authenticated = False
    else:
        if st.session_state.authenticated:
            st.success(f"✅ Logged in as:\n{st.session_state.user_email}")
            if st.button("🚪 Logout", use_container_width=True):
                try:
                    supabase_client.auth.sign_out()
                except Exception:
                    pass
                st.session_state.authenticated = False
                st.session_state.user_email = None
                st.session_state.user_id = None
                st.session_state.access_token = None
                st.rerun()
        else:
            auth_tab = st.radio("Choose:", ["Login", "Sign Up"], horizontal=True)
            email = st.text_input("📧 Email", placeholder="you@example.com")
            password = st.text_input("🔑 Password", type="password", placeholder="Min 6 characters")

            if auth_tab == "Login":
                if st.button("🔓 Login", use_container_width=True):
                    if not email or not password:
                        st.error("Please enter email and password.")
                    else:
                        try:
                            response = supabase_client.auth.sign_in_with_password({
                                "email": email,
                                "password": password
                            })
                            st.session_state.authenticated = True
                            st.session_state.user_email = response.user.email
                            st.session_state.user_id = str(response.user.id)
                            st.session_state.access_token = response.session.access_token
                            st.success("✅ Login successful!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Login failed: {str(e)}")
            else:
                if st.button("📝 Sign Up", use_container_width=True):
                    if not email or not password:
                        st.error("Please enter email and password.")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        try:
                            supabase_client.auth.sign_up({
                                "email": email,
                                "password": password
                            })
                            st.success(
                                "✅ Sign up successful!\n\n"
                                "Check your email to confirm your account, then log in."
                            )
                        except Exception as e:
                            st.error(f"❌ Sign up failed: {str(e)}")

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-info">
        <h4>ℹ️ About</h4>
        <p style="font-size: 0.85rem; color: #a0a0cc;">
            This app predicts if a customer will churn using a Random Forest ML model.
            Enter customer data or upload a CSV to get predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# MAIN HEADER
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔮 Customer Churn Predictor</h1>
    <p>Predict which customers are likely to leave — powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# STATUS CARDS
# ──────────────────────────────────────────────────────────
cols = st.columns(3)
with cols[0]:
    model_status = "✅ Loaded" if model is not None else "❌ Not Found"
    st.markdown(f"""
    <div class="metric-card">
        <h3>ML Model</h3>
        <div class="value">{model_status}</div>
    </div>
    """, unsafe_allow_html=True)
with cols[1]:
    db_status = "✅ Connected" if supabase_client is not None else "⚠️ Not Set"
    st.markdown(f"""
    <div class="metric-card">
        <h3>Database</h3>
        <div class="value">{db_status}</div>
    </div>
    """, unsafe_allow_html=True)
with cols[2]:
    auth_status = f"✅ {st.session_state.user_email}" if st.session_state.authenticated else "🔓 Guest"
    st.markdown(f"""
    <div class="metric-card">
        <h3>User</h3>
        <div class="value" style="font-size: 1rem;">{auth_status}</div>
    </div>
    """, unsafe_allow_html=True)

if model is None:
    st.error(
        "❌ **Model not found!** Make sure `models/model.pkl` exists.\n\n"
        "Run: `cd model_training && python train.py`"
    )
    st.stop()

st.markdown("---")


# ──────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Single Prediction",
    "📁 CSV Upload (Batch)",
    "📊 Visualizations",
    "📜 History"
])


# ──────────────────────────────────────────────────────────
# TAB 1: Single Prediction
# ──────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📝 Predict Churn for One Customer")
    st.markdown("Fill in the customer details below and click **Predict**.")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        senior_citizen = st.selectbox(
            "👴 Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        tenure = st.slider(
            "📅 Tenure (months)", min_value=0, max_value=72, value=12,
            help="How many months has this customer been with our company?"
        )
        monthly_charges = st.number_input(
            "💰 Monthly Charges ($)", min_value=0.0, max_value=200.0,
            value=29.85, step=0.01
        )

    with col2:
        total_charges = st.number_input(
            "💵 Total Charges ($)", min_value=0.0, max_value=10000.0,
            value=358.20, step=0.01
        )
        contract_type = st.selectbox(
            "📋 Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        internet_service = st.selectbox(
            "🌐 Internet Service", ["DSL", "Fiber optic", "No"]
        )
        payment_method = st.selectbox(
            "💳 Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
        )

    st.markdown("")
    if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):
        customer_data = {
            "gender": gender,
            "senior_citizen": senior_citizen,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method
        }

        with st.spinner("🔄 Making prediction..."):
            result = predict_single(customer_data, model, feature_names, label_encoders)

        if "error" in result:
            st.error(f"❌ Prediction failed: {result['error']}")
        else:
            st.session_state.predictions.append(result)

            # Save to Supabase if authenticated
            if st.session_state.authenticated and supabase_client:
                store_prediction_to_db(
                    supabase_client,
                    st.session_state.user_id,
                    customer_data,
                    result["prediction"],
                    result["prediction_label"],
                    result["confidence"]
                )

            # Display result
            if result["prediction"] == 1:
                st.markdown(f"""
                <div class="prediction-churn">
                    <h2>⚠️ CHURN RISK DETECTED</h2>
                    <p style="font-size: 1.2rem;">{result['message']}</p>
                    <p style="font-size: 0.9rem; opacity: 0.7;">Confidence: {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-no-churn">
                    <h2>✅ CUSTOMER WILL STAY</h2>
                    <p style="font-size: 1.2rem;">{result['message']}</p>
                    <p style="font-size: 0.9rem; opacity: 0.7;">Confidence: {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("🔍 View raw prediction data"):
                st.json(result)


# ──────────────────────────────────────────────────────────
# TAB 2: CSV Upload (Batch Prediction)
# ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📁 Upload CSV for Batch Predictions")
    st.markdown(
        "Upload a CSV file with customer data. Required columns:\n"
        "`gender`, `senior_citizen`, `tenure`, `monthly_charges`, `total_charges`, "
        "`contract_type`, `internet_service`, `payment_method`"
    )

    # Show sample format
    st.markdown("**Need a sample?** Here's the expected format:")
    sample_df = pd.DataFrame([
        {
            "gender": "Male", "senior_citizen": 0, "tenure": 12,
            "monthly_charges": 29.85, "total_charges": 358.20,
            "contract_type": "Month-to-month", "internet_service": "DSL",
            "payment_method": "Electronic check"
        },
        {
            "gender": "Female", "senior_citizen": 0, "tenure": 34,
            "monthly_charges": 56.95, "total_charges": 1936.30,
            "contract_type": "One year", "internet_service": "Fiber optic",
            "payment_method": "Mailed check"
        }
    ])
    st.dataframe(sample_df, use_container_width=True)

    uploaded_file = st.file_uploader("📂 Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded {len(df)} rows**")
            st.dataframe(df.head(10), use_container_width=True)

            required_cols = [
                "gender", "senior_citizen", "tenure", "monthly_charges",
                "total_charges", "contract_type", "internet_service", "payment_method"
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
            else:
                if st.button("🔮 Predict All Customers", use_container_width=True, type="primary"):
                    predictions = []
                    churn_count = 0

                    progress_bar = st.progress(0)
                    total = len(df)

                    for i, (_, row) in enumerate(df[required_cols].iterrows()):
                        customer_data = row.to_dict()
                        result = predict_single(customer_data, model, feature_names, label_encoders)

                        if "error" not in result:
                            predictions.append(result)
                            if result["prediction"] == 1:
                                churn_count += 1

                            # Save to Supabase
                            if st.session_state.authenticated and supabase_client:
                                store_prediction_to_db(
                                    supabase_client,
                                    st.session_state.user_id,
                                    customer_data,
                                    result["prediction"],
                                    result["prediction_label"],
                                    result["confidence"]
                                )

                        progress_bar.progress((i + 1) / total)

                    st.session_state.predictions.extend(predictions)

                    # Summary
                    st.markdown("### 📊 Results Summary")
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Total Customers", len(predictions))
                    with m2:
                        st.metric("Predicted to Churn", churn_count)
                    with m3:
                        rate = (churn_count / len(predictions) * 100) if predictions else 0
                        st.metric("Churn Rate", f"{rate:.1f}%")

                    # Add predictions to dataframe
                    df["Prediction"] = [p["prediction_label"] for p in predictions]
                    df["Confidence"] = [f"{p['confidence']:.1%}" for p in predictions]

                    st.markdown("### 📋 Detailed Results")
                    st.dataframe(df, use_container_width=True)

                    # Download
                    csv_output = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results CSV",
                        csv_output,
                        "churn_predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"❌ Error reading CSV: {str(e)}")


# ──────────────────────────────────────────────────────────
# TAB 3: Visualizations
# ──────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📊 Data Visualizations")

    if not st.session_state.predictions:
        st.info(
            "💡 Make some predictions first! Go to the **Single Prediction** or "
            "**CSV Upload** tab to generate data for visualizations."
        )
    else:
        predictions = st.session_state.predictions

        # Pie Chart
        st.markdown("#### 🥧 Prediction Distribution")
        churn_count = sum(1 for p in predictions if p["prediction"] == 1)
        no_churn_count = len(predictions) - churn_count

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        fig1.patch.set_facecolor("#0E1117")
        ax1.set_facecolor("#0E1117")

        colors = ["#51cf66", "#ff6b6b"]
        sizes = [no_churn_count, churn_count]
        labels = [f"Will Stay ({no_churn_count})", f"Will Churn ({churn_count})"]

        if all(s > 0 for s in sizes):
            wedges, texts, autotexts = ax1.pie(
                sizes, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                textprops={"color": "white", "fontsize": 10}
            )
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")
        else:
            ax1.pie(
                sizes, labels=labels, colors=colors,
                startangle=90, textprops={"color": "white", "fontsize": 10}
            )

        ax1.set_title("Churn Prediction Results", color="white", fontsize=14, fontweight="bold")
        st.pyplot(fig1)
        plt.close(fig1)

        # Confidence Histogram
        st.markdown("#### 📈 Confidence Distribution")
        confidences = [p["confidence"] for p in predictions]

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        fig2.patch.set_facecolor("#0E1117")
        ax2.set_facecolor("#1E1E2E")

        ax2.hist(confidences, bins=10, color="#667eea", edgecolor="#764ba2", alpha=0.8)
        ax2.set_xlabel("Confidence Score", color="white", fontsize=11)
        ax2.set_ylabel("Count", color="white", fontsize=11)
        ax2.set_title("Model Confidence Distribution", color="white", fontsize=14, fontweight="bold")
        ax2.tick_params(colors="white")
        ax2.spines["bottom"].set_color("#3d3d5c")
        ax2.spines["left"].set_color("#3d3d5c")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        st.pyplot(fig2)
        plt.close(fig2)

        # Bar Chart
        st.markdown("#### 📊 Predictions Breakdown")
        pred_labels = [p["prediction_label"] for p in predictions]

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        fig3.patch.set_facecolor("#0E1117")
        ax3.set_facecolor("#1E1E2E")

        categories = ["Will Not Churn", "Will Churn"]
        counts = [pred_labels.count("Will Not Churn"), pred_labels.count("Will Churn")]
        bar_colors = ["#51cf66", "#ff6b6b"]

        bars = ax3.bar(categories, counts, color=bar_colors, edgecolor="#1E1E2E", width=0.5)
        for bar, count in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(count), ha="center", color="white", fontsize=12, fontweight="bold")

        ax3.set_ylabel("Number of Predictions", color="white", fontsize=11)
        ax3.set_title("Prediction Counts", color="white", fontsize=14, fontweight="bold")
        ax3.tick_params(colors="white")
        ax3.spines["bottom"].set_color("#3d3d5c")
        ax3.spines["left"].set_color("#3d3d5c")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        st.pyplot(fig3)
        plt.close(fig3)


# ──────────────────────────────────────────────────────────
# TAB 4: Prediction History
# ──────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 📜 Prediction History")

    if not st.session_state.authenticated:
        st.info(
            "🔐 **Login required** to view saved prediction history.\n\n"
            "Use the sidebar to log in with your Supabase account.\n\n"
            "Without login, only this session's predictions are shown below."
        )
    else:
        # Fetch from Supabase
        if supabase_client:
            try:
                result = (
                    supabase_client.table("predictions")
                    .select("*")
                    .eq("user_id", st.session_state.user_id)
                    .order("created_at", desc=True)
                    .limit(50)
                    .execute()
                )
                if result.data:
                    st.markdown(f"**Showing {len(result.data)} saved predictions**")
                    history_df = pd.DataFrame(result.data)
                    st.dataframe(history_df, use_container_width=True)
                else:
                    st.info("No saved predictions yet. Make some predictions to see them here!")
            except Exception as e:
                st.warning(f"Could not fetch history: {e}")

    # Show session predictions
    if st.session_state.predictions:
        st.markdown("#### 📋 This Session's Predictions")
        session_df = pd.DataFrame(st.session_state.predictions)
        st.dataframe(session_df, use_container_width=True)
    elif not st.session_state.authenticated:
        st.info("No predictions made yet in this session.")


# ──────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "🔮 Customer Churn Prediction System | Built with Streamlit + Scikit-learn + Supabase"
    "</p>",
    unsafe_allow_html=True
)
