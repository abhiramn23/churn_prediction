"""
==========================================================
STREAMLIT FRONTEND — app.py
==========================================================

WHAT THIS FILE DOES:
--------------------
This is the FRONTEND of our Customer Churn Prediction System.
It creates a beautiful web interface where users can:

1. Sign up / Log in using Supabase authentication
2. Enter customer data manually (form)
3. Upload a CSV file for batch predictions
4. See prediction results with visualizations
5. View their prediction history

WHAT IS STREAMLIT?
------------------
Streamlit turns Python scripts into web apps — no HTML/CSS/JS needed!
You write Python, and Streamlit renders it as a web page.

HOW TO RUN:
-----------
    cd frontend
    pip install -r requirements.txt
    streamlit run app.py

Then open http://localhost:8501 in your browser.
==========================================================
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────
# The URL of our FastAPI backend
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


# ──────────────────────────────────────────────────────────
# HELPER: Initialize Supabase client
# ──────────────────────────────────────────────────────────
def get_supabase():
    """Get or create a Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────
# HELPER: Make API calls to the backend
# ──────────────────────────────────────────────────────────
def call_api(endpoint: str, method: str = "GET", data: dict = None, token: str = None) -> dict:
    """
    Make an HTTP request to the backend API.

    Args:
        endpoint: API endpoint (e.g., "/predict")
        method: HTTP method ("GET" or "POST")
        data: Request body (for POST requests)
        token: Supabase auth token (optional)

    Returns:
        Response JSON as a dictionary
    """
    url = f"{API_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}

    # Add auth token if available
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        if method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=30)
        else:
            response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            error_detail = response.json().get("detail", response.text)
            return {"success": False, "error": f"API Error ({response.status_code}): {error_detail}"}

    except requests.ConnectionError:
        return {
            "success": False,
            "error": (
                "❌ Cannot connect to the backend API.\n\n"
                f"Make sure the backend is running at: {API_URL}\n\n"
                "Start it with: `cd backend && uvicorn app.main:app --reload`"
            )
        }
    except requests.Timeout:
        return {"success": False, "error": "❌ API request timed out. Please try again."}
    except Exception as e:
        return {"success": False, "error": f"❌ Unexpected error: {str(e)}"}


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
# CUSTOM CSS for better styling
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        color: white !important;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        border: 1px solid #3d3d5c;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h3 {
        color: #a0a0cc;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #fafafa;
    }

    /* Result styling */
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

    /* Hide Streamlit's default menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
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
# SESSION STATE — persists data across Streamlit reruns
# ──────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "predictions" not in st.session_state:
    st.session_state.predictions = []


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
                            st.session_state.access_token = response.session.access_token
                            st.success("✅ Login successful!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Login failed: {str(e)}")

            else:  # Sign Up
                if st.button("📝 Sign Up", use_container_width=True):
                    if not email or not password:
                        st.error("Please enter email and password.")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        try:
                            response = supabase_client.auth.sign_up({
                                "email": email,
                                "password": password
                            })
                            st.success(
                                "✅ Sign up successful!\n\n"
                                "Check your email to confirm your account, then log in."
                            )
                        except Exception as e:
                            st.error(f"❌ Sign up failed: {str(e)}")

    # Sidebar info
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
# API STATUS CHECK
# ──────────────────────────────────────────────────────────
api_status = call_api("/health")
if api_status["success"]:
    health = api_status["data"]
    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <h3>API Status</h3>
            <div class="value">✅ Online</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        model_status = "✅ Loaded" if health.get("model_loaded") else "❌ Not Loaded"
        st.markdown(f"""
        <div class="metric-card">
            <h3>ML Model</h3>
            <div class="value">{model_status}</div>
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
else:
    st.error(api_status.get("error", "Cannot connect to backend API."))
    st.info(
        "💡 **To start the backend:**\n\n"
        "```bash\n"
        "cd backend\n"
        "pip install -r requirements.txt\n"
        "uvicorn app.main:app --reload\n"
        "```"
    )

st.markdown("---")


# ──────────────────────────────────────────────────────────
# TABS — Different ways to use the app
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

    # Create a form with two columns
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("👴 Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        tenure = st.slider("📅 Tenure (months)", min_value=0, max_value=72, value=12, help="How many months has this customer been with our company?")
        monthly_charges = st.number_input("💰 Monthly Charges ($)", min_value=0.0, max_value=200.0, value=29.85, step=0.01)

    with col2:
        total_charges = st.number_input("💵 Total Charges ($)", min_value=0.0, max_value=10000.0, value=358.20, step=0.01)
        contract_type = st.selectbox("📋 Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox("💳 Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

    st.markdown("")
    if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):
        # Build the customer data dictionary
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

        # Call the backend API
        with st.spinner("🔄 Making prediction..."):
            result = call_api(
                "/predict",
                method="POST",
                data=customer_data,
                token=st.session_state.access_token
            )

        if result["success"]:
            prediction = result["data"]
            st.session_state.predictions.append(prediction)

            # Display result with styling
            if prediction["prediction"] == 1:
                st.markdown(f"""
                <div class="prediction-churn">
                    <h2>⚠️ CHURN RISK DETECTED</h2>
                    <p style="font-size: 1.2rem;">{prediction['message']}</p>
                    <p style="font-size: 0.9rem; opacity: 0.7;">Confidence: {prediction['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-no-churn">
                    <h2>✅ CUSTOMER WILL STAY</h2>
                    <p style="font-size: 1.2rem;">{prediction['message']}</p>
                    <p style="font-size: 0.9rem; opacity: 0.7;">Confidence: {prediction['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

            # Show raw data in an expander
            with st.expander("🔍 View raw API response"):
                st.json(prediction)
        else:
            st.error(result["error"])


# ──────────────────────────────────────────────────────────
# TAB 2: CSV Upload (Batch Prediction)
# ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📁 Upload CSV for Batch Predictions")
    st.markdown(
        "Upload a CSV file with customer data. The file should have these columns:\n"
        "`gender`, `senior_citizen`, `tenure`, `monthly_charges`, `total_charges`, "
        "`contract_type`, `internet_service`, `payment_method`"
    )

    # Download sample CSV
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

    # File uploader
    uploaded_file = st.file_uploader("📂 Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded {len(df)} rows**")
            st.dataframe(df.head(10), use_container_width=True)

            # Required columns
            required_cols = [
                "gender", "senior_citizen", "tenure", "monthly_charges",
                "total_charges", "contract_type", "internet_service", "payment_method"
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
            else:
                if st.button("🔮 Predict All Customers", use_container_width=True, type="primary"):
                    # Prepare batch data
                    customers = df[required_cols].to_dict(orient="records")
                    batch_data = {"customers": customers}

                    with st.spinner(f"🔄 Processing {len(customers)} customers..."):
                        result = call_api(
                            "/batch-predict",
                            method="POST",
                            data=batch_data,
                            token=st.session_state.access_token
                        )

                    if result["success"]:
                        batch_result = result["data"]

                        # Summary metrics
                        st.markdown("### 📊 Results Summary")
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Total Customers", batch_result["total_customers"])
                        with m2:
                            st.metric("Predicted to Churn", batch_result["churn_count"])
                        with m3:
                            st.metric("Churn Rate", f"{batch_result['churn_rate']:.1f}%")

                        # Add predictions to the original dataframe
                        preds = batch_result["predictions"]
                        df["Prediction"] = [p["prediction_label"] for p in preds]
                        df["Confidence"] = [f"{p['confidence']:.1%}" for p in preds]

                        st.markdown("### 📋 Detailed Results")
                        st.dataframe(df, use_container_width=True)

                        # Download results as CSV
                        csv_output = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results CSV",
                            csv_output,
                            "churn_predictions.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    else:
                        st.error(result["error"])

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

        # ---- Chart 1: Prediction Distribution (Pie Chart) ----
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
                startangle=90,
                textprops={"color": "white", "fontsize": 10}
            )

        ax1.set_title("Churn Prediction Results", color="white", fontsize=14, fontweight="bold")
        st.pyplot(fig1)
        plt.close(fig1)

        # ---- Chart 2: Confidence Distribution (Histogram) ----
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

        # ---- Chart 3: Predictions Over Time (Bar Chart) ----
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
            "Without login, predictions from this session are shown below."
        )

        # Show session predictions
        if st.session_state.predictions:
            st.markdown("#### 📋 This Session's Predictions")
            session_df = pd.DataFrame(st.session_state.predictions)
            st.dataframe(session_df, use_container_width=True)
        else:
            st.info("No predictions made yet in this session.")
    else:
        # Fetch from Supabase via the backend
        result = call_api(
            "/predictions?limit=50",
            method="GET",
            token=st.session_state.access_token
        )

        if result["success"]:
            data = result["data"]
            st.markdown(f"**Showing {data['total']} saved predictions for {data['user_email']}**")

            if data["predictions"]:
                history_df = pd.DataFrame(data["predictions"])
                st.dataframe(history_df, use_container_width=True)
            else:
                st.info("No saved predictions yet. Make some predictions to see them here!")
        else:
            st.warning(result["error"])

        # Also show current session predictions
        if st.session_state.predictions:
            st.markdown("#### 📋 This Session's Predictions")
            session_df = pd.DataFrame(st.session_state.predictions)
            st.dataframe(session_df, use_container_width=True)


# ──────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "🔮 Customer Churn Prediction System | Built with FastAPI + Streamlit + Scikit-learn"
    "</p>",
    unsafe_allow_html=True
)
