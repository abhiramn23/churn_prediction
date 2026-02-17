# 🔮 Customer Churn Prediction System — Complete Beginner Guide

> **Welcome!** This guide walks you through every part of this project. No prior knowledge assumed — we explain everything from scratch.

---

## TABLE OF CONTENTS

1. [High Level Architecture](#section-1--high-level-architecture)
2. [Folder Structure](#section-2--folder-structure)
3. [Backend Implementation](#section-3--backend-implementation-fastapi)
4. [Model Training Code](#section-4--model-training-code)
5. [Supabase Integration](#section-5--supabase-integration)
6. [Frontend Implementation](#section-6--frontend-implementation-streamlit)
7. [Connecting Frontend to Backend](#section-7--connecting-frontend-to-backend)
8. [Deployment Guide](#section-8--deployment-guide)
9. [Common Beginner Mistakes](#section-9--common-beginner-mistakes)
10. [How to Extend This Project](#section-10--how-to-extend-this-project)

---

## SECTION 1 — High Level Architecture

### What Are We Building?

A **Customer Churn Prediction System** — an app that predicts whether a customer will **leave (churn)** or **stay** with a business, based on their data (how long they've been a customer, how much they pay, etc.).

### Architecture Diagram (Text)

```
┌──────────────────────────────────────────────────────────────────────┐
│                        USER (You in a browser)                       │
└──────────┬───────────────────────────────────────────────┬───────────┘
           │                                               │
           │  Opens web app                                │ Signs up / Logs in
           ▼                                               ▼
┌─────────────────────┐                        ┌────────────────────┐
│   STREAMLIT FRONTEND│                        │      SUPABASE      │
│   (frontend/app.py) │◄──────────────────────►│   Authentication   │
│                     │    Auth tokens          │   + Database       │
│  • Login / Signup   │                        │                    │
│  • Enter data       │                        │  • User accounts   │
│  • Upload CSV       │                        │  • Predictions DB  │
│  • Show charts      │                        └────────────────────┘
│  • View history     │                                 ▲
└────────┬────────────┘                                 │
         │                                              │
         │ HTTP requests (JSON)                         │ Store predictions
         ▼                                              │
┌─────────────────────┐                                 │
│   FASTAPI BACKEND   │─────────────────────────────────┘
│  (backend/app/)     │
│                     │
│  • /predict         │◄──── Loads trained model
│  • /batch-predict   │      (models/model.pkl)
│  • /predictions     │
│  • /health          │
└─────────────────────┘
         ▲
         │ Created by
         │
┌─────────────────────┐
│   MODEL TRAINING    │
│  (model_training/)  │
│                     │
│  • Load CSV data    │
│  • Clean with Pandas│
│  • Train with sklearn│
│  • Save model.pkl   │
└─────────────────────┘
```

### What Each Technology Does

| Technology | Role | Analogy |
|:-----------:|:----:|:-------:|
| **Pandas** | Load and clean raw data | A spreadsheet editor that fixes messy data |
| **Scikit-learn** | Train the ML model | A teacher that learns patterns from examples |
| **Matplotlib** | Create charts and graphs | A chart-drawing tool |
| **FastAPI** | Backend REST API server | A waiter that takes orders and brings food |
| **Streamlit** | Frontend web interface | The restaurant menu customers interact with |
| **Supabase** | Authentication + database | The bouncer (auth) and filing cabinet (database) |
| **Render** | Host the backend online | A building where the kitchen operates 24/7 |
| **Streamlit Cloud** | Host the frontend online | The storefront open to the public |

### How Data Flows

1. **User** opens the Streamlit app in their browser
2. **User** logs in via Supabase (email + password)
3. **User** enters customer data (or uploads a CSV)
4. **Streamlit** sends the data to FastAPI via an HTTP POST request
5. **FastAPI** loads the trained model and makes a prediction
6. **FastAPI** stores the prediction in Supabase database
7. **FastAPI** sends the result back to Streamlit
8. **Streamlit** displays the result with charts

---

## SECTION 2 — Folder Structure

```
datascience project/
│
├── backend/                    ← The API server (FastAPI)
│   ├── app/
│   │   ├── __init__.py         ← Makes 'app' a Python package (can be empty)
│   │   ├── main.py             ← FastAPI entry point — defines all endpoints
│   │   ├── model.py            ← Loads the ML model and makes predictions
│   │   ├── schemas.py          ← Defines the shape of request/response data
│   │   ├── supabase_client.py  ← Connects to Supabase (database + auth)
│   │   └── auth.py             ← Authentication middleware (verifies tokens)
│   ├── requirements.txt        ← Backend Python dependencies
│   ├── Procfile                ← Tells Render how to start the server
│   ├── .env.example            ← Template for environment variables
│   └── (create .env yourself)  ← Your actual secret keys (NOT in git)
│
├── model_training/             ← ML model training
│   ├── train.py                ← The training script — run this first!
│   ├── sample_data.csv         ← 200-row sample customer dataset
│   ├── requirements.txt        ← Training dependencies
│   └── charts/                 ← Generated after running train.py
│       ├── feature_importance.png
│       └── confusion_matrix.png
│
├── frontend/                   ← The web interface (Streamlit)
│   ├── app.py                  ← Main Streamlit application
│   ├── requirements.txt        ← Frontend Python dependencies
│   ├── .env.example            ← Template for environment variables
│   └── .streamlit/
│       └── config.toml         ← Streamlit theme settings
│
├── models/                     ← Saved ML model files
│   ├── model.pkl               ← The trained model (generated by train.py)
│   ├── feature_names.pkl       ← Column names the model expects
│   ├── label_encoders.pkl      ← Text-to-number converters
│   └── .gitkeep                ← Keeps this folder in git
│
├── PROJECT_GUIDE.md            ← ⭐ THIS FILE — you're reading it!
├── README.md                   ← Quick start guide
└── .gitignore                  ← Files git should ignore
```

### File-by-File Explanation

| File | What It Does | When You Touch It |
|:-----|:-------------|:-----------------|
| `train.py` | Loads data, cleans it, trains model, saves model.pkl | Run once to create the model |
| `main.py` | Creates the API server with all endpoints | When adding new API endpoints |
| `model.py` | Loads model.pkl and provides predict() function | When changing prediction logic |
| `schemas.py` | Defines what data the API accepts/returns | When adding new fields |
| `supabase_client.py` | Handles database reads/writes | When changing Supabase setup |
| `auth.py` | Checks if a user is logged in | When changing auth rules |
| `app.py` (frontend) | The entire web interface | When changing the UI |

---

## SECTION 3 — Backend Implementation (FastAPI)

### What Is FastAPI?

FastAPI is a **Python web framework** for building APIs (Application Programming Interfaces). An API is like a waiter in a restaurant:

- You (the frontend) tell the waiter (API) what you want
- The waiter goes to the kitchen (ML model) to prepare it
- The waiter brings back the result

### How to Run the Backend

```bash
# Step 1: Navigate to the backend folder
cd backend

# Step 2: Create a virtual environment (keeps dependencies isolated)
python -m venv venv

# Step 3: Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

# Step 4: Install dependencies
pip install -r requirements.txt

# Step 5: Start the server
uvicorn app.main:app --reload
```

After running, open **http://localhost:8000/docs** in your browser. You'll see a beautiful interactive API documentation page automatically generated by FastAPI!

### API Endpoints Explained

| Endpoint | Method | Auth? | Description |
|:---------|:------:|:-----:|:------------|
| `GET /` | GET | No | Health check — is the server alive? |
| `GET /health` | GET | No | Detailed health info (is model loaded?) |
| `POST /predict` | POST | Optional | Predict churn for ONE customer |
| `POST /batch-predict` | POST | Optional | Predict churn for MANY customers |
| `GET /predictions` | GET | Required | Get your prediction history |

### Example: Testing with curl

After the backend is running, open a **new terminal** and run:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"gender\": \"Male\", \"senior_citizen\": 0, \"tenure\": 12, \"monthly_charges\": 29.85, \"total_charges\": 358.20, \"contract_type\": \"Month-to-month\", \"internet_service\": \"DSL\", \"payment_method\": \"Electronic check\"}"
```

**On Windows PowerShell**, use this instead:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body '{"gender": "Male", "senior_citizen": 0, "tenure": 12, "monthly_charges": 29.85, "total_charges": 358.20, "contract_type": "Month-to-month", "internet_service": "DSL", "payment_method": "Electronic check"}'
```

**Expected Response:**

```json
{
  "prediction": 0,
  "prediction_label": "Will Not Churn",
  "confidence": 0.87,
  "message": "✅ This customer is likely to stay (confidence: 87.0%). Keep up the good service!"
}
```

### Key Concepts in the Backend Code

**1. CORS Middleware** — Allows the frontend (different domain/port) to talk to the backend:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
)
```

**2. Startup Event** — Loads the ML model once when the server starts:

```python
@app.on_event("startup")
async def load_model():
    churn_model.load()
```

**3. Dependency Injection** — FastAPI automatically extracts the auth token:

```python
@app.post("/predict")
async def predict(customer: CustomerData, user = Depends(get_current_user)):
    # 'user' is automatically populated from the auth header
```

---

## SECTION 4 — Model Training Code

### What Is Machine Learning (ML)?

ML is teaching a computer to find patterns in data. In our case:

- **Input**: Customer data (how long they've been here, how much they pay, etc.)
- **Output**: Will they leave? (Yes = 1, No = 0)

The computer looks at 200 past examples where we already know the answer, learns the patterns, and then can predict for new customers.

### How to Train the Model

```bash
# Step 1: Navigate to the model training folder
cd model_training

# Step 2: Create & activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the training script
python train.py
```

### What Happens When You Run train.py?

1. **Load** — Reads `sample_data.csv` into a Pandas DataFrame
2. **Clean** — Drops the ID column, fills any missing values, converts text to numbers
3. **Split** — Divides data into 80% training and 20% testing
4. **Train** — Fits a Random Forest classifier (100 decision trees that vote together)
5. **Evaluate** — Tests accuracy on the 20% test data
6. **Visualize** — Creates feature importance and confusion matrix charts
7. **Save** — Exports the trained model as `models/model.pkl`

### Understanding the Algorithm: Random Forest

Imagine 100 people each look at a customer and make a guess about whether they'll leave:

- Each person sees a slightly different version of the data
- Each person uses simple yes/no rules (a "decision tree")
- We take a vote — the majority answer wins
- This "crowd wisdom" approach is very accurate!

### Understanding the Dataset

| Column | Type | Description |
|:-------|:-----|:------------|
| `customer_id` | ID | Unique identifier (dropped before training) |
| `gender` | Text | Male or Female |
| `senior_citizen` | 0/1 | Is the customer over 65? |
| `tenure` | Number | Months with the company |
| `monthly_charges` | Number | Monthly bill amount in dollars |
| `total_charges` | Number | Total amount paid so far |
| `contract_type` | Text | Month-to-month, One year, or Two year |
| `internet_service` | Text | DSL, Fiber optic, or No |
| `payment_method` | Text | How they pay the bill |
| `churn` | 0/1 | **TARGET** — did they leave? (what we predict) |

---

## SECTION 5 — Supabase Integration

### What Is Supabase?

Supabase is a **Backend-as-a-Service** (BaaS) that gives you:

- **Authentication** — user signup/login with email & password
- **PostgreSQL Database** — store data in tables (like Excel, but better)
- **REST API** — auto-generated API to read/write your data
- **Row Level Security (RLS)** — users can only see their own data

It's free for small projects!

### Step-by-Step Supabase Setup

#### Step 1: Create a Supabase Account

1. Go to **https://supabase.com**
2. Click **"Start your project"**
3. Sign in with GitHub (or email)

ℹ️ *GitHub login is recommended — it's faster.*

#### Step 2: Create a New Project

1. Click **"New Project"**
2. Enter a name: `churn-predictor`
3. Set a database password (save this!)
4. Choose a region close to you
5. Click **"Create new project"**
6. Wait ~2 minutes for setup to complete

#### Step 3: Get Your API Keys

1. In your project dashboard, go to **Settings** → **API** (in the sidebar)
2. Copy these two values:
   - **Project URL**: looks like `https://abcdefg.supabase.co`
   - **anon public key**: a long string starting with `eyJ...`

#### Step 4: Create the Predictions Table

1. Go to **SQL Editor** in the sidebar
2. Click **"New query"**
3. Paste this SQL and click **"Run"**:

```sql
-- Create the predictions table
CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    customer_data JSONB NOT NULL,
    prediction INTEGER NOT NULL,
    prediction_label TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Policy: users can only read their OWN predictions
CREATE POLICY "Users can view own predictions"
    ON predictions FOR SELECT
    USING (auth.uid() = user_id);

-- Policy: users can only insert their OWN predictions
CREATE POLICY "Users can insert own predictions"
    ON predictions FOR INSERT
    WITH CHECK (auth.uid() = user_id);
```

#### Step 5: Configure Your .env Files

**Backend** (`backend/.env`):

```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-public-key-here
```

**Frontend** (`frontend/.env`):

```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-public-key-here
API_URL=http://localhost:8000
```

> ⚠️ **NEVER** commit `.env` files to Git! They contain secrets. The `.gitignore` file already handles this.

#### Step 6: Test Authentication

1. Start the frontend: `cd frontend && streamlit run app.py`
2. In the sidebar, click **Sign Up**
3. Enter an email and password (min 6 characters)
4. Check your email for a confirmation link
5. After confirming, go back and **Login**

---

## SECTION 6 — Frontend Implementation (Streamlit)

### What Is Streamlit?

Streamlit is a Python library that turns scripts into web apps. Instead of writing HTML/CSS/JavaScript, you write Python:

```python
import streamlit as st
st.title("Hello World!")
st.text_input("Your name")
st.button("Click me!")
```

This creates a full web page with a title, text input, and button. Magic! ✨

### How to Run the Frontend

```bash
# Step 1: Navigate to the frontend folder
cd frontend

# Step 2: Create & activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

### Frontend Features

| Feature | Where | Description |
|:--------|:------|:------------|
| **Authentication** | Sidebar | Login / Sign Up with Supabase |
| **Single Prediction** | Tab 1 | Fill a form, get one prediction |
| **CSV Upload** | Tab 2 | Upload a CSV, get batch predictions + download results |
| **Visualizations** | Tab 3 | Pie chart, histogram, bar chart of predictions |
| **History** | Tab 4 | View saved predictions (requires login) |

### Key Streamlit Concepts

**1. Session State** — Data that persists across interactions:

```python
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
```

**2. Tabs** — Organize content into sections:

```python
tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
with tab1:
    st.write("Content for tab 1")
```

**3. File Uploader** — Let users upload files:

```python
file = st.file_uploader("Upload CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
```

**4. Columns** — Side-by-side layout:

```python
col1, col2 = st.columns(2)
with col1:
    st.text_input("Left")
with col2:
    st.text_input("Right")
```

---

## SECTION 7 — Connecting Frontend to Backend

### How the Connection Works

```
 STREAMLIT (Port 8501)           FASTAPI (Port 8000)
 ┌────────────────────┐          ┌────────────────────┐
 │  User clicks       │          │                    │
 │  "Predict" button  │          │                    │
 │         │          │  HTTP    │                    │
 │         ▼          │  POST    │                    │
 │  call_api("/predict") ──────► │  POST /predict     │
 │                    │          │    │               │
 │                    │          │    ▼               │
 │                    │   JSON   │  model.predict()   │
 │  Display result ◄────────────── Return result     │
 │                    │          │                    │
 └────────────────────┘          └────────────────────┘
```

### The call_api() Helper Function

This function lives in `frontend/app.py` and handles ALL communication with the backend:

```python
def call_api(endpoint, method="GET", data=None, token=None):
    url = f"{API_URL}{endpoint}"           # e.g., "http://localhost:8000/predict"
    headers = {"Content-Type": "application/json"}

    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.post(url, json=data, headers=headers)
    return response.json()
```

### Key Points

1. **The frontend and backend run on DIFFERENT ports** — Streamlit on 8501, FastAPI on 8000.
2. **CORS** is configured in the backend to allow cross-origin requests.
3. **Authentication tokens** are passed via the `Authorization: Bearer <token>` header.
4. **Error handling** catches connection errors and shows user-friendly messages.

### Running Both Together (Local Development)

You need **two terminal windows**:

**Terminal 1 — Backend:**

```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload
```

**Terminal 2 — Frontend:**

```bash
cd frontend
venv\Scripts\activate
streamlit run app.py
```

---

## SECTION 8 — Deployment Guide

### Overview

| Component | Platform | URL After Deploy |
|:----------|:---------|:-----------------|
| Backend (FastAPI) | Render | `https://your-app.onrender.com` |
| Frontend (Streamlit) | Streamlit Cloud | `https://your-app.streamlit.app` |

### Prerequisites

1. A **GitHub account** (free) — https://github.com
2. Your project pushed to a **GitHub repository**
3. A **Supabase project** (already set up in Section 5)

### Part A: Push Your Code to GitHub

```bash
# Step 1: Initialize git in your project folder
cd "datascience project"
git init

# Step 2: Add all files
git add .

# Step 3: Make your first commit
git commit -m "Initial commit: Customer Churn Prediction System"

# Step 4: Create a new repository on GitHub
# Go to https://github.com/new
# Name it: customer-churn-predictor
# Keep it public (required for free deployment)
# Do NOT initialize with README (we already have one)

# Step 5: Connect and push
git remote add origin https://github.com/YOUR_USERNAME/customer-churn-predictor.git
git branch -M main
git push -u origin main
```

> ⚠️ **IMPORTANT**: Make sure your `.env` files are NOT committed! Check with `git status` that they don't appear. The `.gitignore` should handle this.

### Part B: Deploy Backend on Render

#### Step 1: Create a Render Account

1. Go to **https://render.com**
2. Click **"Get Started for Free"**
3. Sign in with GitHub

#### Step 2: Create a New Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Select `customer-churn-predictor`

#### Step 3: Configure the Service

Fill in these settings:

| Setting | Value |
|:--------|:------|
| **Name** | `churn-predictor-api` |
| **Region** | Choose closest to you |
| **Branch** | `main` |
| **Root Directory** | `backend` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | Free |

#### Step 4: Add Environment Variables

Click **"Advanced"** → **"Add Environment Variable"**:

| Key | Value |
|:----|:------|
| `SUPABASE_URL` | Your Supabase URL |
| `SUPABASE_KEY` | Your Supabase anon key |

#### Step 5: Important — Include the Model File

Since `models/model.pkl` is in `.gitignore`, you have two options:

**Option A (Recommended): Remove model.pkl from .gitignore**

Edit `.gitignore` and remove the line `models/*.pkl`, then:

```bash
git add models/model.pkl models/feature_names.pkl models/label_encoders.pkl
git commit -m "Add trained model files for deployment"
git push
```

**Option B: Train on Render**

Add to the Build Command: `pip install -r requirements.txt && cd ../model_training && pip install -r requirements.txt && python train.py && cd ../backend`

#### Step 6: Deploy!

1. Click **"Create Web Service"**
2. Wait 3–5 minutes for deployment
3. Your API will be live at: `https://churn-predictor-api.onrender.com`
4. Test it: visit `https://churn-predictor-api.onrender.com/docs`

> ℹ️ **Note**: Free tier Render services spin down after 15 minutes of inactivity. The first request after inactivity takes ~30 seconds.

### Part C: Deploy Frontend on Streamlit Cloud

#### Step 1: Create a Streamlit Cloud Account

1. Go to **https://share.streamlit.io**
2. Sign in with GitHub

#### Step 2: Deploy the App

1. Click **"New app"**
2. Fill in:

| Setting | Value |
|:--------|:------|
| **Repository** | `YOUR_USERNAME/customer-churn-predictor` |
| **Branch** | `main` |
| **Main file path** | `frontend/app.py` |

#### Step 3: Add Secrets

1. Click **"Advanced settings"** before deploying
2. In the **Secrets** box, paste:

```toml
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-anon-public-key-here"
API_URL = "https://churn-predictor-api.onrender.com"
```

> ℹ️ The `API_URL` should be your Render backend URL (no trailing slash).

#### Step 4: Deploy!

1. Click **"Deploy!"**
2. Wait 2–3 minutes
3. Your app will be live at: `https://your-app-name.streamlit.app`

#### Step 5: Update Frontend to Read Secrets

Streamlit Cloud uses `st.secrets` instead of `.env` files. The `app.py` code already reads from environment variables using `os.getenv()`, which works with Streamlit secrets because Streamlit Cloud injects them as environment variables.

### After Both Are Deployed

Your production setup:

```
User → streamlit.app → calls → onrender.com → returns prediction
                ↕                      ↕
            Supabase (auth)     Supabase (database)
```

---

## SECTION 9 — Common Beginner Mistakes

### Mistake 1: Forgetting to Train the Model First

**Problem:** You start the backend and get `FileNotFoundError: model.pkl not found`

**Fix:** Run the training script first!

```bash
cd model_training
python train.py
```

### Mistake 2: Not Activating the Virtual Environment

**Problem:** `ModuleNotFoundError: No module named 'fastapi'`

**Fix:** Always activate your venv before running:

```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### Mistake 3: Backend Not Running When Testing Frontend

**Problem:** Frontend shows "Cannot connect to backend API"

**Fix:** You need TWO terminals — one for backend, one for frontend. Both must be running simultaneously.

### Mistake 4: CORS Errors in Browser Console

**Problem:** Browser console shows `Access-Control-Allow-Origin` error

**Fix:** Make sure the backend has CORS configured (it already is in our code):

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"])
```

### Mistake 5: Committing .env Files to GitHub

**Problem:** Your secret keys are exposed publicly!

**Fix:**

1. Make sure `.env` is in `.gitignore`
2. If already committed, remove it: `git rm --cached .env`
3. Change your Supabase keys immediately at supabase.com

### Mistake 6: Wrong Data Format Sent to API

**Problem:** API returns 422 Validation Error

**Fix:** Check that your JSON matches the schema exactly:

```json
{
    "gender": "Male",           ← Must be a string
    "senior_citizen": 0,        ← Must be 0 or 1
    "tenure": 12,               ← Must be a positive integer
    "monthly_charges": 29.85,   ← Must be a positive float
    "total_charges": 358.20,    ← Must be a positive float
    "contract_type": "Month-to-month",  ← Must match training data values
    "internet_service": "DSL",          ← Must match training data values
    "payment_method": "Electronic check" ← Must match training data values
}
```

### Mistake 7: Using Different Python Versions

**Problem:** Package versions are incompatible

**Fix:** Use Python 3.10 or 3.11 for best compatibility. Check with: `python --version`

### Mistake 8: Render Deploy Fails — can't find model.pkl

**Problem:** The model file was gitignored and not uploaded

**Fix:** See Section 8, Part B, Step 5 for the solution.

### Mistake 9: Streamlit Cloud Can't Find Packages

**Problem:** `ModuleNotFoundError` on Streamlit Cloud

**Fix:** Make sure `frontend/requirements.txt` includes ALL packages your `app.py` imports.

### Mistake 10: Supabase RLS Blocking All Requests

**Problem:** No data is returned from the database

**Fix:** Make sure you created the RLS policies (see Section 5, Step 4). Without policies, RLS blocks everything.

---

## SECTION 10 — How to Extend This Project

### Level 1: Easy Extensions

- **Add more features to the dataset** — add columns like `num_support_tickets`, `online_backup`, `device_protection`
- **Try different models** — replace `RandomForestClassifier` with `GradientBoostingClassifier` or `LogisticRegression`
- **Add more visualizations** — ROC curve, precision-recall curve, SHAP plots
- **Add input validation** — check that `contract_type` is one of the expected values

### Level 2: Medium Extensions

- **User dashboard** — show prediction statistics per user (total predictions, churn rate trend)
- **Email notifications** — send an email when a high-churn customer is detected
- **Multiple models** — let users choose between different ML algorithms
- **Data download** — let users download their prediction history as CSV
- **Password reset** — add Supabase password reset functionality

### Level 3: Advanced Extensions

- **Real-time predictions with WebSockets** — get predictions as you type
- **Model retraining** — let admin users retrain the model with new data via the API
- **A/B testing** — compare two models in production
- **CI/CD pipeline** — auto-deploy when you push to GitHub using GitHub Actions
- **Monitoring** — add logging, metrics, and alerts (e.g., with Sentry)
- **Docker** — containerize the backend for more reliable deployments
- **Use a real dataset** — Kaggle's Telco Customer Churn dataset has 7,043 rows

### Project Ideas Using This Same Architecture

You can reuse this exact architecture for other ML problems:

| Project | Dataset | Model Type |
|:--------|:--------|:-----------|
| Employee Attrition Predictor | HR analytics data | Classification |
| House Price Estimator | Real estate data | Regression |
| Spam Email Detector | Email text data | Classification (NLP) |
| Student Grade Predictor | Academic records | Regression |
| Disease Risk Assessment | Medical records | Classification |
| Credit Default Predictor | Financial data | Classification |

---

## 🎉 You Made It!

Congratulations on building a complete, production-style full-stack data science project! You now understand:

- ✅ How to clean data with **Pandas**
- ✅ How to train an ML model with **Scikit-learn**
- ✅ How to visualize data with **Matplotlib**
- ✅ How to build APIs with **FastAPI**
- ✅ How to create web UIs with **Streamlit**
- ✅ How to add authentication with **Supabase**
- ✅ How to deploy to the cloud with **Render** and **Streamlit Cloud**

**Keep building!** 🚀
