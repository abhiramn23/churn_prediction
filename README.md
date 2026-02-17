# 🔮 Customer Churn Prediction System

A complete end-to-end full-stack data science project built for beginners.

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Cleaning | Pandas | Load, clean, transform CSV data |
| Visualization | Matplotlib | Charts for feature importance, confusion matrix |
| ML Model | Scikit-learn | Train a Random Forest classifier |
| Backend API | FastAPI | REST API to serve predictions |
| Frontend UI | Streamlit | Interactive web dashboard |
| Auth & DB | Supabase | User login/signup + store predictions |
| Deployment | Render + Streamlit Cloud | Host backend + frontend |

## Quick Start

### 1. Clone and set up

```bash
git clone <your-repo-url>
cd datascience-project
```

### 2. Train the model

```bash
cd model_training
pip install -r requirements.txt
python train.py
```

### 3. Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 4. Run the frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## 📖 Full Guide

See **[PROJECT_GUIDE.md](./PROJECT_GUIDE.md)** for the complete 10-section walkthrough covering:
- Architecture explanation
- File-by-file breakdown
- Supabase setup
- Deployment steps
- Common mistakes
- How to extend the project

## License

MIT
