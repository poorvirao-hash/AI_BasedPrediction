"""
app.py
------
Main Flask application for the AI-Based Disease Prediction Web App.

Features:
  - User registration, login, logout (session-based, hashed passwords)
  - Disease prediction using a pre-trained sklearn Pipeline
  - Prediction history dashboard per user
  - SQLite database via SQLAlchemy

Run:
    python app.py
    Open http://127.0.0.1:5000
"""

import os
import sys
import joblib
import pandas as pd
from functools import wraps
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from symptom_assistant import assistant_bp
# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from database import db, User, Prediction, init_db
from ml.preprocess import clean_symptoms, symptoms_list_to_string, build_feature_dict

# ---------------------------------------------------------------------------
# Load environment variables (optional .env file)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.register_blueprint(assistant_bp)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

# SQLite database stored in instance/ folder (Flask convention)

# SQLite database stored in instance/ folder
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")

# Create the folder if it does not exist
os.makedirs(INSTANCE_DIR, exist_ok=True)

DATABASE_PATH = os.path.join(INSTANCE_DIR, "app.db").replace("\\", "/")

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DATABASE_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialise DB
init_db(app)

# ---------------------------------------------------------------------------
# Load ML artifacts
# ---------------------------------------------------------------------------
ML_DIR = os.path.join(BASE_DIR, "ml")

try:
    model = joblib.load(os.path.join(ML_DIR, "model.pkl"))
    encoders = joblib.load(os.path.join(ML_DIR, "encoders.pkl"))
    DISEASE_CLASSES = encoders["classes"]          # list of disease names
    LABEL_ENCODER = encoders["label_encoder"]
    print(f"[INFO] Model loaded. Supports {len(DISEASE_CLASSES)} disease classes.")
except FileNotFoundError:
    model = None
    DISEASE_CLASSES = []
    LABEL_ENCODER = None
    print("[WARNING] ML model not found. Run 'python ml/train_model.py' first.")

# ---------------------------------------------------------------------------
# All available symptoms (for the prediction form checkboxes)
# ---------------------------------------------------------------------------
ALL_SYMPTOMS = sorted([
    "runny nose", "sneezing", "sore throat", "mild fever", "cough", "congestion",
    "fatigue", "high fever", "body aches", "chills", "headache", "chest pain",
    "shortness of breath", "sweating", "persistent cough", "weight loss",
    "night sweats", "blood in cough", "wheezing", "chest tightness",
    "breathlessness", "mucus production", "chest discomfort", "dry cough",
    "loss of taste", "loss of smell", "severe headache", "joint pain",
    "muscle pain", "rash", "nausea", "vomiting", "cyclical fever",
    "sustained fever", "abdominal pain", "weakness", "constipation",
    "loss of appetite", "frequent urination", "excessive thirst",
    "blurred vision", "slow healing", "dizziness", "nosebleed",
    "palpitations", "swollen legs", "sudden numbness", "confusion",
    "trouble speaking", "vision problems", "swelling", "decreased urine",
    "back pain", "jaundice", "dark urine", "pale skin", "cold hands",
    "brittle nails", "stiffness", "reduced range of motion", "redness",
    "light sensitivity", "sound sensitivity", "aura", "seizures",
    "staring spells", "uncontrollable movements", "loss of consciousness",
    "persistent sadness", "sleep disturbance", "appetite changes",
    "concentration problems", "excessive worry", "restlessness",
    "irritability", "muscle tension", "weight gain", "cold intolerance",
    "dry skin", "hair loss", "depression", "rapid heartbeat", "nervousness",
    "tremors", "heat intolerance", "frequent bowel movements", "diarrhea",
    "abdominal cramps", "muscle aches", "rebound tenderness",
    "burning urination", "cloudy urine", "pelvic pain", "itching",
    "hives", "blistering", "red eyes", "eye discharge", "tearing",
    "swollen eyelids", "itchy rash", "blisters",
])

# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    """Landing page."""
    return render_template("home.html")


# ── Registration ────────────────────────────────────────────────────────────

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("predict"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        # Basic validation
        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("register.html")

        # Check duplicate email
        existing = User.query.filter_by(email=email).first()
        if existing:
            flash("An account with this email already exists.", "danger")
            return render_template("register.html")

        # Create user
        hashed_pw = generate_password_hash(password)
        user = User(name=name, email=email, password=hashed_pw)
        db.session.add(user)
        db.session.commit()

        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


# ── Login ────────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("predict"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            flash("Invalid email or password.", "danger")
            return render_template("login.html")

        session["user_id"] = user.id
        session["user_name"] = user.name
        flash(f"Welcome back, {user.name}!", "success")
        return redirect(url_for("predict"))

    return render_template("login.html")


# ── Logout ───────────────────────────────────────────────────────────────────

@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# ── Prediction Form ──────────────────────────────────────────────────────────

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if model is None:
        flash("ML model is not loaded. Please run 'python ml/train_model.py' first.", "danger")
        return render_template("predict.html", symptoms=ALL_SYMPTOMS)

    if request.method == "POST":
        # Collect form data
        age = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()
        selected_symptoms = request.form.getlist("symptoms")  # list of strings

        # Validate
        if not age or not gender or not selected_symptoms:
            flash("Please fill in all fields and select at least one symptom.", "danger")
            return render_template("predict.html", symptoms=ALL_SYMPTOMS)

        try:
            age = int(age)
            if age < 1 or age > 120:
                raise ValueError
        except ValueError:
            flash("Please enter a valid age between 1 and 120.", "danger")
            return render_template("predict.html", symptoms=ALL_SYMPTOMS)

        if len(selected_symptoms) < 1:
            flash("Please select at least one symptom.", "danger")
            return render_template("predict.html", symptoms=ALL_SYMPTOMS)

        # Preprocess — identical to training
        symptoms_str = symptoms_list_to_string(selected_symptoms)
        feature_dict = build_feature_dict(age, gender, symptoms_str)
        X_input = pd.DataFrame([feature_dict])

        # Predict
        pred_encoded = model.predict(X_input)[0]
        pred_proba = model.predict_proba(X_input)[0]

        predicted_disease = LABEL_ENCODER.inverse_transform([pred_encoded])[0]
        confidence = float(max(pred_proba)) * 100  # percentage

        # Save to DB
        prediction_record = Prediction(
            user_id=session["user_id"],
            age=age,
            gender=gender,
            symptoms=symptoms_str,
            predicted_disease=predicted_disease,
            confidence=round(confidence, 1),
        )
        db.session.add(prediction_record)
        db.session.commit()

        # Store result in session for result page
        session["last_prediction"] = {
            "disease": predicted_disease,
            "confidence": round(confidence, 1),
            "symptoms": symptoms_str,
            "age": age,
            "gender": gender,
        }

        return redirect(url_for("result"))

    return render_template("predict.html", symptoms=ALL_SYMPTOMS)


# ── Result Page ──────────────────────────────────────────────────────────────

@app.route("/result")
@login_required
def result():
    prediction = session.get("last_prediction")
    if not prediction:
        flash("No prediction found. Please submit the form.", "warning")
        return redirect(url_for("predict"))
    return render_template("result.html", prediction=prediction)


# ── Dashboard ────────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    predictions = (
        Prediction.query
        .filter_by(user_id=user_id)
        .order_by(Prediction.created_at.desc())
        .all()
    )
    return render_template("dashboard.html", predictions=predictions)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
