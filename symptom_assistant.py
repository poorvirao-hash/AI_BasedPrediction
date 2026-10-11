from flask import Blueprint, render_template, request

assistant_bp = Blueprint("assistant", __name__)

@assistant_bp.route("/assistant")
def symptom_assistant():
    return "Smart Symptom Assistant is working!"