from flask import Flask, render_template, jsonify, request
import subprocess
import sys
import os
import json
import GBCS_Model
from pathlib import Path

print("=== FLASK LOADED ===")
app = Flask(__name__)

process = None  # stores the running python process

def run_python_mode(mode):
    global process

    if process and process.poll() is None:
        return "Already running. Minimize current windows first."

    process = subprocess.Popen(
        [sys.executable, "GBCS_Model.py", str(mode)]
    )
    return "Started! Wait for few seconds. Popup window pops."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/try")
def try_now():
    return render_template("try.html")


@app.route("/alignment")
def alignment():
    msg = run_python_mode(1)
    return jsonify({"status": msg})


@app.route("/calibration")
def calibration():
    run_python_mode(2)
    msg = "Starting Calibration in Few Seconds..."
    return jsonify({"status": msg})


@app.route("/cursor")
def cursor():
    run_python_mode(3)
    msg = "Providing Cursor Control..."
    return jsonify({"status": msg})

# Path for progress 
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PROGRESS_FILE = DATA_DIR / "progress.json"

@app.route("/progress")
def progress():
    try:
        with open(PROGRESS_FILE, "r") as f:
            return jsonify(json.load(f))
        
    except Exception:
        return jsonify({
            "phase":"Idle",
            "progress":0,
            "message":"Waiting...",
            "running":False,
            "completed":False
        })


# Get Saved Profiles
@app.route("/profiles")
def profiles():
    return jsonify(
        GBCS_Model.get_profiles()
    )


# Load Profiles
@app.route("/profile/load", methods=["POST"])
def load_profile():
    data = request.get_json()
    filename = data.get("filename")
    success, message = GBCS_Model.load_profile(filename)
    return jsonify({
        "success": success,
        "message": message
    })


# Save Profile
@app.route("/profile/save", methods=["POST"])
def save_profile():
    data = request.get_json()
    profile_name = data.get("profile_name", "").strip()

    # Check for Duplicate profile name
    overwrite = data.get("overwrite", False)
    safe_filename = "".join(
        c for c in profile_name
        if c.isalnum() or c in (" ", "_", "-")
    ).rstrip()

    safe_filename = safe_filename.replace(" ", "_").lower()
    profile_path = GBCS_Model.PROFILES_DIR / f"{safe_filename}.json"
    if profile_path.exists() and not overwrite:
        existing_name = profile_name
        try:
            with open(profile_path, "r") as f:
                existing_profile = json.load(f)

            existing_name = existing_profile.get(
                "profile_name",
                profile_name
            )

        except Exception:
            pass
        
        return jsonify({
            "success": False,
            "duplicate": True,
            "existing_name": existing_name,
            "message": "A profile with this name already exists."
        })
    
    # Save the profile
    success, message = GBCS_Model.save_current_profile(profile_name)
    return jsonify({
        "success": success,
        "message": message
    })


# Delete Profile
@app.route("/profile/delete", methods=["POST"])
def delete_profile():
    data = request.get_json()
    filename = data.get("filename")
    success, message = GBCS_Model.delete_profile(filename)
    return jsonify({
        "success": success,
        "message": message
    })


# API for set Active Profile
@app.route("/active-profile")
def active_profile():
    if not GBCS_Model.OUT_MAP.exists():
        return jsonify({
            "profile": "No Calibration",
            "status": "none"
        })

    try:
        with open(GBCS_Model.OUT_MAP, "r") as f:
            data = json.load(f)

        # Checking for No valid calibration
        if "wx" not in data or "wy" not in data:
            return jsonify({
                "profile": "No Calibration",
                "status": "none"
            })

        profile = data.get("profile_name", "").strip()

        # Runtime calibration only
        if profile == "" or profile.lower() == "default":
            return jsonify({
                "profile": "Unsaved Calibration",
                "status": "unsaved"
            })

        # Saved profile
        return jsonify({
            "profile": profile,
            "status": "saved"
        })

    except Exception:
        return jsonify({
            "profile": "No Calibration",
            "status": "none"
        })


@app.route("/stop")
def stop():
    GBCS_Model.request_stop()
    return jsonify({
        "status": "Stopping current workflow..."
    })


@app.route("/reset")
def reset():
    GBCS_Model.request_stop()
    GBCS_Model.clear_stop()

    GBCS_Model.update_progress(
        "Idle",
        0,
        'System Ready. Click "Run" on Alignment to begin.',
        running=False,
        completed=False
    )
    return jsonify({
        "status": "Workflow Reset."
    })


if __name__ == "__main__":
    GBCS_Model.update_progress(
        phase="Idle",
        progress=0,
        message='System Ready. Click "Run" on Alignment to begin.',
        running=False,
        completed=False
    )
    GBCS_Model.clear_stop()
    app.run(debug=True)
    # app.run(debug=False)

