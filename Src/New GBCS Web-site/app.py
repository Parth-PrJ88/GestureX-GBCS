from flask import Flask, render_template, jsonify
import subprocess
import sys
import os
import json
import GBCS_Model

app = Flask(__name__)

process = None  # stores the running python process

def run_python_mode(mode):
    global process

    if process and process.poll() is None:
        return "Already running. Minimize current windows first."

    process = subprocess.Popen(
    [sys.executable, "GBCS_Model.py", str(mode)],
    creationflags=subprocess.CREATE_NO_WINDOW
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


# @app.route("/profiles")
# def profiles():
#     run_python_mode(4)
#     return jsonify({"status": "Opening Profile Manager..."})


@app.route("/cursor")
def cursor():
    run_python_mode(3)
    msg = "Providing Cursor Control..."
    return jsonify({"status": msg})


from pathlib import Path
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


# @app.route("/stop")
# def stop():
#     global process
#     if process and process.poll() is None:
#         process.terminate()
#         process = None
#         return jsonify({"status": "Stopped Process."})
#     return jsonify({"status": "No Active Process Running."})
@app.route("/stop")
def stop():
    GBCS_Model.request_stop()
    return jsonify({
        "status": "Stopping current workflow..."
    })

if __name__ == "__main__":
    app.run(debug=True)

