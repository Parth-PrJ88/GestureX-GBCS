from flask import Flask, render_template, jsonify
import subprocess
import sys
import os
import json

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


@app.route("/stop")
def stop():
    global process
    if process and process.poll() is None:
        process.terminate()
        process = None
        return jsonify({"status": "Stopped Process."})
    return jsonify({"status": "No Active Process Running."})


if __name__ == "__main__":
    app.run(debug=True)

