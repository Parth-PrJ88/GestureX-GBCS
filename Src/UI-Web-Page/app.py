from flask import Flask, render_template, jsonify
import subprocess
import sys
import os

app = Flask(__name__)

process = None  # stores the running python process

def run_python_mode(mode):
    global process

    if process and process.poll() is None:
        return "Already running. Close current window first."

    executable = sys.executable

    process = subprocess.Popen(
        ["cmd", "/c", executable, "4x4 Calibration.py", str(mode)]
    )

    return "Started! Check the popup window."

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
    msg = run_python_mode(2)
    return jsonify({"status": msg})


@app.route("/cursor")
def cursor():
    msg = run_python_mode(3)
    return jsonify({"status": msg})


@app.route("/stop")
def stop():
    global process
    if process and process.poll() is None:
        process.terminate()
        process = None
        return jsonify({"status": "Stopped running app."})
    return jsonify({"status": "No active process running."})


if __name__ == "__main__":
    app.run(debug=True)
