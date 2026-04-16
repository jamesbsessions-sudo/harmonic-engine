# ── Add these routes to app.py ────────────────────────────────────────────
# Make sure Flask is imported with:
#   from flask import Flask, send_from_directory
# and that you have:
#   app = Flask(__name__, static_folder='static')

import os
from flask import send_from_directory

# Serve the exercise engine page
@app.route('/exercise')
def exercise():
    return send_from_directory('.', 'exercise.html')

# Serve static files (EP_MK1.js WASM bundle, etc.)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)
