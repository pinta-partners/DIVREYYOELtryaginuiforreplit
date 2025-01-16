from flask import Flask, render_template, request, jsonify, Response, url_for
import logging
import json
from rich.console import Console
from rich.logging import RichHandler
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os
import glob

# Install rich traceback handler
from rich.traceback import install
install(show_locals=True)

# Initialize Rich console
console = Console()

# Configure logging with Rich handler
logging.basicConfig(level=logging.INFO,
                   format="%(message)s",
                   handlers=[RichHandler(rich_tracebacks=True, markup=True)])
logger = logging.getLogger("main_script")

# Flask app initialization
app = Flask(__name__, static_url_path='/static', static_folder='static')

# Import step functions
from step_1 import main as step_1_main
from step_2 import main as step_2_main
from step_3 import main as step_3_main
from step_4 import main as step_4_main

def execute_with_timeout(func, timeout, *args, **kwargs):
    """Run a function with a timeout to prevent hangs."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            logger.error("Timeout occurred during execution.")
            raise TimeoutError(f"Execution of {func.__name__} exceeded {timeout} seconds.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    question = request.form.get('question')

    if not question:
        return jsonify({"error": "No question provided."}), 400

    steps = [
        {
            "name": "Step 1: Understanding The Question",
            "function": lambda: execute_with_timeout(step_1_main, 300, question)
        },
        {
            "name": "Step 2: Reading",
            "function": step_2_main
        },
        {
            "name": "Step 3: Predicting The Future",
            "function": step_3_main
        },
        {
            "name": "Step 4: Making Magic",
            "function": step_4_main
        },
    ]

    results = []

    for step in steps:
        try:
            logger.info(f"Starting {step['name']}...")
            step['function']()
            results.append({"step": step['name'], "status": "success"})
        except TimeoutError as e:
            logger.error(f"Timeout in {step['name']}: {e}")
            return jsonify({"error": f"Timeout in {step['name']}: {str(e)}"}), 500
        except Exception as e:
            logger.error(f"An error occurred in {step['name']}: {e}")
            return jsonify({"error": f"Error in {step['name']}: {str(e)}"}), 500

    folder_path = "data/answers/*/step_4/passage_analysis.json"
    file_list = glob.glob(folder_path)
    latest_file = max(file_list, key=os.path.getctime) if file_list else None

    if latest_file and os.path.exists(latest_file):
        with open(latest_file, 'r', encoding='utf-8') as file:
            analysis_data = json.load(file)
        return jsonify({
            "status": "success",
            "results": results,
            "analysis": analysis_data
        })
    else:
        return jsonify({
            "status": "success",
            "results": results,
            "analysis": "No analysis file found."
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)