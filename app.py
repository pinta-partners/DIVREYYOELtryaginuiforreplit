from flask import Flask, render_template, request, jsonify, Response
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
app = Flask(__name__)

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
            raise TimeoutError(
                f"Execution of {func.__name__} exceeded {timeout} seconds.")


@app.route('/')
def index():
    return '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chasiddus AI</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f7fb; }
        h1 { color: #333; text-align: center; font-size: 36px; margin-bottom: 10px; }
        h2 { color: #666; text-align: center; font-size: 18px; margin-bottom: 30px; }
        form { 
            margin-bottom: 20px; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            gap: 15px;
            max-width: 800px;
            margin: 0 auto;
            background: #e6f3ff;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .chat-input-container {
            flex: 1;
            position: relative;
            order: 1;
        }
        input[type="text"] {
            width: 100%;
            padding: 15px;
            padding-right: 60px;
            border: 1px solid #cce4ff;
            border-radius: 25px;
            font-size: 16px;
            background: white;
            outline: none;
        }
        input[type="text"]:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
        }
        button {
            width: 45px;
            height: 45px;
            border: none;
            border-radius: 50%;
            background-color: #3b82f6;
            color: white;
            font-size: 20px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: #2563eb;
        }
        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: #cce4ff;
            display: flex;
            align-items: center;
            justify-content: center;
            order: 2;
        }
        .avatar svg {
            width: 24px;
            height: 24px;
            color: #3b82f6;
        }
        button:hover {
            background-color: #45a049;
        }
        .result-container { margin-top: 20px; }
        .card { background: #fff; border-radius: 10px; padding: 15px; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); }
        .card p { direction: rtl; text-align: right; }
        .card h3 { margin: 0; padding-bottom: 10px; border-bottom: 1px solid #ddd; direction: rtl; text-align: right; }
        .expand-btn { color: blue; cursor: pointer; }
        .hidden { display: none; }
        .spinner {
            border: 4px solid #f3f3f3; /* צבע רקע */
            border-top: 4px solid #4CAF50; /* צבע החלק המסתובב */
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .card .full-text {
            direction: rtl; /* כיוון הטקסט */
            text-align: justify; /* יישור הטקסט */
            line-height: 1.8; /* מרווח בין שורות */
            margin: 10px 0; /* מרווח בין פסקאות */
            padding: 10px; /* מרווח פנימי */
            background-color: #f9f9f9; /* רקע בהיר */
            border: 1px solid #ddd; /* מסגרת */
            border-radius: 5px; /* פינות מעוגלות */
            font-size: 16px; /* גודל הטקסט */
        }
    </style>
</head>
<body>
    <h1>Chasiddus AI</h1>
    <h2>Search for any Dvar Torah in Chasidishe Seforim using AI. This version has access to the entire Divrey Yoel.</h2>
    <form id="query-form">
        <div class="avatar">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
        </div>
        <div class="chat-input-container">
            <input type="text" id="question" name="question" placeholder="What would you like to know?" required>
        </div>
        <button type="submit" aria-label="Send message">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
            </svg>
        </button>
    </form>
    <div id="loading-spinner" class="hidden">
        <div class="spinner"></div>
    </div>
    <div id="result-container" class="result-container"></div>
    <script>
        const form = document.querySelector('#query-form');
        const spinner = document.getElementById('loading-spinner');
        const resultContainer = document.getElementById('result-container');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            resultContainer.innerHTML = '';
            spinner.classList.remove('hidden'); // הצגת הספינר

            const question = document.getElementById('question').value;
            const response = await fetch('/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: `question=${encodeURIComponent(question)}`
            });
            const data = await response.json();

            spinner.classList.add('hidden');

            if (data.analysis && data.analysis.analyzed_passages) {
                data.analysis.analyzed_passages.forEach((passage, index) => {
                    const card = `
                        <div class="card">
                            <h3>${passage.source}</h3>
                            <p><strong>סיכום:</strong> ${passage.explanation}</p>
                            <p><span class="expand-btn" onclick="toggleExpand('passage-${index}')">View Full Passage</span></p>
                            <div id="passage-${index}" class="hidden full-text">
                                ${passage.passage}
                            </div>
                        </div>
                    `;
                    resultContainer.innerHTML += card;
                });
            } else {
                resultContainer.innerHTML = '<p>No analysis available.</p>';
            }
        });

        function toggleExpand(id) {
            const element = document.getElementById(id);
            element.classList.toggle('hidden');
        }
    </script>
</body>
</html>
    '''


@app.route('/process', methods=['POST'])
def process():
    question = request.form.get('question')

    if not question:
        return jsonify({"error": "No question provided."}), 400

    steps = [
        {
            "name": "Step 1: Understanding The Question",
            "function":
            lambda: execute_with_timeout(step_1_main, 300, question)
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
            return jsonify({"error":
                            f"Timeout in {step['name']}: {str(e)}"}), 500
        except Exception as e:
            logger.error(f"An error occurred in {step['name']}: {e}")
            return jsonify({"error":
                            f"Error in {step['name']}: {str(e)}"}), 500

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
