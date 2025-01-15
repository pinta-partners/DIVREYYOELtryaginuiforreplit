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
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Chasiddus AI</title>
    <style>
      body {
        font-family: Arial, sans-serif;
        margin: 20px;
      }
      h1 {
        color: #333;
        text-align: center;
        font-size: 36px;
        margin-bottom: 10px;
      }
      h2 {
        color: #666;
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
      }

      /* ========== Chat Form (User Input) ========== */
      .chat-form {
        margin-bottom: 20px;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        width: 90vw;
      }
      .chat-input-container {
        flex: 1;
        display: flex;
        background: #f0f4f9;
        border-radius: 25px;
        padding: 8px 15px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
      }
      textarea {
        flex: 1;
        padding: 12px;
        border: none;
        background: transparent;
        font-size: 16px;
        outline: none;
        resize: none;
        overflow: hidden;
      }
      .chat-form button {
        padding: 8px;
        border: none;
        background: transparent;
        cursor: pointer;
        color: #2d7ff9;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .chat-form button:hover {
        color: #1b6eef;
      }
      .chat-form button svg {
        width: 20px;
        height: 20px;
      }
      .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        overflow: hidden;
        background: #fff;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
      }
      .avatar svg {
        width: 100%;
        height: 100%;
        object-fit: cover;
      }

      /* ========== Chat Display ========== */
      .result-container {
        max-width: 800px;
        margin: 0 auto;
      }

      .ai-card {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 8px;
      }

      .ai-card:not(:first-child) {
        padding-right: 60px;
      }
   .ai-bubble {
  position: relative;
  background: #fff; /* AI bubble is white */
  border-radius: 10px;
  padding: 10px 15px;
  width: 70vw;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  text-align: right; /* Ensure content within aligns right */
}


      .ai-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        overflow: hidden;
        background: #fff;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        margin-left: 10px;
        padding: 5px;
      }
      .ai-avatar svg {
        width: 100%;
        height: 100%;
        object-fit: cover;
      }

      /* ========== Typing Indicator (3 dots) ========== */
     /* Adjust Typing Indicator to the right */
.typing-indicator {
  display: inline-flex;
  justify-content: flex-end; /* Align dots to the right */
  width: 24px;
  margin-left: auto; /* Push indicator to the right */
}
      .typing-indicator span {
        width: 6px;
        height: 6px;
        background-color: #666;
        border-radius: 50%;
        margin-right: 3px;
        animation: bounce 1s infinite;
      }
      .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
      }
      .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
      }
      @keyframes bounce {
        0%,
        80%,
        100% {
          transform: scale(0);
        }
        40% {
          transform: scale(1);
        }
      }

      .source {
        font-size: 14px;
        color: #666;
        margin-bottom: 5px;
      }

      .hebrew {
        direction: rtl;
        font-size: 16px;
      }

      /* ========== "Expand" styles ========== */
      .expand-btn {
        color: blue;
        cursor: pointer;
        font-size: 14px;
      }
      .hidden {
        display: none;
      }
      .full-text {
        direction: rtl;
        text-align: justify;
        line-height: 1.8;
        margin-top: 10px;
        padding: 10px;
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-size: 16px;
      }
    </style>
</head>
<body>
    <h1>Chasiddus AI</h1>
    <h2>Search for any Dvar Torah in Chasidishe Seforim using AI. This version has access to the entire Divrey Yoel.</h2>

    <form id="query-form" class="chat-form">
      <div class="avatar">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-circle-user">
          <circle cx="12" cy="12" r="10" />
          <circle cx="12" cy="10" r="3" />
          <path d="M7 20.662V19a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v1.662" />
        </svg>
      </div>
      <div class="chat-input-container">
        <textarea id="question" name="question" placeholder="Type your question..." required rows="1"></textarea>
        <button type="submit">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </div>
    </form>

    <div id="result-container" class="result-container"></div>

    <script>
      const form = document.querySelector("#query-form");
      const resultContainer = document.getElementById("result-container");

      form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const textarea = document.getElementById("question");
        const sendButton = form.querySelector("button");

        textarea.disabled = true;
        textarea.style.cursor = "not-allowed";
        sendButton.style.display = "none";

        const question = textarea.value;

        const aiCard = document.createElement("div");
        aiCard.className = "ai-card";
        aiCard.style.marginBottom = "8px";

        aiCard.innerHTML = `
          <div class="ai-bubble" id="aiBubble">
            <div class="typing-indicator" id="typingIndicator">
              <span></span><span></span><span></span>
            </div>
          </div>
          <div class="ai-avatar">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-bot"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>
          </div>
        `;
        resultContainer.appendChild(aiCard);

        const response = await fetch("/process", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: `question=${encodeURIComponent(question)}`,
        });
        const data = await response.json();

        const aiBubble = document.getElementById("aiBubble");
        const typingIndicator = document.getElementById("typingIndicator");

        if (typingIndicator) {
          typingIndicator.remove();
        }

        if (data.analysis && data.analysis.analyzed_passages) {
          data.analysis.analyzed_passages.forEach((passage, index) => {
            if (index === 0) {
              aiBubble.innerHTML = `
                <p class="source">${passage.source}</p>
                <p class="hebrew"><strong>סיכום:</strong> ${passage.explanation}</p>
                <span class="expand-btn" onclick="toggleExpand('passage-${index}')">View Full Passage</span>
                <div id="passage-${index}" class="hidden full-text">${passage.passage}</div>
              `;
            } else {
              const additionalCard = document.createElement("div");
              additionalCard.className = "ai-card";
              additionalCard.style.marginBottom = "8px";
              additionalCard.innerHTML = `
                <div class="ai-bubble">
                  <p class="source">${passage.source}</p>
                  <p class="hebrew"><strong>סיכום:</strong> ${passage.explanation}</p>
                  <span class="expand-btn" onclick="toggleExpand('passage-${index}')">View Full Passage</span>
                  <div id="passage-${index}" class="hidden full-text">${passage.passage}</div>
                </div>
              `;
              resultContainer.appendChild(additionalCard);
            }
          });
        } else {
          aiBubble.innerHTML = "<p>No analysis available.</p>";
        }
      });

      function toggleExpand(id) {
        const element = document.getElementById(id);
        element.classList.toggle("hidden");
        const expandBtn = element.previousElementSibling;
        expandBtn.style.display = "none";
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
