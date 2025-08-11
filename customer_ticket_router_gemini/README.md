# AI-Powered Customer Support Router

This project is a foundational AI agent designed to automate the initial stage of a customer support process. It uses a Large Language Model (LLM) to classify incoming customer queries and route them to a specific handler for a tailored response.

## Key Features
* **LLM-Powered Classification:** Intelligently categorizes customer queries into predefined categories like "Technical Support" or "Billing Inquiry."
* **Conditional Routing:** Directs the query to the correct processing node based on the classification.
* **Automated Responses:** Generates a category-specific response using the LLM.
* **Orchestration:** The entire workflow is managed by LangGraph.

## Prerequisites
* Python 3.10+
* A Google Gemini API key.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate     # On Windows
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Setup
1.  Obtain a **Google Gemini API key** from the [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Create a file named **`.env`** in the project's root directory.
3.  Add your API key to the `.env` file like this:
    ```
    GOOGLE_API_KEY="your_api_key_here"
    ```

## How to Run

After following the setup steps, run the agent from your terminal:

```bash
python agent.py