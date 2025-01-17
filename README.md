# REval: A Framework for Enhancing Large Language Models with Real-Time Web Search

This project is an **AI Web Scraper** built with Streamlit that leverages the **LangChain OllamaLLM** to scrape, process, and analyze web content. Users can input their queries, and the system fetches relevant content, processes it, and provides concise, context-aware answers.

---

## Features

- **Streamlit Interface**: User-friendly interface for input and interaction.
- **Web Scraping**: Uses Selenium and BeautifulSoup to extract body content from web pages.
- **Content Processing**: Cleans, splits, and parses content for efficient analysis.
- **Reflective Question Answering**: Utilizes `OllamaLLM` for context-aware, relevant responses to user queries.
- **Relevance Scoring**: Evaluates responses based on relevance to user queries using normalized scores.

---

## Installation

### Prerequisites

- Python 3.8+
- ChromeDriver (Ensure compatibility with your browser version)
- Llama 3.1
- Additional dependencies in `requirements.txt`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/softsys4ai/RL-LLM.git
   cd RL-LLM

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Set up ChromeDriver:
   - Download the correct version of ChromeDriver for your browser. ([link:]( https://googlechromelabs.github.io/chrome-for-testing/))
   - Place it in the root directory or specify its path in `parse.py`.
     
4. Install Ollama: Download the correct version of Llama 3.1 for your device.
   
5. Run the Streamlit app:
   ```bash
   streamlit run main.py
