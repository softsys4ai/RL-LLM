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
- Additional dependencies in `requirements.txt`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-web-scraper.git
   cd ai-web-scraper
