from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

import selenium.webdriver as webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import requests
import streamlit as st

template = (
    "You are tasked with answering the following question based on the context provided in the question itself and the content. (Don't mention the question in your answer)"
    "question: {question}\n\n"
    "content: {content}\n\n"
    "Please follow these instructions carefully: \n\n"
    "1. Only pay attention to the question in question and don't answer to other questions."
    "2. Provide accurate and complete answers to the best of your knowledge."
    "3. If there is any exact information corresponding to the question in content, check your answers with it, else answer based on your own knowledge."
    "4. Look content for real-time access if needed."
    "5. Answer as short as possible."
)


model = OllamaLLM(model="llama3.1")


def normalize_scores(scores):
    """Normalize scores to a 0-1 range."""
    min_score = min(scores)
    if not min_score:
        min_score = 0.0
    max_score = max(scores)
    
    # Handle case where all scores are the same
    if max_score == min_score:
        return [1.0] * len(scores)
    
    return [(score - min_score) / (max_score - min_score) for score in scores]


def evaluate_relevance(question, response):
    """
    Evaluate the relevance of the response to the question.
    A possible method could involve a language model's judgment of how well the response answers the question.
    """
    prompt_relevance = f"""
    Evaluate how relevant the following response is to the question:
    Question: {question}
    Response: {response}
    Rate the relevance on a scale from 0 (not relevant) to 1 (very relevant).
    """

    # Pass the prompt directly as a string to the model
    relevance_score = model.invoke(prompt_relevance)  # Passing the prompt as a string

    
    # Use regex to extract the numeric value from the response
    match = re.search(r"(\d+(\.\d+)?)", relevance_score)
    if match:
        return float(match.group(1))  # Extracted number as a float
    else:
        raise ValueError(f"Could not extract a relevance score from: {relevance_score}")



def parse(chunks, question):
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model

    highest_relevance = -1
    best_response = ""
    i = 0
    # Process all chunks and evaluate relevance
    for chunk in chunks:
        response = chain.invoke({"question": question, "content": chunk})

        # Evaluate the relevance of the response to the question
        relevance = evaluate_relevance(question, response)

        # # Print chunk and its score
        # print(f"Chunk {i+1}: {chunk}...")
        # print(f"Relevance Score: {relevance}")
        # print("=" * 50)
        
        if relevance > highest_relevance:
            highest_relevance = relevance
            best_response = response
        i = i + 1
    return best_response



# def search(website):

#     chrome_driver_path = "./chromedriver"
#     options = webdriver.ChromeOptions()
#     options.add_argument("--headless")
#     driver = webdriver.Chrome(service=Service(chrome_driver_path), options=options)
    
#     try:
#         driver.get(website)
#         print("Page loaded...")
#         html = driver.page_source

#         return html
#     finally:
#         driver.quit()

# Tor Configuration
TOR_PROXY = "socks5h://127.0.0.1:9050"
proxies = {
    "http": TOR_PROXY,
    "https": TOR_PROXY
}
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}
def search(url):
    """Make a request through Tor."""
    try:
        response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data: {e}")
        return None

def extract_body_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    if body_content:
        return str(body_content)
    return ""


def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, "html.parser")

    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()

    # Get text or further process the content
    cleaned_content = soup.get_text(separator="\n")
    cleaned_content = "\n".join(
        line.strip() for line in cleaned_content.splitlines() if line.strip()
    )

    return cleaned_content


def split_content(dom_content, max_length=3000):
    return [
        dom_content[i : i + max_length] for i in range(0, len(dom_content), max_length)
    ]