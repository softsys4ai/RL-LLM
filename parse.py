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
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

GOOGLE_API_KEY = "Your API Key"  
SEARCH_ENGINE_ID = "Your Search Engine ID"


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

llama_template = (
    "You are tasked with answering the following question."
    "question: {question}\n\n"
    "Please follow these instructions carefully: \n\n"
    "1. Only pay attention to the question in question and don't answer other questions."
    "2. Provide accurate and complete answers to the best of your knowledge."
    "3. Answer as short as possible."
)


model = OllamaLLM(model="llama3.1")
sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # For similarity checks
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def query_ollama(question):
    prompt = PromptTemplate.from_template(llama_template)
    chain = prompt | model

    response = chain.invoke({"question": question})
    if isinstance(response, dict):
        response = response.get("text", "")
    return response

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
    print(response)
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
        body_content = extract_body_content(chunk)
        cleaned_content = clean_body_content(body_content)
        # Print chunk and its score
        print(f"Chunk {i+1}: {cleaned_content}...")
        print(f"Relevance Score: {relevance}")
        print("=" * 50)
        
        if relevance > highest_relevance:
            highest_relevance = relevance
            best_response = response
        i = i + 1
    return best_response, highest_relevance


def get_similarity_score(answer1, answer2):
    """Compute similarity between two answers using SentenceTransformer."""
    embedding1 = sentence_model.encode(answer1, convert_to_tensor=True)
    embedding2 = sentence_model.encode(answer2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding1, embedding2).item()

def get_most_repeated_answer(answers):
    """Find the most repeated answer based on similarity."""
    similarity_threshold = 0.75  # Threshold to consider answers as similar
    grouped_answers = {}
    
    for i, ans1 in enumerate(answers):
        found_group = None
        for group in grouped_answers:
            if get_similarity_score(ans1, group) > similarity_threshold:
                found_group = group
                break
        
        if found_group:
            grouped_answers[found_group].append(i)
        else:
            grouped_answers[ans1] = [i]
    
    # Sort groups by size
    sorted_groups = sorted(grouped_answers.items(), key=lambda x: len(x[1]), reverse=True)
    ranked_answers = [group[0] for group in sorted_groups]
    
    return ranked_answers

def search_api(query):
    """Fetch the top 3 search results using Google Custom Search API."""
    search_url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query + " -filetype:pdf",  # Exclude PDFs
        "num": 10  # Fetch the top 10 results
    }

    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        search_results = response.json()
        if "items" in search_results:
            urls = [item.get("link", "") for item in search_results["items"]]
            return urls  # Ensure it's a list, NOT a single string
    return []

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


def scoring_system(parsed_results, highest_relevances, input_prompt):
        reference_answer = model.invoke(input_prompt)
        # Compute scores
        scores = [0, 0, 0]  # Initialize scores for each website
        repeatation_scores = [0, 0, 0]
        similarity_score = [0, 0, 0]
        # **Base Score for Each Website**
        base_scores = [6, 4, 2]  # First site gets 6, second 4, third 2
        for i in range(3):
            scores[i] += base_scores[i]

        # **Similarity to Llama 3.1 Answer**
        ref_embedding = embedder.encode([reference_answer])
        parsed_embeddings = embedder.encode(parsed_results)

        similarities = [cosine_similarity([parsed_embeddings[i]], ref_embedding)[0][0] for i in range(3)]
        sorted_indices = sorted(range(3), key=lambda i: similarities[i], reverse=True)

        # Assign similarity-based scores
        similarity_scores = [2, 1, 0]  # Most similar → 2, Second most → 1, Least → 0
        for rank, idx in enumerate(sorted_indices):
            scores[idx] += similarity_scores[rank]
            similarity_score[idx] = similarity_scores[rank]

        # **Answer Repetition Score**
        # Compute pairwise similarity among parsed answers
        similarity_matrix = cosine_similarity(parsed_embeddings)

        # Count similarity occurrences
        similarity_counts = [sum(similarity_matrix[i]) for i in range(3)]
        sorted_rep_indices = sorted(range(3), key=lambda i: similarity_counts[i], reverse=True)

        # Assign repetition-based scores
        rep_scores = [6, 4, 0]  # Most repeated → 6, Second most → 4, Least → 0
        for rank, idx in enumerate(sorted_rep_indices):
            scores[idx] += rep_scores[rank]
            repeatation_scores[idx] = rep_scores[rank]
            scores[idx] *= highest_relevances[idx]
    
        best_index = scores.index(max(scores))

        return parsed_results, scores, similarity_score, repeatation_scores, similarities, parsed_results[best_index]