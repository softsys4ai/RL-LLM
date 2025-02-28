import json
import streamlit as st
import requests
from langchain_ollama import OllamaLLM
import string
import numpy as np
from parse import (
    search_api,
    extract_body_content,
    clean_body_content,
    split_content,
    parse,
    scoring_system
)


# from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize Ollama model and sentence transformer for similarity scoring
model = OllamaLLM(model="llama3.1")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.title("Evaluating...")

# Load the ground_truth file (questions and answers)
with open('ground_truth.json', 'r') as f:
    ground_truth = json.load(f)

# Prepare a dictionary to store model outputs and similarity scores
evaluation_results = {}
total_perplexity = 0
i = 1
right_answer = 0
n = 5

def calculate_perplexity(text):
    words = text.split()
    probabilities = np.full(len(words), 1 / len(words))  # Assume uniform distribution
    log_prob = np.log(probabilities)
    perplexity = np.exp(-np.mean(log_prob))
    return perplexity

def normalize_text(text):
    # Remove punctuation and convert to lowercase
    return set(word.strip(string.punctuation).lower() for word in text.split())


def jaccard_similarity(set1, set2):
    """
    Calculate Jaccard similarity between two sets.
    """
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def ask_binary_question(question, response):
    """
    Ask the model whether the response to a question is 'Yes' or 'No'.
    """
    evaluation_prompt = f"""
    Question: {question}
    Response: {response}
    Based on the response, is the answer to the question 'Yes' or 'No'? Answer with only 'Yes' or 'No'.
    """
    return model.invoke(evaluation_prompt).strip()

# Loop through the questions in the ground_truth file
for entry in ground_truth:
    if i > n:
        break
    try:
        input_question = entry["question"]
        correct_answer = entry["answer"]
    
        print("This is the question:", input_question)
        question = input_question.replace(" ", "+")
        search_results = search_api(question)

        valid_links = []
        st.session_state.contents = []

        for url in search_results:
            if len(valid_links) >= 3:
                break  # Stop once we have 3 valid links
            
            response = requests.get(url)
            raw_content = response.text
            body_content = extract_body_content(raw_content)
            cleaned_content = clean_body_content(body_content)

            if response.status_code == 200 and cleaned_content:
                valid_links.append(url)
                st.session_state.contents.append(cleaned_content)


        reference_answer = model.invoke(input_question)
        parsed_results = []
        highest_relevances = []
        for url in valid_links:
            response = requests.get(url)
            raw_content = response.text
            body_content = extract_body_content(raw_content)
            cleaned_content = clean_body_content(body_content)
            chunks = split_content(cleaned_content)
            print(chunks[2])
            parsed_result, highest_relevance = parse(chunks, input_question)
            parsed_results.append(parsed_result)
            highest_relevances.append(highest_relevance)
        
        # Save parsed results
        st.session_state.parsed_results = parsed_results
        parsed_results, scores, sorted_indices, sorted_rep_indices, similarities, best_response = scoring_system(parsed_results, highest_relevances, input_question)


        correct_words = normalize_text(correct_answer)  # Normalize the correct answer
        generated_words = normalize_text(best_response)  # Normalize the generated answer
        generated_binary = ask_binary_question(question, best_response)

        # Use the model to determine the binary answer
        generated_response = generated_binary + " " + best_response
        generated_sentence = normalize_text(generated_response)

        if correct_words.issubset(generated_sentence) or generated_sentence.issubset(correct_words): # Check if all the words of the correct answer are in the generated answer
            correct = 1
            right_answer += 1
            score = 1
        else:
            score = 0
        # # Calculate similarity score
        # similarity_score = jaccard_similarity(correct_words, generated_sentence)
        # if similarity_score > 0.9:  # Check if similarity score is above 0.9
        #     correct = 1
        #     right_answer += 1
        #     score = 1
        # else:
        #     score = 0
        # Save the question, generated response, correct answer, and similarity score
        evaluation_results[question] = {}
        perplexity = calculate_perplexity(best_response)
        evaluation_results[question]["Perplexity"] = perplexity

        evaluation_results[question] = {
            "generated_answer": generated_response,
            "correct_answer": correct_answer,
            "Match Score": score,
            "Perplexity": perplexity
        }
        print(f"Generated Answer: {generated_response}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Match Score: {score}")
        print(f"Perplexity: {perplexity}")
        print("=" * 50)
        total_perplexity = total_perplexity + perplexity
    except Exception as e:
        print(f"Error processing question: {question}. Error: {e}")
    i += 1
    print(i)

total_accuracy = right_answer * 100 / (n + 1)
evaluation_results["Total Accuracy"] = total_accuracy
evaluation_results["Total Perplexity"] = total_perplexity
with open('evaluation_results_20.json', 'w') as f:
    json.dump(evaluation_results, f, indent=2)

print("Evaluation complete! Results saved to `evaluation_results_20.json`.")
print("Total Accuracy:", total_accuracy)