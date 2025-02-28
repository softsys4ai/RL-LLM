from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import string
import numpy as np

def calculate_perplexity(text):
    words = text.split()
    probabilities = np.full(len(words), 1 / len(words))  # Assume uniform distribution
    log_prob = np.log(probabilities)
    perplexity = np.exp(-np.mean(log_prob))
    return perplexity
template = (
    "You are tasked with answering the following question."
    "question: {question}\n\n"
    "Please follow these instructions carefully: \n\n"
    "1. Only pay attention to the question in question and don't answer other questions."
    "2. Provide accurate and complete answers to the best of your knowledge."
    "3. Answer as short as possible."
)

# For llama 3.1 evaluation
model = OllamaLLM(model="llama3.1")

def normalize_text(text):
    if isinstance(text, str):
        return set(word.strip(string.punctuation).lower() for word in text.split())
    return set()

def query_ollama(question):
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model

    response = chain.invoke({"question": question})
    if isinstance(response, dict):
        response = response.get("text", "")
    return response

def ask_binary_question(question, response):
    """
    Ask the model whether the response to a question is 'Yes' or 'No'.
    """
    evaluation_prompt = f"""
    Question: {question}
    Response: {response}
    Based on the response, is the answer to the question 'Yes' or 'No'? Answer with only 'Yes' or 'No'.
    """
    return model(evaluation_prompt).strip()


with open('ground_truth.json', 'r') as f:
    ground_truth = json.load(f)

evaluation_results = {}
total_perplexity = 0
i = 1
right_answer = 0
n = 20

for entry in ground_truth:
    if i > n:
        break
    try:
        question = entry["question"]
        correct_answer = entry["answer"]
        response = query_ollama(question)
        
        generated_words = normalize_text(response)
        correct_words = normalize_text(correct_answer)
        generated_words = normalize_text(response)  # Normalize the generated answer
        generated_binary = ask_binary_question(question, response)

       # Use the model to determine the binary answer
        generated_response = generated_binary + " " + response
        generated_sentence = normalize_text(generated_response)

        if correct_words.issubset(generated_sentence) or generated_sentence.issubset(correct_words): # Check if all the words of the correct answer are in the generated answer
            correct = 1
            right_answer += 1
            score = 1
        else:
            score = 0
        evaluation_results[question] = {}
        perplexity = calculate_perplexity(response)
        evaluation_results[question]["Perplexity"] = perplexity
        # Save the question, generated response, correct answer, and similarity score
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

total_accuracy = right_answer * 100 / (n + 1)
evaluation_results["Total Accuracy"] = total_accuracy
evaluation_results["Total Perplexity"] = total_perplexity
with open('Llama_evaluation_results_20.json', 'w') as f:
    json.dump(evaluation_results, f, indent=2)

print("Evaluation complete! Results saved to `Llama_evaluation_results_20.json`.")
print("Total Accuracy:", total_accuracy)
