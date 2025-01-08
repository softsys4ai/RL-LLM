from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import string


template = (
    "You are tasked with answering the following question."
    "question: {question}\n\n"
    "Please follow these instructions carefully: \n\n"
    "1. Only pay attention to the question in question and don't answer other questions."
    "2. Provide accurate and complete answers to the best of your knowledge."
    "3. Answer as short as possible."
)

# # Load Llama 2 model and tokenizer
# model_name = "meta-llama/Llama-2-7b-hf"  # Replace with the specific Llama 2 model variant you want to use
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

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

with open('ground_truth.json', 'r') as f:
    ground_truth = json.load(f)

evaluation_results = {}
i = 0
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
        
        if correct_words.issubset(generated_words) or generated_words.issubset(correct_words):
            right_answer += 1
            score = 1
        else:
            score = 0
        
        evaluation_results[question] = {
            "generated_answer": response,
            "correct_answer": correct_answer,
            "Match Score": score
        }
    except Exception as e:
        print(f"Error processing question: {question}. Error: {e}")
    i += 1

with open('Ollama_evaluation_results_20.json', 'w') as f:
    json.dump(evaluation_results, f, indent=2)

total_questions = min(len(ground_truth), n + 1)
print("Total Accuracy", right_answer * 100 / total_questions)
