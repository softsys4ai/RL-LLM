import json
import streamlit as st
from langchain_ollama import OllamaLLM
import string
from parse import (
    scrape,
    extract_body_content,
    clean_body_content,
    split_content,
    parse
)
def normalize_text(text):
    # Remove punctuation and convert to lowercase
    return set(word.strip(string.punctuation).lower() for word in text.split())

# Initialize the model
model = OllamaLLM(model="llama3.1")

# Load the ground_truth file (questions and answers)
with open('ground_truth.json', 'r') as f:
    ground_truth = json.load(f)

# Prepare a dictionary to store model outputs and similarity scores
evaluation_results = {}
i = 0
right_answer = 0
n = 20
# Loop through the questions in the ground_truth file
# for entry in ground_truth['data']:  # Iterate through each data entry
#     if i > n:
#         break
#     for paragraph in entry['paragraphs']:  # Loop through paragraphs
#         if i > n:
#             break
for entry in ground_truth:
  # Loop through the list of questions
    if i > n:
        break
    try:
        question = entry["question"]
        correct_answer = entry["answer"]
        #correct_answer = entry["_id"]  # Assuming "_id" stores the correct answer
        print("This is the question:", question)
        question = question.replace(" ", "+")
        content = scrape(f"https://www.google.com/search?q={question}")
        body_content = extract_body_content(content)
        cleaned_content = clean_body_content(body_content)
        st.session_state.dom_content = cleaned_content
        chunks = split_content(st.session_state.dom_content)
        response = parse(chunks, question)
        # Compute the match
        generated_words = normalize_text(response)  # Normalize the generated answer
        correct_words = normalize_text(correct_answer)  # Normalize the correct answe


        # Check if all the words of the correct answer are in the generated answer
        if correct_words.issubset(generated_words) or generated_words.issubset(correct_words):
            right_answer = right_answer + 1
            score = 1
        else:
            score = 0
        # Save the question, generated response, correct answer, and similarity score
        evaluation_results[question] = {
            "generated_answer": response,
            "correct_answer": correct_answer,
            "Match Score": score
        }
        print(f"Evaluating Question: {question}")
        print(f"Generated Answer: {response}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Match Score: {score}")
        print("=" * 50)
    except Exception as e:
        print(f"Error processing question: {question}. Error: {e}")
    i = i + 1


# Save the evaluation results to a JSON file
with open('evaluation_results_20.json', 'w') as f:
    json.dump(evaluation_results, f, indent=2)

print("Evaluation complete! Results saved to `evaluation_results.json`.")
print("Total Accuracy", right_answer*100/(n+1))