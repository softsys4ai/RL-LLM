import streamlit as st
import requests
from langchain_ollama import OllamaLLM
from parse import (
    search_api, 
    extract_body_content,
    clean_body_content,
    split_content,
    parse,
    scoring_system
)

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize Ollama model and sentence transformer for similarity scoring
model = OllamaLLM(model="llama3.1")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.title("Search & Analyze")

# User input
input_prompt = st.text_area("Describe what you want to know")

if input_prompt:
    question = input_prompt.replace(" ", "+")
    
    # Get initial search results
    search_results = search_api(question)
    st.write("🔎 **Search Results:**", search_results)
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
            st.write(f"✅ Found: [{url}]({url}) - Loaded Successfully!")
            st.session_state.contents.append(cleaned_content)
        else:
            st.write(f"❌ Failed to fetch: [{url}]({url}) - Trying another...")

        # Ensure we have 3 valid links
    if len(valid_links) < 3:
        st.error("Couldn't retrieve content from 3 websites. Consider trying a different query.")


# Question Processing
if "contents" in st.session_state and st.button("Let's go"):
        st.write("🔍 Analyzing retrieved content...")
        
        parsed_results = []
        
        # Generate answer from Llama 3.1 as reference
        st.write("🤖 Generating reference answer from Llama 3.1...")
        reference_answer = model.invoke(input_prompt)
        st.write(f"**Reference Answer:** {reference_answer}")

        # Iterate over all collected content
        parsed_results = []
        highest_relevances = []
        for url in valid_links:
            response = requests.get(url)
            raw_content = response.text
            body_content = extract_body_content(raw_content)
            cleaned_content = clean_body_content(body_content)
            chunks = split_content(cleaned_content)
            parsed_result, highest_relevance = parse(chunks, input_prompt)
            parsed_results.append(parsed_result)
            highest_relevances.append(highest_relevance)
        
        # Save parsed results
        st.session_state.parsed_results = parsed_results
        parsed_results, scores, sorted_indices, sorted_rep_indices, similarities, best_response = scoring_system(parsed_results, highest_relevances, input_prompt)
        # Display Final Scores
        st.write("🏆 **Final Scores:**")
        for i in range(0,3):
            st.write(f"🔹 **Website {i+1}: ", scores[i], ":", "Similarity Score: ", sorted_indices[i], similarities[i], "Repeatation Score:", sorted_rep_indices[i])
            st.write(parsed_results[i])

        st.write("🌟 **Best Answer Based on Scoring System:**")
        st.write(best_response)
