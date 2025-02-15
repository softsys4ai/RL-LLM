# import streamlit as st
# import torch
# from langchain_ollama import OllamaLLM
# import requests
# from parse import (
#     search,
#     extract_body_content,
#     clean_body_content,
#     split_content,
#     parse,
# )

# model = OllamaLLM(model="llama3.1")
# # Scrape the website
# input_prompt = st.text_area("Describe what you want to know")
# question = input_prompt.replace(" ", "+")
# content = search(f"https://www.google.com/search?q={question}")
# body_content = extract_body_content(content)
# cleaned_content = clean_body_content(body_content)
# # Store the content
# st.session_state.content = cleaned_content


# # Ask Questions 
# if "content" in st.session_state:

#     if st.button("Let's go"):
#         if question:
#             st.write("Thinking...")

#             # Provide the content with Ollama
#             chunks = split_content(st.session_state.content)
#             parsed_result = parse(chunks, question)
#             st.write(parsed_result)

import streamlit as st
import torch
from langchain_ollama import OllamaLLM
import requests
from parse import (
    # search_api,
    extract_body_content,
    clean_body_content,
    split_content,
    parse,
)

# Google Custom Search API credentials
GOOGLE_API_KEY = "your_google_api_key"  # Replace with your actual API key
SEARCH_ENGINE_ID = "your_search_engine_id"  # Replace with your Google CSE ID

def search_api(query):
    """Fetch the first search result using Google Custom Search API."""
    search_url = "https://www.googleapis.com/customsearch/v1"
    
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query + " -filetype:pdf",  # Exclude PDFs
        "num": 1  # Get the top result only
    }
    
    response = requests.get(search_url, params=params)
    
    if response.status_code == 200:
        search_results = response.json()
        if "items" in search_results and search_results["items"]:
            return search_results["items"][0]["link"]  # Return the top search result link
    return None
# Initialize Ollama model
model = OllamaLLM(model="llama3.1")

# Streamlit UI
st.title("Search & Analyze")

# User input
input_prompt = st.text_area("Describe what you want to know")

if input_prompt:
    question = input_prompt.replace(" ", "+")
    search_url = search_api(question)
    
    if search_url:
        st.write(f"🔗 Found: [{search_url}]({search_url})")
        
        # Fetch website content
        response = requests.get(search_url)
        if response.status_code == 200:
            raw_content = response.text
            
            # Process content
            body_content = extract_body_content(raw_content)
            cleaned_content = clean_body_content(body_content)
            
            # Store in session
            st.session_state.content = cleaned_content
            st.success("Content loaded successfully!")
        else:
            st.error("Failed to fetch content from the website.")
    else:
        st.error("No relevant results found.")

# Question Processing
if "content" in st.session_state:
    if st.button("Let's go"):
        st.write("Thinking...")

        # Provide the content to Ollama
        chunks = split_content(st.session_state.content)
        parsed_result = parse(chunks, input_prompt)

        st.write(parsed_result)
