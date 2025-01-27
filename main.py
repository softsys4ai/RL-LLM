import streamlit as st
import torch
from langchain_ollama import OllamaLLM

from parse import (
    search,
    extract_body_content,
    clean_body_content,
    split_content,
    parse,
)

model = OllamaLLM(model="llama3.1")
# Scrape the website
input_prompt = st.text_area("Describe what you want to know")
question = input_prompt.replace(" ", "+")
content = search(f"https://www.google.com/search?q={question}")
body_content = extract_body_content(content)
cleaned_content = clean_body_content(body_content)
# Store the content
st.session_state.content = cleaned_content


# Ask Questions 
if "content" in st.session_state:

    if st.button("Let's go"):
        if question:
            st.write("Thinking...")

            # Provide the content with Ollama
            chunks = split_content(st.session_state.content)
            parsed_result = parse(chunks, question)
            st.write(parsed_result)
