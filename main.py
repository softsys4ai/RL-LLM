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
    extract_body_content,
    clean_body_content,
    split_content,
    parse,
)

# Tor Configuration
TOR_PROXY = "socks5h://127.0.0.1:9050"
proxies = {
    "http": TOR_PROXY,
    "https": TOR_PROXY
}
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

def check_tor():
    """Check if Tor is working properly."""
    try:
        response = requests.get("https://check.torproject.org", proxies=proxies, headers=headers, timeout=10)
        if "Congratulations" in response.text:
            return True
        else:
            return False
    except requests.RequestException as e:
        st.error(f"Error checking Tor: {e}")
        return False

def search(url):
    """Make a request through Tor."""
    try:
        response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data: {e}")
        return None

# Initialize the LLM Model
model = OllamaLLM(model="llama3.1")

# Streamlit UI
st.title("Tor-Enabled Web Scraper")

# Check if Tor is running
if check_tor():
    st.success("Tor is enabled. Your connection is anonymous!")
else:
    st.warning("Tor is NOT working. Please ensure it is running.")

# User Input
input_prompt = st.text_area("Describe what you want to know")
question = input_prompt.replace(" ", "+")

if st.button("Let's go"):
    st.write("Searching through Tor...")

    # Avoid scraping Google directly (Use an alternative)
    url = f"https://lite.duckduckgo.com/lite?q={question}"
    content = search(url)

    if content:
        body_content = extract_body_content(content)
        cleaned_content = clean_body_content(body_content)

        # Store the content
        st.session_state.content = cleaned_content
        st.success("Content retrieved successfully!")
        chunks = split_content(st.session_state.content)
        parsed_result = parse(chunks, question)
        st.write(parsed_result)
