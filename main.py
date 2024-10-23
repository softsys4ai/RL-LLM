import streamlit as st
from parse import (
    scrape,
    extract_body_content,
    clean_body_content,
    split_content,
    parse,
)


# Scrape the website
description = st.text_area("Describe what you want to know")
description = description.replace(" ", "+")
content = scrape(f"https://www.google.com/search?q={description}")
body_content = extract_body_content(content)
cleaned_content = clean_body_content(body_content)
# Store the content
st.session_state.dom_content = cleaned_content


# Ask Questions 
if "dom_content" in st.session_state:

    if st.button("Let's go"):
        if description:
            st.write("Analyzing the content...")

            # Provide the content with Ollama
            chunks = split_content(st.session_state.dom_content)
            parsed_result = parse(chunks, description)
            st.write(parsed_result)
