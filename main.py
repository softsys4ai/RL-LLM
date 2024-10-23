import streamlit as st
from parse import (
    scrape_website,
    extract_body_content,
    clean_body_content,
    split_dom_content,
    parse_with_ollama,
)


# Scrape the website
description = st.text_area("Describe what you want to know")
content = description.replace(" ", "+")
dom_content = scrape_website(f"https://www.google.com/search?q={content}")
# dom_content = scrape_website(f"https://www.wolframalpha.com/input?i={equation}")
# dom_content = scrape_website("https://stackoverflow.com/questions/40448278/mathematical-difficulties")
body_content = extract_body_content(dom_content)
cleaned_content = clean_body_content(body_content)
# Store the DOM content in Streamlit session state
st.session_state.dom_content = cleaned_content


# Ask Questions 
if "dom_content" in st.session_state:

    if st.button("Let's go"):
        if description:
            st.write("Analyzing the content...")

            # Provide the content with Ollama
            dom_chunks = split_dom_content(st.session_state.dom_content)
            parsed_result = parse_with_ollama(dom_chunks, description)
            st.write(parsed_result)