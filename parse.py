from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

import selenium.webdriver as webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util



template = (
    "You are tasked with answering the following prompt: {description}. "
    "Please follow these instructions carefully: \n\n"
    "1. Only pay attention to the question in {description} and don't answer to other questions."
    "2. Provide accurate and complete answers to the best of your knowledge."
    "3. If there is any exact information corresponding to the {description} in {dom_content}, check your answers with it, else answer based on your own knowledge."
    "4. Look {dom_content} for real-time access if needed."
    # "5. When you can't answer based on {dom_content}, answer based on your own knowledge."
    # "5. Do not reference external content or indicate where the information comes from. Just provide the a straightforward aswer."
)


# template = (
#     "You are tasked with answering the following prompt: {description}. "
#     "First, provide an initial answer based on your knowledge. "
#     "Then, reflect on your initial answer and determine if it was accurate or not by using {dom_content}. "
#     "If your initial answer was incorrect or incomplete, revise your answer accordingly. "
#     "Respond in the meantime that you are generating your answers."
#     "Please submit only the revised version of your answer."
#     "Use the {example} to understand how you should respond."
# )



model = OllamaLLM(model="llama3.1")


# def parse_with_ollama(dom_chunks, description):
#     prompt = PromptTemplate.from_template(template)
#     chain = prompt | model

#     results = []

#     for i, chunk in enumerate(dom_chunks, start=1):
#         response = chain.invoke(
#             {"dom_content": chunk, "description": description}
#         )
#         print(f"Result batch: {i} of {len(dom_chunks)}")
#         results.append(response)

#     return "\n".join(results)

# Load a pre-trained model for computing similarity

similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def compute_similarity(description, chunk):
    # Encode both description and chunk content into vector embeddings
    description_embedding = similarity_model.encode(description, convert_to_tensor=True)
    chunk_embedding = similarity_model.encode(chunk, convert_to_tensor=True)
    
    # Compute cosine similarity between the description and content chunk
    similarity_score = util.pytorch_cos_sim(description_embedding, chunk_embedding)
    
    return similarity_score.item()  # Convert to a scalar value

def normalize_scores(scores):
    """Normalize scores to a 0-1 range."""
    min_score = min(scores)
    max_score = max(scores)
    
    # Handle case where all scores are the same
    if max_score == min_score:
        return [1.0] * len(scores)
    
    return [(score - min_score) / (max_score - min_score) for score in scores]

def weighted_merge_responses(responses, weights):
    """Merge the responses, weighting by their normalized scores."""
    weighted_result = ""
    
    for response, weight in zip(responses, weights):
        print(response, weight)
        weighted_result += response * int(weight) #It's just considering 0 and 1.
    
    return weighted_result

def parse(dom_chunks, description):
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model

    responses = []
    scores = []

    for i, chunk in enumerate(dom_chunks, start=1):
        # Model generates the response
        response = chain.invoke(
            {"dom_content": chunk, "description": description}
        )
        print(f"Result batch: {i} of {len(dom_chunks)}")
        
        # Compute the relevance score for the current chunk
        score = compute_similarity(description, chunk)
        scores.append(score)  # Store the relevance score
        responses.append(response)  # Store the model's response

    # Normalize the relevance scores to a 0-1 range
    normalized_scores = normalize_scores(scores)

    # Merge the responses, weighted by their normalized scores
    weighted_result = weighted_merge_responses(responses, normalized_scores)
    
    return weighted_result


def scrape(website):

    chrome_driver_path = "./chromedriver"
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(chrome_driver_path), options=options)

    try:
        driver.get(website)
        print("Page loaded...")
        html = driver.page_source

        return html
    finally:
        driver.quit()


def extract_body_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    if body_content:
        return str(body_content)
    return ""


def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, "html.parser")

    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()

    # Get text or further process the content
    cleaned_content = soup.get_text(separator="\n")
    cleaned_content = "\n".join(
        line.strip() for line in cleaned_content.splitlines() if line.strip()
    )

    return cleaned_content


def split_content(dom_content, max_length=3000):
    return [
        dom_content[i : i + max_length] for i in range(0, len(dom_content), max_length)
    ]
