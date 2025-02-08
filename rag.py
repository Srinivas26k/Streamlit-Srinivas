import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests
from PIL import Image
import PyPDF2
import pytesseract
import json
import base64
import time

# Set page configuration (must be first)
st.set_page_config(page_title="Advanced RAG App", layout="wide")

# Load environment variables
load_dotenv()

# --- Custom CSS for styling, including dark mode adjustments ---
st.markdown(
    """
    <style>
    /* General styling for the main container */
    .main-container {
        padding: 20px;
    }
    /* Input container fixed at the bottom is removed in this version */
    /* Answer display styling */
    .qa-box {
        padding: 15px;
        border: 1px solid #ccc;
        border-radius: 10px;
        background-color: #f9f9f9;
        margin-top: 20px;
    }
    /* Dark mode adjustments */
    body.st-dark .qa-box {
        background-color: #2c2c2c;
        border: 1px solid #444;
        color: #e0e0e0;
    }
    body.st-dark {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Cache model loading for performance ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()

# --- Initialize or retrieve FAISS index and document store in session_state ---
if "faiss_index" not in st.session_state:
    dimension = 384  # Embedding dimension
    st.session_state.faiss_index = faiss.IndexFlatL2(dimension)
if "document_store" not in st.session_state:
    st.session_state.document_store = []
faiss_index = st.session_state.faiss_index
document_store = st.session_state.document_store

# --- OpenRouter API Setup ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
MODELS = {
    "Google Gemini Flash Lite 2.0": "google/gemini-flash-lite-2.0-preview",
    "Qwen VL Plus": "qwen/qwen-vl-plus",
    "DeepSeek R1 Distill Llama 70B": "deepseek/deepseek-r1-distill-llama-70b:free",
    "DeepSeek: R1": "deepseek/deepseek-r1:free",
}

# --- Helper Functions ---
def get_embedding(text):
    return model.encode(text)

def add_to_index(text, metadata):
    embedding = get_embedding(text)
    faiss_index.add(np.array([embedding]))
    document_store.append({"text": text, "metadata": metadata})
    return len(document_store) - 1

def search_index(query, k=5):
    query_vector = get_embedding(query)
    D, I = faiss_index.search(np.array([query_vector]), k)
    results = []
    for idx, doc_id in enumerate(I[0]):
        if doc_id < len(document_store):
            results.append((doc_id, D[0][idx]))
    return results

def process_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return ""

def process_image(file):
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
        return text if text.strip() != "" else "No text found in image."
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return "Error processing image."

def query_openrouter(messages, model_key):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "<YOUR_SITE_URL>",
        "X-Title": "<YOUR_SITE_NAME>",
    }
    data = {
        "model": MODELS[model_key],
        "messages": messages,
        "extra_body": {},
    }
    try:
        response = requests.post(OPENROUTER_API_URL + "/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        if "choices" in response_json:
            return response_json["choices"][0]["message"]["content"]
        else:
            st.error(f"Unexpected response format: {response_json}")
            return "Sorry, I couldn't process your request."
    except Exception as e:
        st.error(f"API request failed: {e}")
        return "Sorry, there was an error contacting the language model API."

def get_related_articles(query):
    # Simulated web search results for related articles.
    articles = [
        {
            "title": f"Understanding {query}: An In-depth Overview",
            "snippet": f"This article explores the key aspects of {query} with detailed explanations and examples.",
            "url": "http://example.com/article1",
        },
        {
            "title": f"How {query} is Revolutionizing the Field",
            "snippet": f"An insightful look into how {query} is impacting various industries.",
            "url": "http://example.com/article2",
        },
        {
            "title": f"Top Resources to Learn About {query}",
            "snippet": f"A curated list of resources for comprehensive information on {query}.",
            "url": "http://example.com/article3",
        },
    ]
    return articles

def format_web_context(articles):
    context_lines = []
    for article in articles:
        context_lines.append(f"{article['title']}\n{article['snippet']}\n(Source: {article['url']})")
    return "\n\n".join(context_lines)

def save_state():
    try:
        index_bytes = faiss.serialize_index(faiss_index).tobytes()
        index_b64 = base64.b64encode(index_bytes).decode("utf-8")
        state = {"document_store": document_store, "faiss_index": index_b64}
        with open("rag_state.json", "w") as f:
            json.dump(state, f)
        st.success("State saved successfully!")
    except Exception as e:
        st.error(f"Error saving state: {e}")

def load_state():
    try:
        with open("rag_state.json", "r") as f:
            state = json.load(f)
        loaded_store = state.get("document_store", [])
        index_b64 = state.get("faiss_index", "")
        if index_b64:
            index_bytes = base64.b64decode(index_b64.encode("utf-8"))
            loaded_index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
            st.session_state.document_store = loaded_store
            st.session_state.faiss_index = loaded_index
            st.success("State loaded successfully!")
        else:
            st.error("No index data found in saved state.")
    except Exception as e:
        st.error(f"Error loading state: {e}")

# --- Sidebar: File Upload and Model Selection ---
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Choose a file (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])
    selected_model = st.selectbox("Select Model", list(MODELS.keys()))
    
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(file_details)
        if uploaded_file.type == "application/pdf":
            text = process_pdf(uploaded_file)
        else:
            text = process_image(uploaded_file)
        if text:
            doc_id = add_to_index(text, file_details)
            st.success(f"File processed and added to the index! (Document ID: {doc_id})")

# --- Main Container for Q&A ---
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# Form for user to input a question
with st.form(key="chat_form", clear_on_submit=True):
    user_query = st.text_input("Enter your question here:")
    send_button = st.form_submit_button("Send")

if send_button and user_query:
    with st.spinner("Thinking..."):
        # Retrieve context from uploaded documents via FAISS
        results = search_index(user_query)
        doc_context = []
        for doc_id, similarity in results:
            snippet = document_store[doc_id]['text'][:200].replace("\n", " ")
            doc_context.append(f"Document {doc_id + 1}: {snippet}... (Similarity: {similarity:.4f})")
        document_context = "\n\n".join(doc_context) if doc_context else "No relevant document context found."
    
        # Simulated related articles (web context)
        articles = get_related_articles(user_query)
        web_context = format_web_context(articles)
    
        combined_context = f"Document Context:\n{document_context}\n\nWeb Context:\n{web_context}"
    
        # Prepare messages for the API call
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Provide a detailed and descriptive answer with explanations, citations, "
                    "and references to help the student understand the topic."
                ),
            },
            {"role": "user", "content": f"{combined_context}\n\nQuestion: {user_query}"},
        ]
    
        # Get answer from OpenRouter API
        answer = query_openrouter(messages, selected_model)
    
    # Display the Q&A result in a simple box
    st.markdown("<div class='qa-box'>", unsafe_allow_html=True)
    st.markdown(f"**Q:** {user_query}")
    st.markdown(f"**A:** {answer}")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Save and Load State Options ---
col1, col2 = st.columns(2)
with col1:
    if st.button("Save current state"):
        save_state()
with col2:
    if st.button("Load saved state"):
        load_state()

st.markdown("</div>", unsafe_allow_html=True)
