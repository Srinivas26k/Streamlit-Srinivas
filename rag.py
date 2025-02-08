import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests
import PyPDF2
import json
import base64
import time

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Advanced RAG App", layout="wide")

# Load environment variables
load_dotenv()

# --- Custom CSS for styling, dark mode adjustments, and chat icons ---
st.markdown(
    """
    <style>
    /* Chat message styling */
    .chat-message-user {
        display: flex;
        align-items: center;
        padding: 8px;
        margin: 5px 0;
        background-color: #dcf8c6;
        border-radius: 10px;
    }
    .chat-message-assistant {
        display: flex;
        align-items: center;
        padding: 8px;
        margin: 5px 0;
        background-color: #fff;
        border: 1px solid #ddd;
        border-radius: 10px;
    }
    .chat-icon {
        margin-right: 10px;
        font-size: 1.5rem;
    }
    /* Dark mode adjustments */
    body.st-dark .chat-message-user {
        background-color: #005c4b;
        color: #e0e0e0;
    }
    body.st-dark .chat-message-assistant {
        background-color: #333;
        border: 1px solid #555;
        color: #e0e0e0;
    }
    /* Sidebar container for contribution/social links */
    .sidebar-container {
        padding: 10px;
        border-top: 1px solid #ccc;
        margin-top: 20px;
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
# Primary model is DeepSeek R1 Distill Llama 70B
PRIMARY_MODEL = "deepseek/deepseek-r1-distill-llama-70b:free"
# Fallback models list (order matters, fallback silently)
FALLBACK_MODELS = [
    "deepseek/deepseek-chat:free",
    "deepseek/deepseek-r1:free",
    "nvidia/llama-3.1-nemotron-70b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

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

def query_openrouter_with_fallback(messages):
    """Try the primary model first; if error occurs, fall back to alternate models."""
    model_list = [PRIMARY_MODEL] + FALLBACK_MODELS
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "<YOUR_SITE_URL>",
        "X-Title": "<YOUR_SITE_NAME>",
    }
    for model_str in model_list:
        data = {
            "model": model_str,
            "messages": messages,
            "extra_body": {},
        }
        try:
            response = requests.post(OPENROUTER_API_URL + "/chat/completions", json=data, headers=headers)
            response.raise_for_status()
            response_json = response.json()
            # If valid answer is returned, use it.
            if "choices" in response_json:
                return response_json["choices"][0]["message"]["content"]
        except Exception:
            # Silently catch errors (such as rate limits) and try the next model.
            continue
    # If all models fail, return a generic error message.
    return "Sorry, I couldn't process your request at this time."

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
    # Format articles with clickable links for sources/citations.
    context_lines = []
    for article in articles:
        context_lines.append(f"[{article['title']}]({article['url']}) - {article['snippet']}")
    return "\n\n".join(context_lines)

# --- Sidebar: File Upload & Contribution Section ---
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    st.info("Model: DeepSeek R1 Distill Llama 70B")
    
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(file_details)
        text = process_pdf(uploaded_file)
        if text:
            doc_id = add_to_index(text, file_details)
            st.success(f"PDF processed and added to the index! (Document ID: {doc_id})")
            
    # Contribution and Social Links Section in Sidebar
    container = st.container()
    container.write("# --- Contribution and Social Links Section ---")
    container.markdown("---")
    container.markdown("Created by **Nampalli Srinivas**")
    container.markdown("---")
    container.markdown("### Support this project")
    container.markdown("[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/srinivaskiv)")
    container.markdown("---")
    container.markdown("### Connect with me")
    container.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/srinivas-nampalli/)")
    container.markdown("[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/Srinivas26k)")
    container.markdown("### Report Issues")
    container.markdown("[![GitHub Issues](https://img.shields.io/badge/GitHub-Issues-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Srinivas26k/Ropar_Network_Project/issues)")
    container.markdown("---")
    container.markdown("""
I extend my heartfelt gratitude to the esteemed [Sudarshan Iyengar Sir](https://www.linkedin.com/in/sudarshan-iyengar-3560b8145/) for teaching me and offering a unique perspective on AI.  
A special thanks to my friends [Prakhar Gupta](https://www.linkedin.com/in/prakhar-kselis/), [Jinal Gupta](https://www.linkedin.com/in/jinal-gupta-220a652b6/), and [Purba Bandyopadhyay](https://www.linkedin.com/in/purba-b-88pb/),(Navya Mehta)[https://www.linkedin.com/in/navya-mehta-70b092225/] for constantly motivating and encouraging me to take on such a wonderful project.  
Your guidance and support have been truly invaluable!
""")
    container.markdown("---")

# --- Chat Interface using st.chat_input and st.chat_message ---
st.title("Chat with Your Document Assistant")

if prompt := st.chat_input("Enter your question:"):
    # Display user message with a user icon
    st.chat_message("user").markdown("👤 " + prompt)
    with st.spinner("Thinking..."):
        # Retrieve document context from uploaded PDFs via FAISS
        results = search_index(prompt)
        doc_context = []
        for doc_id, similarity in results:
            snippet = document_store[doc_id]['text'][:200].replace("\n", " ")
            doc_context.append(f"Document {doc_id + 1}: {snippet}... (Similarity: {similarity:.4f})")
        document_context = "\n\n".join(doc_context) if doc_context else "No relevant document context found."
    
        # Retrieve related articles (simulated)
        articles = get_related_articles(prompt)
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
            {"role": "user", "content": f"{combined_context}\n\nQuestion: {prompt}"},
        ]
    
        answer = query_openrouter_with_fallback(messages)
    # Append the sources (citations) to the answer
    final_response = "🤖 " + answer + "\n\n**Sources:**\n" + web_context
    st.chat_message("assistant").markdown(final_response)

# --- Sentiment Rating ---
sentiment_mapping = ["one", "two", "three", "four", "five"]
selected = st.feedback("stars")
if selected is not None:
    st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")