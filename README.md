# Advanced RAG App

**Advanced RAG App** is an interactive Retrieval-Augmented Generation (RAG) system designed to help students and users obtain detailed answers from their own PDF documents. The app allows users to upload PDFs, processes them to extract text and generate embeddings using Sentence Transformers, and indexes the content using FAISS. Users can then engage in a continuous chat with the system that retrieves relevant context from the documents and generates answers using the OpenRouter API. Optionally, users can include web search results to further enhance the answers.

## Architecture

Our system architecture is designed to efficiently process documents and handle user queries. Below is our architecture diagram:

![RAG System Architecture](assets/rag-system-flow.mermaid)
```mermaid
flowchart LR
    A[User] -->|Uploads PDF| B[Document Processing]
    B --> C[Text Extraction [PyPDF2]]
    C --> D[Embedding Generation [Sentence Transformers]]
    D --> E[Indexing [FAISS]]
    A -->|Enters Query| F[Chat Interface [Streamlit]]
    F --> G[Context Retrieval [FAISS Search]]
    G --> H[Query Assembly]
    H --> I[LLM Query [OpenRouter API]]
    I --> J[Answer Generation]
    J --> F

```


> **Note:** To view the diagram rendered on GitHub, ensure that Mermaid is enabled in your repository settings.

## Tech Stack

- **Frontend:**  
  - **Streamlit:**  
    Used to build the interactive chat interface, file upload UI, and overall user experience. Streamlit enables rapid prototyping and deployment with minimal code.

- **Document Processing:**  
  - **PyPDF2:**  
    Extracts text from uploaded PDF documents.  
  - **Sentence Transformers:**  
    Converts extracted text into semantic embeddings using the `all-MiniLM-L6-v2` model, which captures the meaning of the text effectively.  
  - **FAISS:**  
    An efficient similarity search library from Facebook AI Research used to index and retrieve document embeddings quickly.

- **LLM Backend:**  
  - **OpenRouter API:**  
    Provides access to advanced LLM models (primarily DeepSeek R1 Distill Llama 70B, with fallback options) to generate context-aware responses based on the document content.  
    - **Model Fallback:**  
      In the event of rate limits or errors, the system automatically falls back to alternate models, ensuring seamless user experience.

- **Optional Web Search Augmentation:**  
  - Simulated web search results can be enabled to combine external information with document context.  
  - Clickable source links and related images are displayed when web search is active.

- **User Feedback:**  
  - A sentiment rating slider is provided for users to rate the quality of the answers.

## How It Works

1. **Upload a Document:**  
   - Use the sidebar to upload a PDF file.
   - The text is extracted and converted into embeddings, which are indexed with FAISS.

2. **Ask a Question:**  
   - Enter your query in the chat interface.
   - Toggle the "Include web search" option near the input box if you wish to augment your answer with external sources.
   - The system retrieves relevant document context and (if enabled) web search results, and constructs a prompt for the LLM.

3. **Receive an Answer:**  
   - The assistant generates a detailed answer using the document context (and web results if selected).
   - The conversation is maintained (up to 10 messages) for continuous interaction.
   - If web search is active, related images and clickable sources are displayed to support the answer.

4. **Provide Feedback:**  
   - Use the sentiment slider to rate the answer quality.

## Getting Started

### Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/)
- Required packages (see `requirements.txt`)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Srinivas26k/Streamlit-Srinivas.git
   cd Streamlit-Srinivas
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment Variables:**  
   Create a `.env` file in the root directory and add your `OPENROUTER_API_KEY` (and any other required variables).

4. **Run the application:**
   ```bash
   streamlit run rag.py
   ```

## Contribution and Social Links

Created by **Nampalli Srinivas**

### Support this project
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/srinivaskiv)

### Connect with me
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/srinivas-nampalli/)  
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/Srinivas26k)

### Report Issues
[![GitHub Issues](https://img.shields.io/badge/GitHub-Issues-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Srinivas26k/Ropar_Network_Project/issues)

I extend my heartfelt gratitude to the esteemed [Sudarshan Iyengar Sir](https://www.linkedin.com/in/sudarshan-iyengar-3560b8145/) for teaching me and offering a unique perspective on AI.  
A special thanks to my friends [Prakhar Gupta](https://www.linkedin.com/in/prakhar-kselis/), [Jinal Gupta](https://www.linkedin.com/in/jinal-gupta-220a652b6/), and Purba Bandyopadhyay for constantly motivating and encouraging me to take on such a wonderful project.  
Your guidance and support have been truly invaluable!

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Thanks to the OpenRouter API team for providing access to advanced LLM models.
- Thanks to the Streamlit community for continuous support and inspiration.
```
