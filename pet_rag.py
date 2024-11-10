import os
import sys
import time
import json
import logging
import requests
import subprocess
from bs4 import BeautifulSoup
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_ollama import OllamaLLM
from prompt_templates import structured_response_template
from dotenv import load_dotenv
from langchain.callbacks.tracers import LangChainTracer
from langsmith.run_helpers import traceable
from langchain.smith import RunEvalConfig, run_on_dataset

# Load env variables
load_dotenv()

# Updated LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "petgpt"
os.environ["USER_AGENT"] = "PetGPT Bot/1.0"

# Initialize LangSmith tracer
tracer = LangChainTracer(
    project_name = "petgpt",
)

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and settings
USER_AGENT = os.getenv("USER_AGENT", "PetGPT Bot/1.0")
HEADERS = {"User-Agent": USER_AGENT}
PET_CARE_URLS = [
    "https://www.rocketdogrescue.org/",
    "https://www.petmd.com/",
    "https://www.akc.org/"
]

@traceable(project_name="petgpt")

def query_llama_claude(prompt):
    """Run the ollama CLI to get a response from the llama-claude model."""
    try:
        result = subprocess.run(
            ["ollama", "run", "llama-claude"],
            input=prompt,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        logger.error(f"Error querying LlamaClaude: {e}")
        return "Model query failed."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
@traceable(project_name="petgpt")
def ask_petgpt():
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
        
    # Create prompt template 
    prompt = structured_response_template(question)

    # pass the formatted prompt_template to querry_llama_claude
    answer = query_llama_claude(prompt)

    return jsonify({
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def clean_text(text):
    """Clean text by removing extra whitespace and unwanted characters."""
    return " ".join(text.split()) if text else ""

@traceable(project_name="petgpt")
def clean_text(text):
    """Clean text by removing extra whitespace and unwanted characters."""
    return " ".join(text.split()) if text else ""

@traceable(project_name="petgpt")
def scrape_website(url):
    """Scrape content from a single website, with error handling and respectful timing."""
    time.sleep(2)  # Avoid overwhelming the server
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
            
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
            
        main_content = soup.find(['main', 'article']) or soup.body
        content = clean_text(main_content.get_text()) if main_content else None
        return content
        
    except requests.RequestException as e:
        logger.error(f"Error scraping {url}: {e}")
        return None

@traceable(project_name="petgpt")
def load_pet_data():
    """Load and clean data from multiple pet care websites."""
    documents = []
    logger.info("Starting data load from websites...")
    
    for url in PET_CARE_URLS:
        content = scrape_website(url)
        if content:
            documents.append(Document(page_content=content, metadata={"source": url}))
            logger.info(f"Scraped content from {url}")
        else:
            logger.warning(f"No content found for {url}")
    
    if not documents:
        logger.error("No documents successfully loaded.")
        return None
    return documents

@traceable(project_name="petgpt")
def split_documents(documents):
    """Split documents into smaller chunks for processing."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

@traceable(project_name="petgpt")
def create_vector_store(splits):
    """Create and persist vector store from document chunks."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(splits, embedding=embeddings, persist_directory="pet_knowledge_base")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None

@traceable(project_name="petgpt")
def setup_rag():
    """Set up the RAG pipeline with document loading, splitting, and vector store creation."""
    try:
        llm = OllamaLLM(model="llama-claude")
        documents = load_pet_data()
        if not documents:
            raise ValueError("Failed to load documents.")
            
        splits = split_documents(documents)
        vectorstore = create_vector_store(splits)
        if not vectorstore:
            raise ValueError("Vector store creation failed.")
            
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=vectorstore.as_retriever(),
            callbacks=[tracer]
            )

        logger.info("RAG setup complete!")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Error setting up RAG: {e}")
        return None

@traceable(project_name="petgpt")
def answer_pet_question(qa_chain, question):
    """Generate answer using the RAG pipeline."""
    try:
        logger.info(f"Processing question: {question}")
        answer = qa_chain.invoke(
            {"query": question},
            callbacks=[tracer]
        )        
        return answer
    
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return f"Error: {e}"

def main():
    print("Welcome to PetGPT! Setting up the system, please wait...")
    qa_chain = setup_rag()
    
    if not qa_chain:
        print("System setup failed. Check logs for details.")
        sys.exit(1)
    
    print("Setup complete! Type 'quit' to exit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break
        print("\nThinking...")
        print("Answer:", answer_pet_question(qa_chain, question))

if __name__ == '__main__':
    # Verify LangSmith environment variables
    required_vars = [
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_ENDPOINT"
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    app.run(debug=True)