__import__('pysqlite3')
import sys
# Redirect sqlite3 to use pysqlite3 for compatibility
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
import requests
import logging
import tempfile
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from rank_bm25 import BM25Okapi
import json

# Set up logging to help with debugging and status messages.
logging.basicConfig(level=logging.INFO)

# Constants for default API endpoints and collection names.
DEFAULT_LLM_API_URL = "https://pawwi-conv-ai-assignment-2-llm-api.hf.space/generate/"  # Replace with your API URL
DEFAULT_COLLECTION_NAME = "rag_collection"

# Function to check for harmful or inappropriate requests.
def is_harmful_request(text):
    harmful_keywords = ["violence", "hate speech", "self-harm", "explicit content"]
    return any(keyword in text.lower() for keyword in harmful_keywords)

# Embedding function using SentenceTransformerEmbeddings for ChromaDB.
class ChromaDBEmbeddingFunction:
    def __init__(self):
        # Initialize the sentence transformer model.
        self.model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def __call__(self, input):
        # Embed input text(s); supports both a single string and a list of strings.
        return self.model.embed_documents([input] if isinstance(input, str) else input)

# Main class for the Retrieval-Augmented Generation (RAG) Chatbot.
class RAG_Chatbot:
    def __init__(self, collection_name=DEFAULT_COLLECTION_NAME, llm_api_url=DEFAULT_LLM_API_URL):
        # Initialize text splitter for document chunking.
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.collection_name = collection_name
        self.llm_api_url = llm_api_url
        self.embedding = ChromaDBEmbeddingFunction()
        
        # Create or get an existing collection for RAG using ChromaDB.
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))
        self.vector_store = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "A collection for RAG"},
            embedding_function=self.embedding
        )
        # BM25 is used for sparse retrieval and confidence scoring.
        self.bm25 = None
        self.documents = []
        self.list_collections()

    def list_collections(self):
        """Prints and returns the list of available collections."""
        print(self.chroma_client.list_collections())
        return self.chroma_client.list_collections()

    def clear(self):
        """Clears the vector store, retriever, chain, BM25 index, and loaded documents."""
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.bm25 = None
        self.documents = []

    def split_document_into_chunks(self, document, chunk_size=512):
        """Splits a document into smaller chunks based on the specified chunk size."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        chunks = splitter.split_text(document)
        return chunks

    def add_documents_to_collection(self, documents, ids):
        """
        Adds documents to the vector store after computing embeddings.
        Uses provided ids to index the documents.
        """
        try:
            embeddings = self.embedding(documents)
            self.vector_store.add(embeddings=embeddings, ids=ids)
            logging.info(f"Successfully added {len(documents)} documents.")
        except Exception as e:
            logging.error(f"Error adding documents to collection: {e}")
            st.error(f"Error adding documents to collection: {e}")

    def ingest(self, pdf_path):
        """
        Loads a PDF, splits it into chunks, filters metadata, creates a new collection,
        and indexes the document chunks.
        """
        # Load PDF document
        docs = PyPDFLoader(file_path=pdf_path).load()
        # Split the document into manageable chunks
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        # Define a new collection name for this ingestion process.
        collection_name = "rag_collection3"
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db3"))
        self.vector_store = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "A collection for RAG with Ollama"},
            embedding_function=self.embedding
        )

        # Extract text from chunks and generate unique IDs.
        chunked_documents = [chunk.page_content for chunk in chunks]
        chunked_ids = [str(i) for i in range(len(chunked_documents))]
        self.add_documents_to_collection(chunked_documents, chunked_ids)
        # Extend the list of documents for BM25 indexing.
        self.documents.extend(chunked_documents)
        # Build BM25 index on the tokenized documents.
        self.bm25 = BM25Okapi([doc.split() for doc in self.documents])
        print(f"Ingested {len(chunks)} chunks from {pdf_path}")

    def read_and_save_file(self):
        """
        Reads files uploaded via Streamlit, ingests each PDF, and clears previous data.
        """
        self.clear()
        st.session_state["messages"] = []
        st.session_state["user_input"] = ""

        for file in st.session_state["file_uploader"]:
            # Create a temporary file for the uploaded PDF.
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(file.getbuffer())
                file_path = tf.name

            # Use spinner UI to indicate ingestion progress.
            with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
                self.ingest(file_path)
            # Remove the temporary file after ingestion.
            os.remove(file_path)

    def query_chromadb(self, query_text, n_results=3):
        """
        Queries the dense vector store (ChromaDB) with the given query text.
        Returns the retrieved documents and their metadata.
        """
        results = self.vector_store.query(query_texts=[query_text], n_results=n_results)
        print("Query Text:", query_text)
        print("Retrieved Documents:", results["documents"])
        return results["documents"], results["metadatas"]

    def query_bm25(self, query_text, n_results=3):
        """
        Queries the BM25 index for sparse retrieval.
        Returns the top n_results based on BM25 scoring.
        """
        if self.bm25 is None:
            return []
        query_tokens = query_text.split()
        scores = self.bm25.get_scores(query_tokens)
        # Get indices of the top-scoring documents.
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        return [self.documents[i] for i in top_n_indices]

    def query_llm(self, prompt):
        """
        Sends the augmented prompt to the LLM API and returns the response.
        Includes error handling for request and JSON decoding errors.
        """
        try:
            response = requests.post(self.llm_api_url, json={"prompt": prompt})
            print("Response Status Code:", response.status_code)
            print("Response Text:", response.text)  # Debugging line
            # print("Response Confidence Score:", response.text.confidence_score)  # Debugging line
            response_json = response.json()  # Parse JSON response
            return response_json.get("response", "No response from LLM.")
        except requests.exceptions.RequestException as e:
            logging.error(f"LLM API Request Error: {e}")
            return "Error communicating with LLM."
        except ValueError as e:  # JSON decoding error
            logging.error(f"LLM API Response Error: {e}")
            return f"Invalid response from LLM: {response.text}"
    
    def query_chromadb_all(self):
        """
        Retrieves all documents from the vector store.
        """
        count = self.vector_store.count()
        results = self.vector_store.query(query_texts=[""], n_results=count)
        return results["documents"], results["metadatas"]

    def get_memory_context(self):
        """
        Retrieves previous conversation history from session state to be used as memory in the prompt.
        """
        memory_context = ""
        if "chat_history" in st.session_state:
            for chat in st.session_state["chat_history"]:
                memory_context += f"User: {chat['user']}\nAssistant: {chat['ollama']}\n"
        return memory_context

    def get_confidence_score(self, query_text):
        """
        Computes a confidence score using BM25 scores.
        Returns the maximum BM25 score as the confidence measure.
        """
        if self.bm25 is None:
            return 0.0
        query_tokens = query_text.split()
        scores = self.bm25.get_scores(query_tokens)
        if scores:
            return max(scores)
        return 0.0

    def rag_pipeline(self, query_text):
        """
        The main retrieval-augmented generation pipeline.
        - Checks for harmful content.
        - Retrieves documents using both dense and sparse methods.
        - Augments the prompt with context and previous conversation (memory).
        - Queries the LLM and returns the final response with a confidence score.
        """
        if is_harmful_request(query_text):
            return "Request contains harmful or inappropriate content and cannot be processed."

        # Retrieve context using dense and sparse methods.
        dense_retrieved_docs, dense_metadata = self.query_chromadb(query_text)
        sparse_retrieved_docs = self.query_bm25(query_text)
        print("######## Dense Retrieved Documents ########")
        print(dense_retrieved_docs)
        print("######## Sparse Retrieved Documents ########")
        print(sparse_retrieved_docs)

        # Convert all retrieved documents to string format.
        dense_retrieved_docs = [
            " ".join(str(d) for d in doc if d is not None) if isinstance(doc, list) else str(doc) 
            for doc in dense_retrieved_docs if doc is not None
        ]
        sparse_retrieved_docs = [" ".join(doc) if isinstance(doc, list) else doc for doc in sparse_retrieved_docs]

        # Combine dense and sparse retrieved documents as context.
        context = " ".join(dense_retrieved_docs + sparse_retrieved_docs) if (dense_retrieved_docs or sparse_retrieved_docs) else "No relevant documents found."
        print("######## Retrieved Context ########")
        print(context)

        # Include previous conversation history as memory.
        memory_context = self.get_memory_context()
        print("######## Memory Context ########")
        print(memory_context)

        # Build the augmented prompt for the LLM.
        augmented_prompt = f"Memory: {memory_context}\n\nContext: {context}\n\nQuestion: {query_text}. Provide only the answer and add the explanation only if it looks necessary and appropriate.\nAnswer:"
        print("######## Augmented Prompt ########")
        print(augmented_prompt)

        # Get response from LLM.
        response = self.query_llm(augmented_prompt)
        # Compute confidence score using BM25.
        # confidence = self.get_confidence_score(query_text)
        response_json = json.loads(response)
        response_text = response_json["response"]
        confidence_score = response_json["confidence_score"]
        final_response = f"Confidence Score: {confidence_score:.2f}\n\n{response_text}"
        return final_response

    def generate_response(self, input_text):
        """Generates a response by running the full RAG pipeline."""
        return self.rag_pipeline(input_text)
    
    def display_chat_history(self):
        """Displays the conversation history on the Streamlit app."""
        if "chat_history" not in st.session_state:
            st.session_state['chat_history'] = []

        st.write("## Chat History")

        for chat in st.session_state['chat_history']:
            st.write(f"**User**: {chat['user']}")
            st.write(f"**Assistant**: {chat['ollama']}")
            st.write("....")

def Page():
    """
    Main function for the Streamlit app.
    Handles file uploads, form submissions, and displaying responses and chat history.
    """
    # Initialize session state and assistant if not already done.
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = RAG_Chatbot()

    st.title("Financial Earnings Analysis Application")

    st.subheader("Upload a document")

    # File uploader for PDF documents.
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=lambda: st.session_state["assistant"].read_and_save_file(),
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    st.header("Financial Earning Analysis Responses")

    # Form for user input queries.
    with st.form("llm-form"):
        text = st.text_area("Enter your question or statement:")
        submit = st.form_submit_button("Submit")

    # On form submission, generate a response and update the chat history.
    if submit and text:
        with st.spinner("Generating response..."):
            response = st.session_state["assistant"].generate_response(text)
            st.session_state['chat_history'].append({"user": text, "ollama": response})
            st.write(response)

    # Display the conversation history.
    st.session_state["assistant"].display_chat_history()

# Run the Streamlit app if this file is executed as the main program.
if __name__ == "__main__":
    Page()
