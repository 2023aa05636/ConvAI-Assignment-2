import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from rank_bm25 import BM25Okapi
import tempfile
import logging
logging.basicConfig(level=logging.INFO)

# Constants
DEFAULT_LLM_MODEL = "llama3"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_COLLECTION_NAME = "rag_collection"

# Placeholder for LlamaGuard
def is_harmful_request(text):
    harmful_keywords = ["violence", "hate speech", "self-harm", "explicit content"]
    for keyword in harmful_keywords:
        if keyword in text.lower():
            return True
    return False

class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

class RAG_Chatbot:
    def __init__(self, llm_model=DEFAULT_LLM_MODEL, base_url=DEFAULT_BASE_URL, collection_name=DEFAULT_COLLECTION_NAME):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.llm_model = llm_model
        self.base_url = base_url
        self.collection_name = collection_name
        self.embedding = ChromaDBEmbeddingFunction(OllamaEmbeddings(model=self.llm_model, base_url=self.base_url))
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))
        self.vector_store = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "A collection for RAG with Ollama"},
            embedding_function=self.embedding
        )
        self.bm25 = None
        self.documents = []

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.bm25 = None
        self.documents = []

    def split_document_into_chunks(self, document, chunk_size=512):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        chunks = splitter.split_text(document)
        return chunks

    def add_documents_to_collection(self, documents, ids):
        try:
            embeddings = self.embedding(documents)
            self.vector_store.add(embeddings=embeddings, ids=ids)
            logging.info(f"Successfully added {len(documents)} documents.")
        except Exception as e:
            logging.error(f"Error adding documents to collection: {e}")
            st.error(f"Error adding documents to collection: {e}")

    def ingest(self, pdf_path):
        docs = PyPDFLoader(file_path=pdf_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        collection_name = "rag_collection3"
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db3"))
        self.vector_store = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "A collection for RAG with Ollama"},
            embedding_function=self.embedding
        )

        chunked_documents = [chunk.page_content for chunk in chunks]
        chunked_ids = [str(i) for i in range(len(chunked_documents))]
        self.add_documents_to_collection(chunked_documents, chunked_ids)
        self.documents.extend(chunked_documents)
        self.bm25 = BM25Okapi([doc.split() for doc in self.documents])
        print(f"Ingested {len(chunks)} chunks from {pdf_path}")

    def read_and_save_file(self):
        self.clear()
        st.session_state["messages"] = []
        st.session_state["user_input"] = ""

        for file in st.session_state["file_uploader"]:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(file.getbuffer())
                file_path = tf.name

            with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
                self.ingest(file_path)
            os.remove(file_path)

    def query_chromadb(self, query_text, n_results=3):
        results = self.vector_store.query(query_texts=[query_text], n_results=n_results)
        print("Query Text:", query_text)
        print("Retrieved Documents:", results["documents"])
        return results["documents"], results["metadatas"]

    def query_bm25(self, query_text, n_results=3):
        if self.bm25 is None:
            return []
        query_tokens = query_text.split()
        scores = self.bm25.get_scores(query_tokens)
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        top_n_docs = [self.documents[i] for i in top_n_indices]
        return top_n_docs

    def query_ollama(self, prompt):
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke(prompt)

    def query_chromadb_all(self):
        count = self.vector_store.count()
        results = self.vector_store.query(query_texts=[""], n_results=count)
        return results["documents"], results["metadatas"]

    def rag_pipeline(self, query_text):
        if is_harmful_request(query_text):
            return "Request contains harmful or inappropriate content and cannot be processed."

        dense_retrieved_docs, dense_metadata = self.query_chromadb(query_text)
        sparse_retrieved_docs = self.query_bm25(query_text)
        print("######## Dense Retrieved Documents ########")
        print(dense_retrieved_docs)
        print("######## Sparse Retrieved Documents ########")
        print(sparse_retrieved_docs)

        # Ensure all documents are strings before concatenating
        dense_retrieved_docs = [
            " ".join(str(d) for d in doc if d is not None) if isinstance(doc, list) else str(doc) 
            for doc in dense_retrieved_docs if doc is not None
        ]
        sparse_retrieved_docs = [" ".join(doc) if isinstance(doc, list) else doc for doc in sparse_retrieved_docs]

        context = " ".join(dense_retrieved_docs + sparse_retrieved_docs) if dense_retrieved_docs or sparse_retrieved_docs else "No relevant documents found."
        print("######## Retrieved Context ########")
        print(context)

        augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
        print("######## Augmented Prompt ########")
        print(augmented_prompt)

        response = self.query_ollama(augmented_prompt)
        return response

    def generate_response(self, input_text):
        response = self.rag_pipeline(input_text)
        return response

    def display_chat_history(self):
        if "chat_history" not in st.session_state:
            st.session_state['chat_history'] = []

        st.write("## Chat History")

        for chat in st.session_state['chat_history']:
            st.write(f"**User**: {chat['user']}")
            st.write(f"**Assistant**: {chat['ollama']}")
            st.write("....")

def Page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = RAG_Chatbot()

    st.title("Financial Earnings Analysis Application")

    st.subheader("Upload a document")

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

    with st.form("llm-form"):
        text = st.text_area("Enter your question or statement:")
        submit = st.form_submit_button("Submit")

    if submit and text:
        with st.spinner("Generating response..."):
            response = st.session_state["assistant"].generate_response(text)
            st.session_state['chat_history'].append({"user": text, "ollama": response})
            st.write(response)

    st.session_state["assistant"].display_chat_history()

if __name__ == "__main__":
    Page()
