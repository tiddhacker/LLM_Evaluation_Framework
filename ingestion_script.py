import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Change this to where you want the vectordb folder
PERSIST_DIR = "./vectordb"

def ingest_docs_to_chroma(file_paths):
    # 0) Ensure the folder exists
    full_path = os.path.abspath(PERSIST_DIR)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Created directory: {full_path}")
    else:
        print(f"Using existing directory: {full_path}")

    # 1) load PDFs or Word docs
    all_docs = []
    for file in file_paths:
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file)
        else:
            loader = UnstructuredWordDocumentLoader(file)
        docs = loader.load()
        all_docs.extend(docs)

    # 2) split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(all_docs)

    # 3) embed and store in persistent Chroma
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(
        collection_name="rag_contexts",
        persist_directory=full_path,
        embedding_function=embeddings
    )
    vectordb.add_documents(split_docs)
    vectordb.persist()

    print(f"Chroma vector DB stored at: {full_path}")

if __name__ == "__main__":
    files = [
        r"context_files/Java_8.pdf",
        r"context_files/javanotes8.pdf",
    ]
    ingest_docs_to_chroma(files)