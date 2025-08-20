import os
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "./vectordb"


def get_md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def ingest_docs_to_chroma(file_paths):
    full_path = os.path.abspath(PERSIST_DIR)
    os.makedirs(full_path, exist_ok=True)

    # 1) Load PDFs or Word docs
    all_docs = []
    for file in file_paths:
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file)
        else:
            loader = UnstructuredWordDocumentLoader(file)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = os.path.basename(file)         # Add filename in metadata for deduplication tracking
        all_docs.extend(docs)

    # 2) Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(all_docs)

    # 3) Assign unique ID to each chunk (based on hash)
    for doc in split_docs:
        content_hash = get_md5(doc.page_content)
        doc.metadata["doc_id"] = content_hash

    # 4) Load existing Chroma DB
    embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL_NAME"))
    vectordb = Chroma(
        collection_name="rag_contexts",
        persist_directory=full_path,
        embedding_function=embeddings
    )

    # 5) Check existing doc_ids and avoid duplicates
    existing_ids = set()
    if vectordb._collection.count() > 0:
        existing = vectordb._collection.get(include=['metadatas'])
        for metadata in existing['metadatas']:
            if metadata and "doc_id" in metadata:
                existing_ids.add(metadata["doc_id"])

    new_docs = [doc for doc in split_docs if doc.metadata["doc_id"] not in existing_ids]

    if new_docs:
        BATCH_LIMIT = 5000
        for i in range(0, len(new_docs), BATCH_LIMIT):
            batch = new_docs[i:i + BATCH_LIMIT]
            vectordb.add_documents(batch)

        print(f"Added {len(new_docs)} new document chunks.")
    else:
        print("No new documents to add. Everything already exists.")

    print(f"Chroma vector DB at: {full_path}")


if __name__ == "__main__":
    files = [
        r"context_files/Selenium Full Material.pdf",
        r"context_files/List-of-Presidents-of-India.pdf",
        r"context_files/Java_8.pdf",
        r"context_files/javanotes8.pdf",
        r"context_files/software_testing_tutorial.pdf"
    ]
    ingest_docs_to_chroma(files)
