import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#==================================================================
#====================Load Vector DB================================
#==================================================================
PERSIST_DIR = r"C:\Users\VM116ZZ\PycharmProjects\POC\vectordb"
TOP_K = 3

def load_vectordb():
    embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL_NAME"))
    vectordb = Chroma(
        collection_name="rag_contexts",
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vectordb.as_retriever(search_kwargs={"k": TOP_K})
