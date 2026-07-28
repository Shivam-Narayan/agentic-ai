# initial_setup.py

import os
#from llama_index.llms.vertex import Vertex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext


# Set environment variables
os.environ['GOOGLE_API_KEY'] = 'AQ.Ab8RN6IJtnMVryerLVLpypfd1BmgwI5bVek303KUNs8BWz1vPA'

# Set embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Set LLM model
from llama_index.llms.gemini import Gemini
Settings.llm = Gemini(model_name="models/gemini-pro-latest")

# Load documents
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader(input_files=['KT_document_from_a_real_client_project.docx']).load_data()

# Split text into chunks
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
Settings.text_splitter = text_splitter

index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter])

index.storage_context.persist(persist_dir="./indexing_data")

print("Indexing is saved")

