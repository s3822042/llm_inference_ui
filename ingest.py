import os
import streamlit as st
import whisper

from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import JSONLoader
from chromadb.errors import InvalidDimensionException
import json

ALLOWED_AUDIO_EXTENIONS = None # ["mp3", "wav"]
ALLOW_MULTIPLE_FILES = True
FILE_STORAGE_PATH = "./files"

model = None

with st.form("my-form", clear_on_submit=True):
    uploaded_files = st.file_uploader(
        "Choose a file", 
        type=ALLOWED_AUDIO_EXTENIONS,
        accept_multiple_files=ALLOW_MULTIPLE_FILES
        )
    
    submitted = st.form_submit_button("submit")
    
def upload_file(name, bytes):
    file_path = f'{FILE_STORAGE_PATH}/{name}'
    
    # convert file path to file object

    if (os.path.exists(file_path)):
        st.warning(f'File {name} already exists.')
        return
    
    with open(file_path, 'wb') as f:
        f.write(bytes)
        st.success(f'File {name} successfully uploaded.')

def get_model(version="base"):
    global model
    if (model == None):
        model = whisper.load_model(version)
    
    return model

def transcript_file(name):
    file_path = f'{FILE_STORAGE_PATH}/{name}'
    data_load_state = st.text('transcribing data...')
    options = {
        "language": "English"
    }
    result = get_model().transcribe(
        file_path,
        **options
    )
    data_load_state.text('transcribing data...done!')
    return result

def ingest_into_db(transcription):
    # Load and process the text
    loader = JSONLoader(transcription['segments'])
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        db = Chroma.from_documents(texts, embedding_function)
    except InvalidDimensionException:
        Chroma().delete_collection()
        db = Chroma.from_documents(texts, embedding_function)

    docs = db.similarity_search("Summarize the conversation!", k=5)

    st.divider()
    st.write(docs)

for uploaded_file in uploaded_files:
    upload_file(uploaded_file.name, uploaded_file.read())
    transcription = transcript_file(uploaded_file.name)
    ingest_into_db(transcription)
