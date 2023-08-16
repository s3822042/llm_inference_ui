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

class IngestionTranscription:
    def __init__(self):
        self.ALLOWED_AUDIO_EXTENSIONS = None  # ["mp3", "wav"]
        self.ALLOW_MULTIPLE_FILES = True
        self.FILE_STORAGE_PATH = "./files"
        self.model = None

    def run(self):
        with st.form("my-form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "Choose a file",
                type=self.ALLOWED_AUDIO_EXTENSIONS,
                accept_multiple_files=self.ALLOW_MULTIPLE_FILES,
            )

            submitted = st.form_submit_button("submit")

        if uploaded_files:
            for uploaded_file in uploaded_files:
                self.upload_file(uploaded_file.name, uploaded_file.read())
                transcription = self.transcript_file(uploaded_file.name)
                # self.ingest_into_db(transcription)

    def upload_file(self, name, bytes):
        file_path = os.path.join(self.FILE_STORAGE_PATH, name)

        if os.path.exists(file_path):
            st.warning(f"File {name} already exists.")
            return

        with open(file_path, "wb") as f:
            f.write(bytes)
            st.success(f"File {name} successfully uploaded.")

    def get_model(self, version="base"):
        if self.model is None:
            self.model = whisper.load_model(version)
        return self.model

    def transcript_file(self, name):
        file_path = os.path.join(self.FILE_STORAGE_PATH, name)
        data_load_state = st.text("transcribing data...")
        options = {"language": "English"}
        result = self.get_model().transcribe(file_path, **options)
        data_load_state.text("transcribing data...done!")
        return result

    # def ingest_into_db(self, transcription):
    # # Load and process the text
    # loader = JSONLoader(transcription['segments'])
    # documents = loader.load()

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    # texts = text_splitter.split_documents(documents)
    # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # try:
    #     db = Chroma.from_documents(texts, embedding_function)
    # except InvalidDimensionException:
    #     Chroma().delete_collection()
    #     db = Chroma.from_documents(texts, embedding_function)

    # docs = db.similarity_search("Summarize the conversation!", k=5)

    # st.divider()
    # st.write(docs)

if __name__ == "__main__":
    app = IngestionTranscription()
    app.run()
