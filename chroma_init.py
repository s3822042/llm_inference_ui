from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from chromadb.errors import InvalidDimensionException
from langchain.embeddings import TensorflowHubEmbeddings

class ChromaProcessor:
    _instance = None

    def __new__(cls, model_name='sentence-transformers/all-mpnet-base-v2'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize(model_name)
        return cls._instance

    def initialize(self, model_name):
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        self.db = None

    def process_and_persist(self, text_file_path):
        print(text_file_path)
        loader = TextLoader(text_file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        try:
            db = Chroma.from_documents(texts, self.embeddings)
            self.db = db
        except:
            Chroma().delete_collection()
            db = Chroma.from_documents(texts, self.embeddings)
            self.db = db
        print("Finish processing documents")
