from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import TextLoader


class ChromaProcessor:
    _instance = None
    PERSISTED_DB = 'chromadb'

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
        # doc: https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.chroma.Chroma.html
        self.db = Chroma(
            persist_directory=self.PERSISTED_DB,
            embedding_function=self.embeddings
        )

    def load_document(self, document):
        # only text for now
        loader = TextLoader(document)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            # higher -> you have more relevant context, but increase token size 
            # lower -> you have less relevant context, but more diverse context            
            chunk_size=400, 
            chunk_overlap=0
        )
        docs = text_splitter.split_documents(documents)

        self.db.add_documents(docs)

    def search(self, query, k=5):
        # doc: https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.chroma.Chroma.html#langchain.vectorstores.chroma.Chroma.search
        return self.db.similarity_search(query, k)

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
