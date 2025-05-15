# !pip install sentence-transformers faiss-cpu
import faiss
import pickle

from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
### read chunks into below
import glob
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import  util

all_transcripts = glob.glob(f'{ROOT_DIR}/data/transcripts/*.txt')


all_raw_docs = []
for i in all_transcripts:
    with open(i, "r", encoding="utf-8") as f:
        text = f.read()  # `text` is one big string
    all_raw_docs.append([text.replace('\n', '')])

documents = [Document(inner[0]) for inner in all_raw_docs]


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(documents)

all_chunks = [x.page_content for x in all_splits]
embeddings = model.encode(all_chunks)