from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 1. Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load FAISS vector store
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model)

# 3. Load llama.cpp model
llm = LlamaCpp(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=512,
    n_ctx=2048,
    verbose=True,
)

# 4. Add conversational memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 5. Build Conversational RAG chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    verbose=True
)


while True:
    question = input("You: ")
    if question.lower() in ("exit", "quit"):
        break
    result = rag_chain.run(question)
    print("Assistant:", result)