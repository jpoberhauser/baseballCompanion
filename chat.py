from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain_community.chat_models import ChatLlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import multiprocessing
# 1. Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load FAISS vector store
vectorstore = FAISS.load_local("utils/faiss_index_may_21_2025/", 
                               embeddings=embedding_model,
                               allow_dangerous_deserialization=True)
# 3. Load llama.cpp model
# llm = LlamaCpp(
#      model_path="llama.cpp/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#     temperature=0.7,
#     max_tokens=1024,
#     n_ctx=2048,
#     verbose=True,
# )
llm = ChatLlamaCpp(
    temperature=0.5,
    model_path='llama.cpp/models/Hermes-2-Pro-Llama-3-8B-Q8_0.gguf',
    n_ctx=10000,
    n_gpu_layers=8,
    n_batch=300,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    max_tokens=1028,
    n_threads=multiprocessing.cpu_count() - 2,
    repeat_penalty=1.5,
    top_p=0.5,
    verbose=True,
)
# 4. Add conversational memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""<|system|>
You are a helpful baseball assistant.
<|user|>
Use the following context to answer the question.

Context:
{context}

Question:
{question}
<|assistant|>"""
)


# 5. Build Conversational RAG chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    verbose=True
)


print("Ask your baseball questions. Type 'exit' or 'quit' to stop.\n")
while True:
    question = input("You: ")
    if question.lower() in ("exit", "quit"):
        break
    result = rag_chain.run(question)
    print("Assistant:", result)