from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
import os

# 1. Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load FAISS vector store
vectorstore = FAISS.load_local("utils/faiss_index_may_23_2025/", 
                               embeddings=embedding_model,
                               allow_dangerous_deserialization=True)

# 3. Initialize OpenAI Chat model
# Make sure to set your OPENAI_API_KEY environment variable
# export OPENAI_API_KEY="your-api-key-here"
llm = ChatOpenAI(
    model="gpt-4o-mini",  # or "gpt-4o", "gpt-3.5-turbo"
    temperature=0.7,
    max_tokens=1000,
)

# 4. Set up retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Create RAG prompt template
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 

Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# 6. Memory for conversation history
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 7. Helper functions
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_chat_history(memory):
    """Convert memory to chat history format"""
    messages = memory.chat_memory.messages
    return messages

def add_to_memory(memory, question, answer):
    """Add question and answer to memory"""
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(answer)

# 8. Create the RAG chain with proper context handling
def create_rag_chain_input(question):
    """Create the input dictionary for the RAG chain"""
    return {
        # "context": format_docs(retriever.invoke(question)),
        "context": '',
        "question": question,
        "chat_history": get_chat_history(memory)
    }

# Simplified RAG chain
rag_chain = rag_prompt | llm | StrOutputParser()

def display_sources(docs):
    """Display the source documents"""
    print("\n" + "="*60)
    print("RELEVANT SOURCES:")
    print("="*60)
    for i, doc in enumerate(docs, 1):
        print(f"\nSource {i}:")
        print("-" * 40)
        # Show metadata if available
        if hasattr(doc, 'metadata') and doc.metadata:
            for key, value in doc.metadata.items():
                print(f"{key}: {value}")
            print("-" * 40)
        
        # Show content (truncated if too long)
        content = doc.page_content
        # if len(content) > 300:
        #     content = content[:300] + "..."
        print(f"Content: {content}")
    print("="*60 + "\n")

def chat_with_rag():
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("RAG-enabled Chat Assistant initialized!")
    print("Using OpenAI GPT model")
    print("Ask your questions. Type 'exit' or 'quit' to stop.\n")
    
    while True:
        question = input("You: ")
        if question.lower() in ("exit", "quit"):
            break
        
        try:
            # Get relevant documents using the modern invoke method
            docs = retriever.invoke(question)
            
            # Display sources
            display_sources(docs)
            
            print("Generating response...")
            
            # Create input for RAG chain
            chain_input = create_rag_chain_input(question)
            
            # Get answer from RAG chain
            result = rag_chain.invoke(chain_input)
            
            # Add to memory
            add_to_memory(memory, question, result)
            
            print(f"Assistant: {result}\n")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    chat_with_rag()