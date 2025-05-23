import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import os
from datetime import datetime

# RAG Backend imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

class RAGChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BaseballCompanion")
        self.root.geometry("900x700")
        self.root.configure(bg='#1a1a2e')
        
        # Queue for thread communication
        self.response_queue = queue.Queue()
        
        # Initialize RAG components
        self.setup_rag()
        
        # Setup GUI
        self.setup_gui()
        
        # Start checking for responses
        self.check_response_queue()
        
    def setup_rag(self):
        """Initialize the RAG backend"""
        try:
            # Check API key
            if not os.getenv("OPENAI_API_KEY"):
                messagebox.showerror("Error", "OPENAI_API_KEY environment variable not set!\nPlease set it and restart the app.")
                return
            
            # Load embedding model
            self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Load FAISS vector store
            self.vectorstore = FAISS.load_local(
                "utils/faiss_index_may_23_2025/", 
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
            
            # Set up retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Initialize OpenAI Chat model
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1000,
            )
            
            # Create RAG prompt template
            self.rag_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an assistant for question-answering tasks. You are an expert baseball analyst and deeply understand baseball statistics and performance metrics. Use the following pieces of retrieved context to answer the question.  Use three sentences maximum and keep the answer concise.

Context: {context}"""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])
            
            # Memory for conversation history
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Create the RAG chain
            self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()
            
            self.rag_initialized = True
            
        except Exception as e:
            messagebox.showerror("RAG Initialization Error", f"Failed to initialize RAG system:\n{str(e)}")
            self.rag_initialized = False
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Configure ttk style for dark theme
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Title.TLabel', 
                       background='#1a1a2e', 
                       foreground='#ffffff', 
                       font=('SF Pro Display', 20, 'bold'))
        
        style.configure('Custom.TFrame', 
                       background='#1a1a2e',
                       relief='flat')
        
        style.configure('Chat.TFrame',
                       background='#16213e',
                       relief='solid',
                       borderwidth=1)
        
        style.configure('Input.TFrame',
                       background='#1a1a2e')
        
        style.configure('Custom.TEntry',
                       fieldbackground='#16213e',
                       foreground='#ffffff',
                       bordercolor='#0f3460',
                       lightcolor='#0f3460',
                       darkcolor='#0f3460',
                       font=('SF Pro Text', 12))
        
        style.configure('Send.TButton',
                       background='#e94560',
                       foreground='#ffffff',
                       font=('SF Pro Text', 11, 'bold'),
                       borderwidth=0,
                       focuscolor='none')
        
        style.map('Send.TButton',
                 background=[('active', '#ff6b7a'),
                           ('pressed', '#d63447')])
        
        style.configure('Status.TLabel',
                       background='#1a1a2e',
                       foreground='#8892b0',
                       font=('SF Pro Text', 10))
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20", style='Custom.TFrame')
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title with baseball emoji
        title_label = ttk.Label(main_frame, text="⚾ BaseballCompanion", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Chat display area
        self.chat_frame = ttk.Frame(main_frame, style='Chat.TFrame')
        self.chat_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        self.chat_frame.columnconfigure(0, weight=1)
        self.chat_frame.rowconfigure(0, weight=1)
        
        # Chat text area with scrollbar
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame, 
            wrap=tk.WORD, 
            width=85, 
            height=28,
            font=('SF Pro Text', 12),
            bg='#0f1419',
            fg='#ffffff',
            insertbackground='#ffffff',
            selectbackground='#264f78',
            selectforeground='#ffffff',
            relief='flat',
            borderwidth=15
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=15, pady=15)
        self.chat_display.config(state=tk.DISABLED)
        
        # Input frame
        input_frame = ttk.Frame(main_frame, style='Input.TFrame')
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        input_frame.columnconfigure(0, weight=1)
        
        # Input field
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(
            input_frame, 
            textvariable=self.input_var,
            style='Custom.TEntry',
            font=('SF Pro Text', 12)
        )
        self.input_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 15))
        self.input_entry.bind('<Return>', self.send_message)
        
        # Send button
        self.send_button = ttk.Button(
            input_frame, 
            text="Send 🚀", 
            command=self.send_message,
            style='Send.TButton'
        )
        self.send_button.grid(row=0, column=1)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready" if self.rag_initialized else "RAG system not initialized")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              style='Status.TLabel')
        status_bar.grid(row=3, column=0, sticky=tk.W)
        
        # Focus on input
        self.input_entry.focus()
        
        # Add welcome message
        self.add_message("BaseballCompanion", "Hello! I'm your personal baseball assistant. Ask me anything about baseball! ⚾", "#64b5f6")
    
    def add_message(self, sender, message, color="#2c3e50"):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Add sender and message
        self.chat_display.insert(tk.END, f"[{timestamp}] {sender}: ", f"sender_{sender}")
        self.chat_display.insert(tk.END, f"{message}\n\n")
        
        # Configure tag colors for modern dark theme
        self.chat_display.tag_config(f"sender_{sender}", 
                                   foreground=color, 
                                   font=('SF Pro Text', 12, 'bold'))
        
        # Scroll to bottom
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def add_sources(self, docs):
        """Add source information to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        self.chat_display.insert(tk.END, "📚 Sources:\n", "sources_header")
        
        for i, doc in enumerate(docs, 1):
            # Show metadata if available
            source_info = f"Source {i}: "
            if hasattr(doc, 'metadata') and doc.metadata:
                source_info += " | ".join([f"{k}: {v}" for k, v in doc.metadata.items() if v])
            
            self.chat_display.insert(tk.END, f"{source_info}\n", "sources")
            
            # Show content preview
            content = doc.page_content
            if len(content) > 150:
                content = content[:150] + "..."
            self.chat_display.insert(tk.END, f"   {content}\n\n", "sources_content")
        
        # Configure source tags with modern colors
        self.chat_display.tag_config("sources_header", 
                                   foreground="#ffa726", 
                                   font=('SF Pro Text', 11, 'bold'))
        self.chat_display.tag_config("sources", 
                                   foreground="#ab47bc", 
                                   font=('SF Pro Text', 10, 'bold'))
        self.chat_display.tag_config("sources_content", 
                                   foreground="#90a4ae", 
                                   font=('SF Pro Text', 10))
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def send_message(self, event=None):
        """Send a message and get response"""
        if not self.rag_initialized:
            messagebox.showerror("Error", "RAG system not initialized properly.")
            return
        
        message = self.input_var.get().strip()
        if not message:
            return
        
        # Clear input
        self.input_var.set("")
        
        # Add user message to display
        self.add_message("You", message, "#4fc3f7")
        
        # Disable send button and show status
        self.send_button.config(state=tk.DISABLED)
        self.status_var.set("Generating response...")
        
        # Start response generation in separate thread
        threading.Thread(target=self.generate_response, args=(message,), daemon=True).start()
    
    def generate_response(self, question):
        """Generate response using RAG (runs in separate thread)"""
        try:
            # Get relevant documents
            docs = self.retriever.invoke(question)
            
            # Create input for RAG chain
            chain_input = {
                "context": self.format_docs(docs),
                "question": question,
                "chat_history": self.get_chat_history()
            }
            
            # Get response
            response = self.rag_chain.invoke(chain_input)
            
            # Add to memory
            self.add_to_memory(question, response)
            
            # Put results in queue
            self.response_queue.put(("success", response, docs))
            
        except Exception as e:
            self.response_queue.put(("error", str(e), None))
    
    def check_response_queue(self):
        """Check for responses from the RAG system"""
        try:
            while True:
                result_type, response, docs = self.response_queue.get_nowait()
                
                if result_type == "success":
                    # Add sources first
                    if docs:
                        self.add_sources(docs)
                    
                    # Add assistant response
                    self.add_message("BaseballCompanion", response, "#64b5f6")
                    
                elif result_type == "error":
                    self.add_message("System", f"Error: {response}", "#ff5722")
                
                # Re-enable send button
                self.send_button.config(state=tk.NORMAL)
                self.status_var.set("Ready")
                
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_response_queue)
    
    def format_docs(self, docs):
        """Format documents for context"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def get_chat_history(self):
        """Get chat history from memory"""
        return self.memory.chat_memory.messages
    
    def add_to_memory(self, question, answer):
        """Add question and answer to memory"""
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(answer)

def main():
    root = tk.Tk()
    app = RAGChatApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()