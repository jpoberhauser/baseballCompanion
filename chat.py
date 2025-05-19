from langchain.llms import LlamaCpp

llm = LlamaCpp(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=512,
    n_ctx=2048,
    verbose=True,
)



from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

template = """You are a helpful baseball assistant.

{question}

Answer:"""

prompt = PromptTemplate(input_variables=["question"], template=template)
chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run("Who won the 2016 World Series?")
print(response)



from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

print(conversation.run("Who is Shohei Ohtani?"))
print(conversation.run("Where does he play now?"))