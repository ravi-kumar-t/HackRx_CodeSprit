import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import warnings

load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    raise ValueError("GOOGLE_API_KEY not found.")
genai.configure(api_key=gemini_api_key)

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")

def create_rag_chain():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})  # increased

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.0)
    template = """
    You are an expert for "Arogya Sanjeevani Policy".
    Give decision and short justification in one line: Yes, No, or More information needed.

    Context:
    {context}

    Question:
    {question}

    Output: Decision, justification
    """
    prompt = PromptTemplate.from_template(template)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def handle_query(query: str):
    try:
        rag_chain = create_rag_chain()
        print(f"\nQuery: {query}")
        print("\nAnswer:", rag_chain.invoke(query))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    while True:
        q = input("Ask a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        handle_query(q)
