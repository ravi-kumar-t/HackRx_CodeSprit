import os
import requests
import time
import asyncio
import json
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict

from dotenv import load_dotenv
import google.generativeai as genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document

# --- 1. CONFIGURATION ---
load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment.")
genai.configure(api_key=gemini_api_key)

app = FastAPI(
    title="HackRx 6.0 Query-Retrieval System",
    description="An LLM-powered RAG system for insurance policy documents.",
    version="4.0.0",
)

DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "documents")
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise ValueError("API_TOKEN not found in environment.")
security = HTTPBearer()

# --- 2. REQUEST MODEL ---
class HackRxRequest(BaseModel):
    documents: List[HttpUrl]
    questions: List[str]

# --- 3. RESPONSE MODEL ---
class UserFriendlyAnswer(BaseModel):
    message: str = Field(description="A concise, self-contained answer in plain English, 1–3 sentences.")

# --- 4. HELPERS ---
def _parse_llm_json_output(llm_output: str) -> dict:
    cleaned_output = llm_output.strip().strip("```json").strip("```")
    try:
        return json.loads(cleaned_output)
    except json.JSONDecodeError:
        print(f"⚠ JSON parsing failed. Raw output:\n{cleaned_output}")
        raise ValueError("Invalid JSON returned from LLM.")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

def _get_chunks_from_url(url: str) -> List[Document]:
    """Downloads a PDF, loads it, and splits it into chunks."""
    try:
        print(f"📄 Downloading document: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        temp_file = os.path.join(DOCUMENTS_DIR, "temp_policy.pdf")
        with open(temp_file, "wb") as f:
            f.write(response.content)

        loader = PyPDFLoader(temp_file)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(documents)
        print(f"✅ Document split into {len(chunks)} chunks.")

        os.remove(temp_file)
        return chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed for URL '{url}': {e}")


def filter_context(question: str, docs: List[Document]) -> str:
    """Keep only chunks that are highly relevant to the question."""
    question_lower = question.lower()
    relevant_chunks = [
        doc.page_content for doc in docs
        if any(word in doc.page_content.lower() for word in question_lower.split())
    ]
    return "\n\n".join(relevant_chunks[:3])

# --- 5. CHAINS ---
def create_query_rewriter_chain(llm):
    prompt = PromptTemplate.from_template("""
        You are a query rewriter. Rewrite the question into 2–3 alternative queries for better retrieval.
        Output only JSON: {{"queries": ["query1", "query2", "query3"]}}

        Original Question: {question}
    """)
    return prompt | llm

def create_rag_chain(llm):
    parser = PydanticOutputParser(pydantic_object=UserFriendlyAnswer)

    template = """
    You are an expert insurance claims adjuster. 
    Answer the question strictly from the provided CONTEXT.

    RULES:
    - Provide a complete, self-contained answer in plain English.
    - If the question is about coverage, start with "Yes" or "No" only if the policy clearly states it.
    - Include any specific limits, durations, conditions, or exclusions mentioned in the context.
    - Keep the answer concise but informative (1–3 sentences).
    - If the information is missing, say "The policy does not specify this."
    - Do not guess or add information not in the context.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    return prompt | llm | parser

# --- 6. PROCESS SINGLE QUESTION ---
async def process_single_question(question: str, retriever, rewriter_chain, rag_chain):
    try:
        print(f"\n❓ Processing: {question}")
        rewritten_output = await asyncio.to_thread(
            rewriter_chain.invoke, {"question": question}
        )
        rewritten_queries_json = _parse_llm_json_output(
            rewritten_output.content if hasattr(rewritten_output, "content") else str(rewritten_output)
        )
        search_queries = [question] + rewritten_queries_json.get("queries", [])

        retrieved_docs = []
        for q in search_queries:
            retrieved_docs.extend(await asyncio.to_thread(retriever.invoke, q))

        unique_docs = {doc.page_content: doc for doc in retrieved_docs}.values()

        filtered_context = filter_context(question, unique_docs)

        response = await asyncio.to_thread(
            rag_chain.invoke, {"context": filtered_context, "question": question}
        )
        print(f"✅ Finished: {question}")
        return response.message

    except Exception as e:
        print(f"❌ Error processing '{question}': {e}")
        return "An error occurred while processing this question."

# --- 7. API ENDPOINT ---
@app.post("/hackrx/run", response_model=Dict[str, List[str]])
async def run_hackrx_submission(
    request_body: HackRxRequest,
    is_authenticated: bool = Depends(verify_token)
):
    print("🚀 Starting request.")
    start_time = time.time()
    
    all_chunks = []
    for doc_url in request_body.documents:
        # Process each document and collect all chunks
        all_chunks.extend(_get_chunks_from_url(str(doc_url)))

    # Create a single vector store from all the chunks from all documents
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.0)
    rewriter_chain = create_query_rewriter_chain(llm)
    rag_chain = create_rag_chain(llm)

    tasks = [
        process_single_question(q, retriever, rewriter_chain, rag_chain)
        for q in request_body.questions
    ]
    answers = await asyncio.gather(*tasks)

    print(f"✅ All questions processed in {time.time() - start_time:.2f}s.")
    return {"answers": answers}