import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# CONFIGURATION
# ============================================================

# Set GOOGLE_API_KEY in your environment before running this script.
GOOGLE_API_KEY = "AQ.Ab8RN6J9p8rBGPvIaMeYzJxw3ZqLoKq6Uakj47xQtjQii3cFsg"
PDF_FILE = "Novatech_Innovations_Demo_Company_Profile(1).pdf"
PDF_FILE = "Kerala_Cabinet_Ministers_2026 (1).pdf"
DF_FILE = "SURYA Presume (1).pdf"
FAISS_FOLDER = "faiss_store"

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is missing. Set it as an environment variable before running."
    )

# ============================================================
# LOAD PDF
# ============================================================

print("Loading PDF...")

loader = PyPDFLoader(PDF_FILE)
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# ============================================================
# SPLIT DOCUMENTS
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================
# FAISS VECTOR STORE
# ============================================================

if os.path.exists(FAISS_FOLDER):
    print("Loading existing FAISS index...")
    vectorstore = FAISS.load_local(
        FAISS_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("Creating new FAISS index...")
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(FAISS_FOLDER)
    print("FAISS index saved")

# ============================================================
# RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ============================================================
# GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# ============================================================
# TEST WITHOUT RAG
# ============================================================

test_question = "Who is the CEO of Novatech Innovations?"
direct_response = llm.invoke(test_question)

print("\n===== DIRECT LLM RESPONSE =====")
print(direct_response.content)
print("===============================\n")

# ============================================================
# PROMPT TEMPLATE
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI assistant.

    Use the provided context to answer the user's question.

    If the answer is present in the context,
    answer from the context.

    If the answer is not available in the context,
    you may use your general knowledge.

    Context:
    {context}

    Question:
    {input}

    Answer:
    """
)

# ============================================================
# MODERN LCEL RETRIEVAL PIPELINE
# ============================================================
# Helper function to format the retrieved documents into text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Connect components natively using the piping symbol (|)
qa_chain = (
    {
        "context": retriever | format_docs, 
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG Chatbot Ready!")
print("Type 'exit' to quit.\n")

# ============================================================
# CHAT LOOP
# ============================================================

while True:
    query = input("Ask a question: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    try:
        # Run the modern pipeline
        response_text = qa_chain.invoke(query)

        print("\nBot:")
        print(response_text)
        print()

    except Exception as e:
        print(f"\nError: {e}\n")
