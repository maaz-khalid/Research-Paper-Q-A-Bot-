import os
import streamlit as st
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
# from langchain_objectbox.vectorstores import ObjectBox

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Research Paper Analysis 🐼")

llm =ChatGroq(groq_api_key=groq_api_key,model="Gemma2-9b-It")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
Also give me the document name where you are referring from.
<context>
{context}
<context>
Questions: {input}
"""
)

prompt2 = (
    "Summarise the DataBase and give me the summary of research paper in the following format:"
    " 1. Name of the Research paper : "
    " 2. Name of the people involved/Authors : "
    " 3. Proposed Methodology : "
)

prompt3 = ChatPromptTemplate.from_template(
    """
You are an expert researcher.
Please provide the most accurate response based on the question.
First answer the question using context and then share your thoughts on this.
<context>
{context}
<context>
Questions: {input}
"""
)

# Vector Embedding and ObjectBox Vectorstore DB

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./research_papers")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        
        for i, doc in enumerate(st.session_state.docs):
            doc.metadata["name"] = f"Document_{i+1}"

        for doc in st.session_state.final_documents:
            doc.metadata["name"] = doc.metadata.get("name", "Unknown Document")

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

col1, col2 = st.columns(2)

with col1:
    if st.button("Read the Documents"):
        vector_embedding()
        st.write("Memory Updated")
with col2:
    if st.button("Summarise the Research Paper"):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()

        response = retrieval_chain.invoke({'input': prompt2})

        print("Response time:", time.process_time() - start)
        st.write(response['answer'])    

input_prompt = st.text_input("Enter your question on research papers")

if st.button("Answer"):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()

    response = retrieval_chain.invoke({'input': input_prompt})

    print("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(f"Document Name: {doc.metadata.get('name', 'Unknown Document')}")
            st.write(doc.page_content)
            st.write("--------------------------------")

input_prompt2 = st.text_input("Follow Up Question")

if st.button("Go"):
    document_chain = create_stuff_documents_chain(llm, prompt3)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()

    response = retrieval_chain.invoke({'input': input_prompt2})

    print("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # # With a Streamlit expander
    # with st.expander("Document Similarity Search"):
    #     # Find the relevant chunks
    #     for i, doc in enumerate(response["context"]):
    #         st.write(f"Document Name: {doc.metadata.get('name', 'Unknown Document')}")
    #         st.write(doc.page_content)
    #         st.write("--------------------------------")

if st.button("Project Information"): 
    tech_summary = """
    **Technologies Used:**
    - **Language:** Python
    - **Libraries:** Streamlit, Langchain, dotenv, PyPDF2
    - **Model Used:** ChatGroq (Gemma2-9b-It)
    - **APIs Used:** GOOGLE API, GROQ API
    """
    
    project_report = """
    **Project Report:**
    This project is designed to analyze research papers using advanced language models and vector embedding techniques. The main objective is to provide users with the ability to query a set of research papers and receive accurate, context-based responses.

    The project leverages the Langchain framework, which facilitates the integration of various language models and embeddings. Specifically, it uses the ChatGroq model (Llama3-70b-8192) to generate responses based on the context of the research papers. The OpenAI API and GROQ API are used to power the language model and embeddings, ensuring high-quality and relevant responses.

    The data ingestion process involves loading research papers from a specified directory using the PyPDFDirectoryLoader, followed by splitting the documents into manageable chunks with the RecursiveCharacterTextSplitter. These chunks are then embedded using the OpenAIEmbeddings and stored in an ObjectBox vector store, which enables efficient retrieval of relevant document sections.

    Users can interact with the system through a Streamlit interface, where they can input their queries, read documents, and get summaries of the research papers. The system processes the input, retrieves relevant document sections, and generates accurate answers to the users' questions. Additionally, the project includes functionality to summarize the database of research papers, providing a concise overview of each paper's main contributions.

    Overall, this project demonstrates the effective use of advanced language models and vector embeddings to enhance research paper analysis and information retrieval.
    """

    # st.markdown(creator_info)
    st.markdown(tech_summary)
    st.markdown(project_report)

# Add this at the end of your Streamlit app code to display the footer
st.markdown("""
<style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
    }
</style>
<div class="footer">
    <p>Designed by <a href="https://www.linkedin.com/in/maaz-khalid-73bba21b3/" target="_blank">Maaz Khalid</a></p>
</div>
""", unsafe_allow_html=True)