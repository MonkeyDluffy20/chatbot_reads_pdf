import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import Chroma  # Replace FAISS with Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """Splits text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Creates a Chroma vector store to store text embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_texts(
        text_chunks, embedding=embeddings, persist_directory="chromadb2"  # Persist data
    )
    return vector_store


def get_conversational_chain():
    """Creates a question-answering chain using ChatGoogleGenerativeAI."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question, vector_store):
    """Processes user input, retrieves relevant documents, and generates a response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # user_embedding = embeddings.encode_text(user_question)

    # Retrieve documents from Chroma using similarity search
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

def main():
    """Manages the Streamlit application and user interaction."""
    st.set_page_config("Chat bot that reads PDF", page_icon=":robot_face:")
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url("D:\Downloads\pexels-stywo-1261728.jpg") center;
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.header("batiyailo thoda humse")

    user_question = st.text_input("Ask a Question from the PDF Files")

    uploaded_pdfs = st.file_uploader("Upload your PDFs, press on submit button and wait", accept_multiple_files=True)

    if uploaded_pdfs:
        with st.spinner("Wait for a while and have a sip of your tea till then it will be all set to go 🚀......."):
            raw_text = get_pdf_text(uploaded_pdfs)
            text_chunks = get_text_chunks(raw_text)

            vector_store = get_vector_store(text_chunks)

        if user_question:
            user_input(user_question, vector_store)

    with st.sidebar:
        st.title("Menu:")
        # uploaded_url = st.text_input("Upload URL")
        if st.button("Add Contact Button"):
            st.markdown("[Contact Us](mailto:suryanshpathania2003@gmail.com)")

if __name__ == "__main__":
    main()
