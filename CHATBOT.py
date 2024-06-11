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

from chatpdf1 import user_input

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


def main():
    """Manages the Streamlit application and user interaction."""
    st.set_page_config("Chat bot that reads PDF", page_icon=":robot_face:")
    st.markdown("""
        <style>
        .reportview-container {
            background: url("D:/Downloads/pexels-stywo-1261728.jpg") center;
            background-size: cover;
        }
        .stTextInput input, .stFileUploader > div:first-child input {
            border-radius: 10px;
            border: 2px solid #008CBA; /* Blue */
            padding: 10px;
            box-shadow: 0px 0px 10px #008CBA; /* Blue */
            color: black !important; /* Text color inside input boxes */
        }
        .stButton > button, .stMultiselect > div:first-child button, .stFileUploader > div:first-child button {
            background-color: #4CAF50; /* Green */
            border: none;
            color: white;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            transition-duration: 0.4s;
            cursor: pointer;
            border-radius: 12px;
            width: 100%; /* Set width to 100% */
            padding: 10px 0; /* Add padding for better appearance */
        }
        .stButton > button:hover, .stMultiselect > div:first-child button:hover, .stFileUploader > div:first-child button:hover {
            background-color: #45a049;
        }
        .dialogue-box {
            background-color: #e0e0e0; /* Light gray */
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            box-shadow: 0px 0px 10px #888888; /* Gray */
            color: black; /* Text color inside dialogue boxes */
        }
        .highlighted-text {
            color: #FF5733; /* Orange */
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

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

        # Add a button with a specific response
        if st.button("Do you wanna know something?"):
            st.write("Code too much, drink chai 🍵")

        # Add a contact button
        if st.button("Contact me if you face any issue"):
            st.markdown("[Contact Us](mailto:suryanshpathania2003@gmail.com)")

        # Text input widget to store multiple texts
        st.subheader("Store 🚽 :")
        stored_texts = st.text_area("store anything inside me and after pressing store don't forget to empty me in the last to use delete", value="", height=100)
        if st.button("Store"):
            # Store the text
            # You can perform any additional actions here if needed
            pass

    # Display stored texts
    if stored_texts:
        st.subheader("Stored Texts:")
        texts_to_delete = []
        for idx, text in enumerate(stored_texts.split("\n")):
            # Display each stored text with a delete button
            st.write(text)
            if st.button(f"if you want to delete the content that you have stored empty the store box and press me 🍑💨"):
                # Queue this text for deletion
                texts_to_delete.append(idx)
        
        # Delete the selected texts
        updated_texts = "\n".join([text for idx, text in enumerate(stored_texts.split("\n")) if idx not in texts_to_delete])

if __name__ == "__main__":
    main()