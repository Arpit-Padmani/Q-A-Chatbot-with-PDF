import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pdfplumber
from pathlib import Path
import fitz  
import io
from PIL import Image
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import  css,bot_template,user_template
import uuid

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        if Path(pdf.name).suffix == '.txt':
            return "Not Found"
        else:
            # Extract text with PyPDF2
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
            pdf.seek(0)
            # Extract tables with pdfplumber
            with pdfplumber.open(pdf) as plumber_pdf:
                for page in plumber_pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            row_text = ' | '.join(cell.strip() if cell else '' for cell in row)
                            text += row_text + "\n"
    return text

def get_pdf_images(pdf_docs, min_width=100, min_height=100):
    all_images = []
    for pdf in pdf_docs:
        pdf.seek(0)
        pdf_file = fitz.open(stream=pdf.read(), filetype="pdf")  # Open PDF from stream
        for page_index in range(len(pdf_file)):
            page = pdf_file[page_index]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list, start=1):
                xref = img[0]
                base_image = pdf_file.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                if image.width >= min_width and image.height >= min_height:
                    all_images.append((f"{pdf.name}_page{page_index+1}_{img_index}", image))
    return all_images

def get_text_chunks(raw_text):
    text_spiltter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_spiltter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    load_dotenv()
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embedding)
    return vectorstore

def get_conversation_chain(vector):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",  # Updated model name
        temperature=0.5,
        google_api_key=""
    )
    # add duplicate key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                               retriever=vector.as_retriever(),
                                                               memory=memory)
    return conversation_chain
def msg_print(chat_placeholder):
    if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
        with chat_placeholder.container():
            for i, message in enumerate(reversed(st.session_state.chat_history)):
                if i % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        with chat_placeholder.container():
            st.info("No active chat. Upload PDF to start.")

def handle_userinput(user_question,chat_placeholder):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history=response['chat_history']
        msg_print(chat_placeholder)
    else:
        st.warning("Please upload and process a PDF first.")

def main():
    load_dotenv()
    st.set_page_config(page_title="ChatBot With PDF", page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    st.session_state.chat_ended = False

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = str(uuid.uuid4())

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history= None

    st.header("ChatBot With PDF :books:")
    user_quetion=st.text_input('Ask Question about PDF')
    chat_placeholder = st.empty()

    if user_quetion:
        handle_userinput(user_quetion, chat_placeholder)
    else:
        msg_print(chat_placeholder)


    with st.sidebar:
        col1, col2, col3 = st.columns([1, 1, 2])  # adjust weights for alignment
        with col3:
            if  st.button("🔚 End Chat"):
                st.session_state.chat_history = None
                st.session_state.conversation = None
                st.session_state.chat_ended = True
                st.session_state["uploader_key"] = str(uuid.uuid4())
                chat_placeholder.empty()
                st.toast("🔚 Chat has been successfully ended.", icon="⚠️")

        st.subheader("Your Documents")
        pdf_docs = st.file_uploader('Upload PDF Or Documents', accept_multiple_files=True,key=st.session_state["uploader_key"])

        if st.button('Process'):
            with st.spinner("Processing..."):
                if pdf_docs:
                    # Extract and show text
                    # raw_txt = get_pdf_text(pdf_docs)
                    st.subheader("PDF Text / Tables")
                    # st.text(raw_txt)
                    raw_txt=get_pdf_text(pdf_docs)
                    text_chunks=get_text_chunks(raw_txt)
                    vector=get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vector)
                    st.success("PDFs Processed")

                    # Extract and display images
                    images = get_pdf_images(pdf_docs)
                    if images:
                        st.subheader("PDF Images")
                        for name, img in images:
                            st.image(img, caption=name)
                    else:
                        st.info("No images found in the uploaded PDFs.")
                else:
                    st.warning("Please upload at least one PDF file!")

if __name__ == '__main__':
    main()
