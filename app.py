import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import io
import uuid
import pdfplumber
from PyPDF2 import PdfReader
import fitz
from PIL import Image
import pytesseract
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from htmlTemplate import css, bot_template, user_template

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMMINI_API_KEY")

# ------------------------ Get Text from images ----------------------
def extract_text_from_image(img_input):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    if isinstance(img_input, Image.Image):
        img = img_input
    else:
        img = Image.open(img_input)
    
    text = pytesseract.image_to_string(img)
    return text.strip()

# ------------------------- Greeting with LLM -------------------------
def get_llm_greeting(user_input):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY   
    )
    try:
        response = llm.invoke(user_input)
        return response.content
    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        return "Hi there! How are you today?"

# ------------------------- PDF Utilities -------------------------
def get_pdf_text(pdf_docs):
    text = ""
    total_pages = 0
    for pdf in pdf_docs:
        if Path(pdf.name).suffix == ".txt":
            continue
        pdf_reader = PdfReader(pdf)
        total_pages += len(pdf_reader.pages)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        pdf.seek(0)
        with pdfplumber.open(pdf) as plumber_pdf:
            for page in plumber_pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        row_text = " | ".join(cell.strip() if cell else "" for cell in row)
                        text += row_text + "\n"
    return text,total_pages

# ------------------------- PDF images -------------------------
def get_pdf_images(pdf_docs, min_width=100, min_height=100):
    all_images = []
    for pdf in pdf_docs:
        pdf.seek(0)
        pdf_file = fitz.open(stream=pdf.read(), filetype="pdf")
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

# ------------------------- Convert Texts into Small Chunks  -------------------------
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(raw_text)

# ------------------------- get Vector stored in FAISS -------------------------
def get_vectorstore(text_chunks):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embedding)

# ------------------------- get conversaion chain or history -------------------------
def get_conversation_chain(vector):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.5,
        google_api_key=GOOGLE_API_KEY
    )

    custom_template = """
    You are a helpful assistant.

    User Question:
    {question}

    Document Context:
    {context}

   Guidelines:
    - If the context contains relevant info, answer directly using it.
    - If the context is empty or unrelated, still answer naturally in 1–2 short sentences.
    - Do NOT mention documents, AI, or being a language model.
    - Keep the tone friendly and professional.
    """

    prompt = PromptTemplate(
        template=custom_template,
        input_variables=["question", "context"]
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector.as_retriever(),
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": prompt}, 
        return_source_documents=False
    )

# ------------------------- All chat print or msg printing -------------------------
def msg_print(chat_placeholder):
    if st.session_state.chat_history:
        with chat_placeholder.container():
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.write(user_template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True)

# ------------------------- handle user input or verify sue quetion -------------------------
def handle_userinput(user_question, chat_placeholder):
    if st.session_state.conversation:
        response = st.session_state.conversation({"question": user_question})
        ai_answer = response.get("answer", "")

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})

    else:
        response_text = get_llm_greeting(user_question)

        st.session_state.memory.chat_memory.add_user_message(user_question)
        st.session_state.memory.chat_memory.add_ai_message(response_text)

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

    msg_print(chat_placeholder)

# ------------------------- Streamlit App -------------------------
def main():
    st.set_page_config(
    page_title="Smart ChatBot",
    page_icon="🧠")
    st.write(css, unsafe_allow_html=True)
    st.header("🤖 Smart ChatBot (Greetings + PDF Q&A)")

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello 👋, you can upload a file and ask me questions about it."}
        ]
        
    if "error_msg" not in st.session_state:
        st.session_state.error_msg = ""

    chat_placeholder = st.empty()
    msg_print(chat_placeholder)

    user_input = st.text_input("You:", key="user_input")
    if user_input:
        handle_userinput(user_input, chat_placeholder)

    with st.sidebar:
        if st.button("❌ End Chat"):
            st.session_state.clear()
            st.rerun()
        st.subheader("📂 Upload PDFs")
        if "uploader_key" not in st.session_state:
            st.session_state["uploader_key"] = str(uuid.uuid4())

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type=["pdf"],
            key=st.session_state["uploader_key"]
        )
        if st.button("Process"):
            with st.spinner("Processing PDFs..."):
                if pdf_docs:

                    for file in pdf_docs:
                        if not file.name.lower().endswith(".pdf"):
                            st.error(f"❌ Invalid file type: {file.name}. Only PDF files are allowed.")
                            st.session_state["uploader_key"] = str(uuid.uuid4())  
                            st.rerun()

                    raw_txt,page_count  = get_pdf_text(pdf_docs)
                    st.write(page_count)
                    if page_count < 15:
                        st.session_state.error_msg = (
                            f"⚠️ Invalid Upload!\n\n"
                            f"The uploaded PDFs have **{page_count} pages**.\n\n"
                            f"Please upload files with at least **15 pages**."
                        )
                        st.session_state["uploader_key"] = str(uuid.uuid4())
                        st.rerun()
                    else:
                        st.session_state.error_msg = ""
                        st.success(f"✅ PDFs processed successfully! Total Pages: {page_count}. Now ask me questions about them.")
                        st.session_state["pdf_text"] = raw_txt 

                    images = get_pdf_images(pdf_docs)
                    st.session_state["pdf_images"] = images
                    st.session_state["image_count"] = len(images) 

                    image_info_text = ""
                    if images:
                        extracted_texts = []
                        for _, img in images:
                            extracted_texts.append(extract_text_from_image(img))
                        image_info_text = "\n".join(extracted_texts)

                    full_text = raw_txt + "\n\n" + image_info_text
                    if st.session_state["image_count"] > 0:
                        full_text += f"\n\n[INFO] This PDF contains {st.session_state['image_count']} images."

                    text_chunks = get_text_chunks(full_text)
                    vector = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vector)

                    if images:
                        st.subheader("Extracted Images")
                        for name, img in images:
                            st.image(img, caption=name)
                else:
                    st.warning("Please upload at least one PDF!")
    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)


if __name__ == "__main__":
    main()
