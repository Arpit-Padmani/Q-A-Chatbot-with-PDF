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
import cv2
import pytesseract
import numpy as np
from sklearn.cluster import DBSCAN
import webcolors
from sentence_transformers import SentenceTransformer, util
from PIL import Image




def extract_text_from_image(img_input):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    if isinstance(img_input, Image.Image):
        img = img_input
    else:
        img = Image.open(img_input)
    
    text = pytesseract.image_to_string(img)
    return text.strip()

def closest_color(requested_color):
    min_distance = float("inf")
    closest_name = None
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        distance = (r_c - requested_color[0])**2 + (g_c - requested_color[1])**2 + (b_c - requested_color[2])**2
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    return closest_name

def extract_text_colors(img_input):
    if isinstance(img_input, Image.Image):
        img_byte_arr = io.BytesIO()
        img_input.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img = cv2.imdecode(np.frombuffer(img_byte_arr.read(), np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(img_input)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb_img, output_type=pytesseract.Output.DICT)

    collected_colors = []

    for i in range(len(data['text'])):
        if data['text'][i].strip() != "":
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            word_region = rgb_img[y:y+h, x:x+w]

            if word_region.size > 0:
                gray = cv2.cvtColor(word_region, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
                text_pixels = word_region[mask > 0]

                if len(text_pixels) > 0:
                    avg_color = np.mean(text_pixels, axis=0)
                    collected_colors.append(avg_color)

    if len(collected_colors) == 0:
        return []

    collected_colors = np.array(collected_colors)
    clustering = DBSCAN(eps=25, min_samples=2).fit(collected_colors)
    results = []

    for label in set(clustering.labels_):
        if label == -1:
            continue
        cluster_points = collected_colors[clustering.labels_ == label]
        avg_cluster_color = np.mean(cluster_points, axis=0).astype(int)

        hex_color = '#%02x%02x%02x' % tuple(avg_cluster_color)
        rgb = tuple(avg_cluster_color)
        color_name = closest_color(rgb)
        results.append((hex_color, rgb, color_name))

    return results

# -----------------------------
# Extract text + color info from PIL image
# -----------------------------
def extract_text_and_colors_from_pil(img, img_name="Image"):
    text = extract_text_from_image(img)
    color_info = extract_text_colors(img)

    if color_info:
        color_text = "Colors used in text: " + ", ".join([f"{name} ({hex_code})" for hex_code, rgb, name in color_info])
    else:
        color_text = "Colors used in text: None"

    combined_info = f"{img_name}:\n{text}\n{color_text}\n"
    return combined_info

# -----------------------------
# Process list of images from PDF
# -----------------------------
def extract_info_from_images(images):
    """
    images: list of tuples (name, PIL.Image.Image)
    """
    combined_info = ""
    for name, img in images:
        combined_info += extract_text_and_colors_from_pil(img, img_name=name) + "\n"
    return combined_info






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
                    images = get_pdf_images(pdf_docs) 
                    image_info_text = ""
                    if images:
                        image_info_text = extract_info_from_images(images)

                    raw_txt=get_pdf_text(pdf_docs)
                    
                    full_text = raw_txt + "\n" + image_info_text
                    text_chunks=get_text_chunks(full_text)
                    vector=get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vector)
                    st.success("PDFs Processed")

                    # Extract and display images
                    
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
