import streamlit as st
import uuid
from PyPDF2 import PdfReader
import pdfplumber
from pathlib import Path

# ------------------ PDF Extract ------------------
def get_pdf_text(pdf_docs):
    text = ""
    page_count = 0
    for pdf in pdf_docs:
        if Path(pdf.name).suffix == ".txt":
            continue
        pdf_reader = PdfReader(pdf)
        page_count += len(pdf_reader.pages)
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
    return text, page_count


# ------------------ Streamlit App ------------------
def main():
    st.set_page_config(page_title="ChatBot with Greeting + PDF", page_icon="🤖")
    st.header("🤖 Smart ChatBot (Greetings + PDF Q&A)")

    # Keep error message in memory
    if "error_msg" not in st.session_state:
        st.session_state.error_msg = ""

    # Sidebar PDF upload
    with st.sidebar:
        st.subheader("📂 Upload PDFs")
        if "uploader_key" not in st.session_state:
            st.session_state["uploader_key"] = str(uuid.uuid4())

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            key=st.session_state["uploader_key"]
        )

        process_clicked = st.button("Process")

    # 👉 Validation
    if process_clicked and pdf_docs:
        raw_txt, page_count = get_pdf_text(pdf_docs)

        if page_count < 15:
            st.session_state.error_msg = (
                f"⚠️ Invalid Upload!\n\n"
                f"The uploaded PDFs have **{page_count} pages**.\n\n"
                f"Please upload files with at least **15 pages**."
            )
            # 🚨 Reset uploader (clear uploaded file from UI)
            st.session_state["uploader_key"] = str(uuid.uuid4())
            st.rerun()

        else:
            st.session_state.error_msg = ""
            st.success(f"✅ PDFs processed successfully! Total Pages: {page_count}")

    # Show error if exists
    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)


if __name__ == "__main__":
    main()
