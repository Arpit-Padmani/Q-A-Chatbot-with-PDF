
# Streamlit PDF Chatbot

A Streamlit-based chatbot that allows users to upload PDF files and interact with their content using AI. This project leverages LangChain, HuggingFace embeddings, Google Generative AI (Gemini API), and FAISS for semantic search and conversational capabilities.

---

## Features

- Upload multiple PDF files.
- Extract text from PDFs using **PyPDF2** and **pdfplumber**.
- Optical Character Recognition (OCR) for scanned PDFs with **pytesseract**.
- Semantic search over PDF content using **FAISS** vector store.
- Conversational AI with context memory using **LangChain** and **Google Generative AI (Gemini API)**.
- Supports image extraction from PDFs using **PyMuPDF (fitz)** and **Pillow**.
- Interactive UI built with **Streamlit**.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Arpit-Padmani/Q-A-Chatbot-with-PDF.git
cd Q-A-Chatbot-with-PDF
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure **Tesseract OCR** is installed on your system:

* Windows: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
* macOS: `brew install tesseract`
* Linux: `sudo apt install tesseract-ocr`

---

## Usage

1. Add your **Gemini API key** to a `.env` file in the project root:

```env
GOOGLE_API_KEY=<your-gemini-api-key>
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Upload your PDF files via the UI.
4. Chat with the bot to get answers based on the PDF content.

---

## Project Structure

```
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── htmlTemplate.py        # HTML templates for chat UI
├── README.md              # Project documentation
```

---

## Dependencies

* Python 3.10+
* Streamlit
* PyPDF2, pdfplumber
* PyMuPDF (fitz), Pillow
* pytesseract
* langchain, langchain-community
* HuggingFace Transformers & Sentence-Transformers
* FAISS (vector search)
* accelerate
* OpenCV (opencv-python)
* LangChain Google Generative AI (Gemini API)

---

## Notes

* `uuid` is used internally for unique identifiers.
* Make sure your `.env` file contains your **Gemini API key** for Google Generative AI.
---

## License

MIT License © 2025 Arpit Padmani
