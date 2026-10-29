
import streamlit as st
from pathlib import Path
import shutil
import rag

st.set_page_config(
    page_title="SmartDocAI",
    page_icon="📚",
    layout="centered"
)

st.title("📚 SmartDocAI")
st.write("Upload a PDF and ask questions about its content.")

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    pdf_path = UPLOAD_DIR / "sample.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"PDF uploaded: {uploaded_file.name}")

    if st.button("Process PDF"):
        try:
            with st.spinner("Reading PDF and creating embeddings..."):
                rag.index_pdf()

            st.session_state["pdf_processed"] = True
            st.session_state["chat_history"] = []
            st.success("PDF processed successfully!")

        except Exception as e:
            st.error(f"Error processing PDF: {e}")

if st.session_state.get("pdf_processed", False):
    st.subheader("💬 Ask your PDF")

    question = st.chat_input("Type your question here...")

    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question:
        st.session_state["chat_history"].append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Finding the answer..."):
                    answer, pages = rag.ask_question(question)

                st.markdown(answer)

                if pages:
                    st.caption(f"Source pages: {pages}")

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer}
            )

        except Exception as e:
            st.error(f"Error: {e}")
