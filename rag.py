
from pathlib import Path

import chromadb
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# Project paths
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "uploads" / "sample.pdf"
DB_PATH = str(BASE_DIR / "chroma_db")

# Ollama models
CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

COLLECTION_NAME = "smartdoc"


def read_pdf(pdf_path=PDF_PATH):
    """Read text from every page of a PDF."""
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Upload a PDF using the SmartDocAI interface."
        )

    reader = PdfReader(str(pdf_path))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append({
                "text": text,
                "page": page_number
            })

    if not documents:
        raise ValueError(
            "No readable text found. The PDF may be scanned or image-only."
        )

    return documents


def create_chunks(documents):
    """Split PDF page text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = []

    for document in documents:
        page_chunks = splitter.split_text(document["text"])

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page": document["page"]
            })

    return chunks


def get_embedding(text):
    """Generate an embedding using Ollama."""
    response = ollama.embed(
        model=EMBED_MODEL,
        input=text
    )

    return response["embeddings"][0]


def index_pdf():
    """Read the PDF, create embeddings and save them to ChromaDB."""
    print("Reading PDF...")

    documents = read_pdf()
    print(f"Read {len(documents)} pages.")

    chunks = create_chunks(documents)
    print(f"Created {len(chunks)} text chunks.")

    client = chromadb.PersistentClient(path=DB_PATH)

    # Rebuild the collection for the currently uploaded PDF.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    print("Creating embeddings and saving to ChromaDB...")

    # Process batches to avoid keeping every embedding in memory.
    batch_size = 20

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        embeddings = [
            get_embedding(chunk["text"])
            for chunk in batch
        ]

        ids = [
            f"chunk_{start + index}"
            for index in range(len(batch))
        ]

        collection.add(
            ids=ids,
            documents=[chunk["text"] for chunk in batch],
            embeddings=embeddings,
            metadatas=[
                {"page": chunk["page"]}
                for chunk in batch
            ]
        )

    print("PDF indexing completed successfully!")

    return len(chunks)


def retrieve_context(question, top_k=3):
    """Retrieve the most relevant PDF chunks for a question."""
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_collection(name=COLLECTION_NAME)

    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    found_documents = results.get("documents", [[]])[0]
    found_metadatas = results.get("metadatas", [[]])[0]

    context_parts = []
    pages = []

    for document, metadata in zip(found_documents, found_metadatas):
        page_number = metadata.get("page", "?")

        context_parts.append(
            f"[Page {page_number}]\n{document}"
        )

        if page_number not in pages:
            pages.append(page_number)

    context = "\n\n".join(context_parts)

    return context, pages


def generate_answer(question, context):
    """Generate an answer grounded in the retrieved PDF context."""
    if not context.strip():
        return "I couldn't find relevant information in the uploaded PDF."

    prompt = f"""
You are SmartDocAI, a PDF question-answering assistant.

Rules:
1. Answer using only the PDF context provided below.
2. Do not invent facts or use outside knowledge.
3. If the context does not contain the answer, clearly say:
   "I couldn't find the answer in the uploaded PDF."
4. Explain the answer clearly and simply.
5. Reply in the same language as the user's question when possible.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()


def ask_question(question):
    """Return the answer and source pages for the Streamlit interface."""
    context, pages = retrieve_context(question)
    answer = generate_answer(question, context)

    return answer, pages


def main():
    """Run the terminal-based chatbot."""
    print("=" * 50)
    print("SmartDocAI - PDF Question Answering")
    print("=" * 50)

    try:
        index_pdf()

        while True:
            question = input("\nYour question (or type 'exit'): ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Thank you for using SmartDocAI!")
                break

            if not question:
                print("Please enter a question.")
                continue

            answer, pages = ask_question(question)

            print("\nAnswer:")
            print(answer)

            if pages:
                print(f"\nSource pages: {pages}")

    except FileNotFoundError as error:
        print(f"\nFILE ERROR:\n{error}")

    except ollama.ResponseError as error:
        print(f"\nOLLAMA ERROR:\n{error}")
        print(
            "Check that Ollama is running and that the required "
            "models are downloaded."
        )

    except Exception as error:
        print(f"\nERROR: {error}")


if __name__ == "__main__":
    main()
