import streamlit as st
from sentence_transformers import SentenceTransformer, util
import openai
import os
import PyPDF2
import docx
import re

# Load model from Hugging Face (will download automatically)
@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def read_file(file):
    if file.type == "application/pdf":
        return read_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx(file)
    else:
        return file.getvalue().decode("utf-8")

def chunk_text(text, chunk_size=200):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def embed_text(embedder, texts):
    return embedder.encode(texts, convert_to_tensor=True)

def semantic_search(source_embedding, target_embeddings, top_k=1):
    hits = util.semantic_search(source_embedding, target_embeddings, top_k=top_k)
    return hits[0]

def openai_quality_assessment(source_text, translation_text):
    if not openai.api_key:
        return "OpenAI API key not provided. Skipping quality assessment."
    prompt = f"""
    You are a translation quality evaluator.
    Source text (English): "{source_text}"
    Translation: "{translation_text}"
    Please provide a brief evaluation of the translation quality, highlighting any errors or issues.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content": prompt}],
        max_tokens=150,
        temperature=0.5,
    )
    return response['choices'][0]['message']['content']

def main():
    st.title("English Translation Quality Checker")

    openai_api_key = st.text_input("Enter your OpenAI API key (optional)", type="password")
    if openai_api_key:
        openai.api_key = openai_api_key

    embedder = load_model()

    source_file = st.file_uploader("Upload English source document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    if source_file:
        source_text = read_file(source_file)
        source_chunks = chunk_text(source_text)
        st.success(f"Source document loaded with {len(source_chunks)} chunks.")

        translation_files = st.file_uploader("Upload translation documents (multiple allowed)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        if translation_files:
            translations = {}
            for file in translation_files:
                text = read_file(file)
                chunks = chunk_text(text)
                translations[file.name] = chunks
            st.success(f"Loaded {len(translations)} translation documents.")

            chunk_index = st.number_input(f"Select source chunk index (0 to {len(source_chunks)-1})", min_value=0, max_value=len(source_chunks)-1, value=0)
            source_chunk = source_chunks[chunk_index]
            st.markdown(f"### Source chunk [{chunk_index}]:")
            st.write(source_chunk)

            source_emb = embed_text(embedder, [source_chunk])

            for name, chunks in translations.items():
                st.markdown(f"### Translation: {name}")
                translation_emb = embed_text(embedder, chunks)
                hits = semantic_search(source_emb, translation_emb, top_k=1)
                best_hit = hits[0]
                best_chunk = chunks[best_hit['corpus_id']]
                similarity = best_hit['score']
                st.write(f"Most similar chunk (similarity score: {similarity:.3f}):")
                st.write(best_chunk)

                if openai_api_key:
                    with st.spinner("Evaluating translation quality..."):
                        assessment = openai_quality_assessment(source_chunk, best_chunk)
                    st.markdown("**Quality Assessment:**")
                    st.write(assessment)
                else:
                    st.info("Enter OpenAI API key to get quality assessment.")

if __name__ == "__main__":
    main()
