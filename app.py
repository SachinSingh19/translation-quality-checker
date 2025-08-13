import streamlit as st
from sentence_transformers import SentenceTransformer, util
import openai
import PyPDF2
import docx

# Load model
@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def read_pdf_pages(file):
    reader = PyPDF2.PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        pages.append(text if text else "")
    return pages

def read_docx_pages(file):
    doc = docx.Document(file)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip() != ""]
    return ["\n".join(paragraphs)]  # Treat whole doc as one page

def read_txt_pages(file):
    text = file.getvalue().decode("utf-8")
    return [text]  # Treat whole txt as one page

def read_file_pages(file):
    if file.type == "application/pdf":
        return read_pdf_pages(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx_pages(file)
    else:
        return read_txt_pages(file)

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
    Source page text: "{source_text}"
    Translation page text: "{translation_text}"
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
    st.title("Page-Level Translation Quality Checker")

    openai_api_key = st.text_input("Enter your OpenAI API key (optional)", type="password")
    if openai_api_key:
        openai.api_key = openai_api_key

    embedder = load_model()

    source_file = st.file_uploader("Upload English source document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    if source_file:
        source_pages = read_file_pages(source_file)
        st.success(f"Source document loaded with {len(source_pages)} pages.")

        translation_files = st.file_uploader("Upload translation documents (multiple allowed)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        if translation_files:
            translations = {}
            for file in translation_files:
                pages = read_file_pages(file)
                translations[file.name] = pages
            st.success(f"Loaded {len(translations)} translation documents.")

            page_index = st.number_input(f"Select source page number (1 to {len(source_pages)})", min_value=1, max_value=len(source_pages), value=1)
            source_page_text = source_pages[page_index - 1]
            st.markdown(f"### Source page [{page_index}]:")
            st.write(source_page_text)

            source_emb = embed_text(embedder, [source_page_text])

            for name, pages in translations.items():
                st.markdown(f"### Translation: {name}")
                translation_emb = embed_text(embedder, pages)
                hits = semantic_search(source_emb, translation_emb, top_k=1)
                best_hit = hits[0]
                best_page_text = pages[best_hit['corpus_id']]
                similarity = best_hit['score']
                st.write(f"Most similar page (similarity score: {similarity:.3f}):")
                st.write(best_page_text)

                if openai_api_key:
                    with st.spinner("Evaluating translation quality..."):
                        assessment = openai_quality_assessment(source_page_text, best_page_text)
                    st.markdown("**Quality Assessment:**")
                    st.write(assessment)
                else:
                    st.info("OpenAI API key not provided. Showing similarity scores only.")

if __name__ == "__main__":
    main()
