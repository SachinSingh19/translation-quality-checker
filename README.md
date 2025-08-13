# Translation Quality Checker

This Streamlit app compares translations of French documents using semantic similarity.

## Setup

1. Download the model files from [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and place them in `models/all-MiniLM-L6-v2`.

2. Install dependencies:

```bash
pip install streamlit sentence-transformers openai PyPDF2 python-docx
