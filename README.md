# 🛡️ Cybersecurity & Networking RAG Chatbot

This project is an intelligent **Retrieval-Augmented Generation (RAG)** chatbot designed to answer queries related to cybersecurity and networking. It leverages Hugging Face datasets, BGE embeddings, a FAISS vector store, and the Qwen2.5-3B-Instruct LLM.

## 🚀 Features
- **Hybrid Knowledge Base:** A combination of custom CSV files and top Hugging Face datasets (`mariiazhiv/cybersecurity_qa`, `jobby32/cybersecurity-QA-with-negatives`, and networking commands).
- **Efficient Retrieval:** Utilizes BGE small embeddings and Cross-Encoder Reranking to extract the most accurate references.
- **Local LLM Integration:** Powered by the Qwen2.5-3B-Instruct model, which fits easily on a Google Colab T4 GPU.
- **Interactive UI:** Features a Gradio chat interface that provides a shareable public link.

## 🛠️ Tech Stack
- **Python**
- **PyTorch & Transformers** (Hugging Face)
- **Sentence-Transformers & FAISS** (Vector Search)
- **Gradio** (Web UI)

## ⚙️ How to Run (Google Colab)
1. Open the repository's notebook in **Google Colab**[cite: 2].
2. Set the runtime type to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`)[cite: 2].
3. Run the cells sequentially and upload your custom dataset when prompted.
4. Interact with the chatbot once the Gradio public link is generated!
