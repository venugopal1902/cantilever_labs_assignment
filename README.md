# 🎓 AI-Powered Learning Assistant

An interactive **Streamlit web app** that generates personalized learning modules using **Google Gemini AI**, **Wikipedia**, and **RAG (Retrieval-Augmented Generation)** with FAISS.  
This project helps learners prepare, review, quiz, and revise any topic with structured AI-powered content.

---

## 🚀 Features

- **Pre-Class Preparation**:  
  Generates structured notes with prerequisites, foundational concepts, learning goals, and key terminologies.

- **Post-Class Review**:  
  Summarized revision with key takeaways, real-world examples, and a mind-map suggestion.

- **Auto-Generated Quizzes**:  
  Multiple-choice and fill-in-the-blank questions with instant evaluation and explanations.

- **Flashcards**:  
  Quick study flashcards for important concepts.

- **Dynamic Images**:  
  Fetches relevant images from **Wikipedia** (with fallback support).

- **PDF Export**:  
  Download generated materials as a professional-looking PDF.

- **Knowledge Base Integration**:  
  Upload PDFs or add custom text to build your own knowledge base.  
  If no data is uploaded, the app automatically uses **Wikipedia** as fallback.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – Web UI  
- [Google Gemini API](https://ai.google.dev/) – AI Content Generation  
- [Wikipedia API](https://pypi.org/project/wikipedia-api/) – Context & Images  
- [LangChain + FAISS](https://python.langchain.com/) – Vector Store for RAG  
- [PyMuPDF](https://pymupdf.readthedocs.io/) – PDF Text Extraction  
- [xhtml2pdf](https://pypi.org/project/xhtml2pdf/) – PDF Generation  
- [HuggingFace Sentence Transformers](https://huggingface.co/) – Embeddings  

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone git@github.com:venugopal1902/cantilever_labs_assignment.git
cd cantilever_labs_assignment

conda create -n venu python=3.10 -y
conda activate venu

pip install -r requirements.txt
