import streamlit as st
import google.generativeai as genai
from datetime import datetime
import json
import re
import requests
from io import BytesIO
from xhtml2pdf import pisa
import markdown2
import fitz # PyMuPDF
from PIL import Image

# --- NEW IMPORTS for RAG with Vector DB ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. Configuration & Initial Setup ---
st.set_page_config(page_title="AI Learning Assistant", page_icon="🎓", layout="wide")

# --- API Key Configuration ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except (KeyError, FileNotFoundError):
    st.error("API Key not found. Please create a `.streamlit/secrets.toml` file with your GEMINI_API_KEY.")
    st.stop()

# --- 2. AI & Backend Functions ---

@st.cache_resource
def get_embeddings_model():
    """Loads a sentence-transformer model for creating embeddings."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store(text_content):
    """Splits text, creates embeddings, and stores them in a FAISS vector store."""
    if not text_content:
        return None
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text_content)

        embeddings = get_embeddings_model()
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Failed to create knowledge base: {e}")
        return None

def get_relevant_context(vector_store, query, k=10):
    """Retrieves the most relevant text chunks from the vector store."""
    if vector_store is None:
        return ""
    try:
        relevant_docs = vector_store.similarity_search(query=query, k=k)
        return "\n\n".join(doc.page_content for doc in relevant_docs)
    except Exception:
        return ""  # Return empty if search fails

def generate_content(prompt, context=""):
    """Generates content using the Gemini model with optional context."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        full_prompt = (
            "You are an expert educator. Please answer the user's request based on the context provided.\n\n"
            f"Context:\n{context}\n\n"
            f"Request:\n{prompt}"
        )
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        if "429" in str(e):  # Handle rate limiting
            st.error("API quota exceeded. Please wait a moment and try again.")
        else:
            st.error(f"An error occurred with the Gemini API: {e}")
        return None

# ---------------------------
# Robust Wikipedia helpers (search -> summary -> fallback images)
# ---------------------------
def get_wikipedia_image_url(topic):
    """
    Robust function to get a Wikipedia image URL for a given topic.
    Steps:
    1) Use MediaWiki search to find the best matching page title.
    2) Try the REST summary endpoint to get the page thumbnail.
    3) If no thumbnail, list all media on the page and pick the first raster image (.jpg/.jpeg/.png).
    Returns: direct image URL or None.
    """
    headers = {"User-Agent": "AI-Learning-Assistant/1.0"}
    api_url = "https://en.wikipedia.org/w/api.php"

    # 1) Resolve best page title via search
    try:
        search_params = {"action": "query", "list": "search", "srsearch": topic, "format": "json"}
        resp = requests.get(api_url, params=search_params, headers=headers, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return None
        page_title = search_results[0]["title"]
    except Exception:
        return None

    # 2) Try REST summary to get thumbnail
    try:
        rest_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title.replace(' ', '_')}"
        resp = requests.get(rest_url, headers=headers, timeout=6)
        if resp.status_code == 200:
            summary_json = resp.json()
            thumb = summary_json.get("thumbnail", {}).get("source")
            if thumb and isinstance(thumb, str):
                if thumb.lower().endswith((".jpg", ".jpeg", ".png")):
                    return thumb
                # if thumbnail is svg/other, continue to fallback
    except Exception:
        pass

    # 3) Fallback: fetch all images listed on the page and pick first raster
    try:
        images_params = {"action": "query", "titles": page_title, "prop": "images", "format": "json", "imlimit": "max"}
        resp = requests.get(api_url, params=images_params, headers=headers, timeout=6)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        file_titles = []
        for _, pg in pages.items():
            if "images" in pg:
                for im in pg["images"]:
                    file_titles.append(im.get("title"))
        if not file_titles:
            return None

        # Filter & pick first raster image file
        valid_exts = (".jpg", ".jpeg", ".png")
        for file_title in file_titles:
            if not file_title:
                continue
            lowered = file_title.lower()
            # Skip obvious non-content logos if present
            if "commons-logo" in lowered or ("logo" in lowered and lowered.endswith(".svg")):
                continue

            info_params = {"action": "query", "titles": file_title, "prop": "imageinfo", "iiprop": "url", "format": "json"}
            try:
                info_resp = requests.get(api_url, params=info_params, headers=headers, timeout=6)
                info_resp.raise_for_status()
                info_pages = info_resp.json().get("query", {}).get("pages", {})
                for _, ipg in info_pages.items():
                    if "imageinfo" in ipg:
                        img_url = ipg["imageinfo"][0].get("url")
                        if img_url and img_url.lower().endswith(valid_exts):
                            return img_url
            except Exception:
                continue
    except Exception:
        return None

    return None

def get_wiki_content(topic):
    """Fetches a summary from Wikipedia using the REST summary endpoint (more reliable than wikipediaapi for search strings)."""
    headers = {"User-Agent": "AI-Learning-Assistant/1.0"}
    # Resolve page title
    try:
        api_url = "https://en.wikipedia.org/w/api.php"
        search_params = {"action": "query", "list": "search", "srsearch": topic, "format": "json"}
        resp = requests.get(api_url, params=search_params, headers=headers, timeout=6)
        resp.raise_for_status()
        search_results = resp.json().get("query", {}).get("search", [])
        if not search_results:
            return ""
        page_title = search_results[0]["title"]
        # Use REST summary endpoint
        rest_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title.replace(' ', '_')}"
        resp2 = requests.get(rest_url, headers=headers, timeout=6)
        if resp2.status_code == 200:
            data = resp2.json()
            return data.get("extract", "")[:2500]
    except Exception:
        return ""
    return ""

def create_pdf_from_markdown(markdown_text, title, image_url, image_caption):
    """Converts Markdown text and an image into a downloadable PDF."""
    html_content = markdown2.markdown(markdown_text, extras=["tables", "fenced-code-blocks"])
    image_html = ""
    if image_url:
        image_html = f'<div style="text-align: center;"><img src="{image_url}" style="max-width: 80%; margin: 1em auto; border-radius: 8px;"><p><em>{image_caption}</em></p></div>'
    
    css_style = """
    body { font-family: 'Helvetica', 'Arial', sans-serif; line-height: 1.6; color: #333; }
    h1, h2, h3 { color: #005a9c; border-bottom: 2px solid #f0f0f0; padding-bottom: 5px; }
    h1 { font-size: 24px; } h2 { font-size: 20px; }
    table { width: 100%; border-collapse: collapse; margin-bottom: 1em; }
    th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
    th { background-color: #f2f8ff; font-weight: bold; }
    blockquote { border-left: 4px solid #ccc; padding-left: 1em; margin-left: 0; font-style: italic; }
    code { background-color: #f5f5f5; padding: 2px 5px; border-radius: 4px; font-family: 'Courier New', monospace; }
    """
    full_html = f"<html><head><style>{css_style}</style></head><body><h1>{title}</h1>{image_html}{html_content}</body></html>"
    
    result = BytesIO()
    pdf = pisa.CreatePDF(BytesIO(full_html.encode("UTF-8")), dest=result)
    
    if not pdf.err:
        return result.getvalue()
    else:
        st.error(f"PDF creation failed: {pdf.err}")
        return None

def parse_json_response(text, expected_keys):
    """Safely extracts and validates JSON from a string response."""
    if not text: return None
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if not json_match:
        st.error("Could not find a valid JSON array in the AI's response.")
        return None
    
    try:
        data = json.loads(json_match.group(0))
        if isinstance(data, list) and all(isinstance(item, dict) and all(key in item for key in expected_keys) for item in data):
            return data
        else:
            st.error("The generated JSON does not have the expected structure.")
            return None
    except json.JSONDecodeError:
        st.error("Failed to decode JSON from the AI's response.")
        return None

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        return "".join(page.get_text() for page in doc)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# --- 3. Prompt Templates (Unchanged) ---
PRE_CLASS_PROMPT_TEMPLATE = """Generate a pre-class preparation document for the topic: "{topic}". The document must be well-structured with markdown formatting. Include these sections: 
1. **Prerequisites**: What should I know before starting?
2. **Foundational Concepts**: Explain the core ideas simply.
3. **Learning Goals**: List 3-5 clear objectives for this topic.
4. **Key Terminologies**: Present a markdown table with terms and their definitions."""

POST_CLASS_PROMPT_TEMPLATE = """Generate a {length} post-class revision document for "{topic}". Use markdown for structure. Include:
1. **Concise Summary**: A brief overview of the main points.
2. **Key Takeaways**: Bullet points of the most crucial information.
3. **Real-World Examples**: 2-3 practical applications.
4. **Mind-Map Suggestion**: A hierarchical list of concepts for a mind map."""

QUIZ_PROMPT_TEMPLATE = """Generate a quiz with 5 questions for "{topic}" at {difficulty} difficulty. The output MUST be a single, valid JSON array of objects. Each object MUST contain these keys: "question_number", "question_text", "question_type" (must be "MCQ" or "FIB"), "options" (an array of 4 strings for MCQ, empty for FIB), "answer", and "explanation". Ensure all fields have values. Do not wrap the JSON in markdown code blocks."""

FLASHCARD_PROMPT_TEMPLATE = """Generate 8-10 flashcards for the topic "{topic}". The output MUST be a single, valid JSON array of objects. Each object must contain two keys: "term" (a key concept or name) and "definition" (a concise, clear explanation). Do not wrap the JSON in markdown code blocks."""

# --- 4. State Management ---
def reset_state():
    """Resets the session state for a new topic generation."""
    keys_to_delete = [
        'pre_class_doc', 'post_class_doc', 'quiz_data', 'flashcards', 
        'quiz_submitted', 'user_answers', 'rag_context',
        'score_recorded', 'quiz_attempt_count', 'vector_store'
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.flashcard_index = 0

# --- 5. Streamlit UI ---
st.title("🎓 AI-Powered Learning Assistant")
st.markdown("Your personal guide to mastering any topic. Provide a topic and optional study materials to begin.")

if 'history' not in st.session_state: st.session_state.history = []
if 'flashcard_index' not in st.session_state: st.session_state.flashcard_index = 0

with st.sidebar:
    st.header("⚙️ Controls")
    doc_length = st.select_slider("Document Length", options=["short", "medium", "detailed"], value="medium")
    quiz_difficulty = st.select_slider("Quiz Difficulty", options=["easy", "medium", "hard"], value="medium")
    st.info("Your Google Gemini API Key is configured securely from Streamlit Secrets.")

topic = st.text_input("Enter your topic of interest", key="topic_input", placeholder="e.g., 'Neural Networks' or 'The French Revolution'")

with st.expander("📚 Add Your Knowledge Base (Optional)"):
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    user_text = st.text_area("Or paste text here", height=150)

if st.button("🚀 Generate Materials", disabled=not topic, type="primary"):
    reset_state()
    with st.spinner("Building your learning module... This may take a moment."):
        # 1. Gather all text from user sources
        full_text_context = (user_text or "")
        for file in uploaded_files:
            full_text_context += "\n\n" + extract_text_from_pdf(file)
        
        # 2. Create vector store if custom content is provided
        if full_text_context.strip():
            with st.spinner("Processing documents and building knowledge base..."):
                st.session_state.vector_store = create_vector_store(full_text_context)
                st.session_state.rag_context_source = "custom_docs"
        else:
            # 3. Fallback to Wikipedia if no custom content
            st.info("No custom knowledge base provided. Fetching context from Wikipedia as a fallback.")
            wiki_context = get_wiki_content(topic)
            if not wiki_context:
                st.warning("Could not fetch context. Generating based on general knowledge.")
                st.session_state.rag_context = ""
            else:
                st.session_state.rag_context = wiki_context
            st.session_state.rag_context_source = "wikipedia"

        st.session_state.quiz_attempt_count = 1

        # Helper to pick RAG context on demand
        def get_current_rag_context(query_topic):
            if st.session_state.get('rag_context_source') == "custom_docs":
                return get_relevant_context(st.session_state.get('vector_store'), query_topic)
            return st.session_state.get('rag_context', "")

        # Generate documents and other assets
        pre_class_context = get_current_rag_context(topic)
        st.session_state.pre_class_doc = generate_content(PRE_CLASS_PROMPT_TEMPLATE.format(topic=topic), pre_class_context)

        post_class_context = get_current_rag_context(topic)
        st.session_state.post_class_doc = generate_content(POST_CLASS_PROMPT_TEMPLATE.format(topic=topic, length=doc_length), post_class_context)

        quiz_context = get_current_rag_context(topic)
        quiz_json_str = generate_content(QUIZ_PROMPT_TEMPLATE.format(topic=topic, difficulty=quiz_difficulty), quiz_context)

        flashcard_context = get_current_rag_context(topic)
        flashcard_json_str = generate_content(FLASHCARD_PROMPT_TEMPLATE.format(topic=topic), flashcard_context)

        st.session_state.quiz_data = parse_json_response(quiz_json_str, ["question_text", "question_type", "options", "answer"])
        st.session_state.flashcards = parse_json_response(flashcard_json_str, ["term", "definition"])

        if st.session_state.pre_class_doc:
            st.success("Your personalized learning module is ready!")

# --- Display Generated Content in Tabs ---
if 'pre_class_doc' in st.session_state:
    tabs = ["📚 Pre-Class Prep", "📝 Post-Class Review", "🧠 Take the Quiz", "📇 Flashcards", "📊 Performance History"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)

    # Get Wikipedia image URL (if any)
    image_bytes = None
    image_caption = "A visual representation for your topic"
    wiki_image_url = get_wikipedia_image_url(topic)
    if wiki_image_url:
        try:
            resp = requests.get(wiki_image_url, headers={"User-Agent": "AI-Learning-Assistant/1.0"}, timeout=10)
            if resp.status_code == 200 and 'image' in resp.headers.get('Content-Type', ''):
                image_bytes = resp.content
                image_caption = f"Image from Wikipedia for: {topic}"
        except requests.RequestException:
            # silently skip image if download fails
            image_bytes = None

    with tab1:
        st.header(f"Pre-Class Preparation for: {topic}")
        if st.session_state.pre_class_doc:
            if image_bytes:
                st.image(image_bytes, caption=image_caption)
            st.markdown(st.session_state.pre_class_doc)
            pdf_bytes = create_pdf_from_markdown(st.session_state.pre_class_doc, f"Pre-Class: {topic}", None if not image_bytes else wiki_image_url, image_caption)
            if pdf_bytes:
                st.download_button("📥 Download as PDF", pdf_bytes, f"pre_class_{topic.replace(' ', '_')}.pdf", "application/pdf")
        else:
            st.warning("The pre-class document could not be generated. Please check API keys and logs.")

    with tab2:
        st.header(f"Post-Class Review for: {topic}")
        if st.session_state.post_class_doc:
            st.markdown(st.session_state.post_class_doc)
            pdf_bytes = create_pdf_from_markdown(st.session_state.post_class_doc, f"Post-Class: {topic}", None, "")
            if pdf_bytes:
                st.download_button("📥 Download as PDF", pdf_bytes, f"post_class_{topic.replace(' ', '_')}.pdf", "application/pdf")
        else:
            st.warning("The post-class document could not be generated.")

    # --- Quiz tab (unchanged logic) ---
    with tab3:
        st.header(f"Quiz on: {topic}")
        if st.session_state.get('quiz_data') and not st.session_state.get('quiz_submitted'):
            with st.form("quiz_form"):
                user_answers = {}
                for i, q in enumerate(st.session_state.quiz_data):
                    st.subheader(f"Question {i + 1}")
                    q_text, q_type = q.get('question_text'), q.get('question_type')
                    if q_type == "MCQ":
                        user_answers[i] = st.radio(q_text, options=q.get('options', []), key=f"q_{i}", index=None)
                    elif q_type == "FIB":
                        st.write(q_text.replace("____", " \_\_\_\_ "))
                        user_answers[i] = st.text_input("Your answer:", key=f"q_{i}")
                if st.form_submit_button("Submit Quiz"):
                    st.session_state.quiz_submitted = True
                    st.session_state.user_answers = user_answers
                    st.rerun()
        elif st.session_state.get('quiz_submitted'):
            st.header("Quiz Results")
            score, total = 0, len(st.session_state.quiz_data)
            for i, q in enumerate(st.session_state.quiz_data):
                user_ans = st.session_state.user_answers.get(i)
                correct_ans = q.get('answer')
                with st.expander(f"Question {i+1}: {q.get('question_text', '')}", expanded=True):
                    is_correct = user_ans and str(user_ans).strip().lower() == str(correct_ans).strip().lower()
                    if is_correct:
                        st.success(f"✔️ Your answer: **{user_ans}** (Correct!)")
                        score += 1
                    else:
                        st.error(f"❌ Your answer: **{user_ans or 'No answer'}**")
                        st.info(f"Correct answer: **{correct_ans}**")
                    st.markdown(f"**Explanation:** {q.get('explanation', 'No explanation provided.')}")
            final_score = (score / total) * 100 if total > 0 else 0
            if 'score_recorded' not in st.session_state:
                attempt = st.session_state.get('quiz_attempt_count', 1)
                topic_label = f"{topic} (Attempt {attempt})"
                st.session_state.history.append({"Topic": topic_label, "Score (%)": f"{final_score:.1f}", "Date": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.session_state.score_recorded = True
            st.subheader(f"Your Final Score: {score}/{total} ({final_score:.1f}%)")
            if final_score >= 80: st.balloons()
            if st.button("Re-attempt Quiz ↻"):
                st.session_state.quiz_attempt_count += 1
                for key in ['quiz_data', 'quiz_submitted', 'user_answers', 'score_recorded']:
                    if key in st.session_state: del st.session_state[key]
                with st.spinner("Generating new questions..."):
                    def get_current_rag_context(query_topic):
                        if st.session_state.get('rag_context_source') == "custom_docs":
                            return get_relevant_context(st.session_state.get('vector_store'), query_topic)
                        return st.session_state.get('rag_context', "")
                    quiz_context = get_current_rag_context(topic)
                    quiz_json_str = generate_content(QUIZ_PROMPT_TEMPLATE.format(topic=topic, difficulty=quiz_difficulty), quiz_context)
                    st.session_state.quiz_data = parse_json_response(quiz_json_str, ["question_text", "question_type", "options", "answer"])
                st.rerun()

    # --- Flashcards tab (unchanged) ---
    with tab4:
        st.header("📇 Flashcards")
        if st.session_state.get('flashcards'):
            card_index = st.session_state.flashcard_index
            total_cards = len(st.session_state.flashcards)
            card = st.session_state.flashcards[card_index]
            with st.container():
                st.subheader(f"Term: {card['term']}")
                with st.expander("Click to see definition"):
                    st.write(card['definition'])
            col1, col2, col3 = st.columns([1,2,1])
            if col1.button("⬅️ Previous", use_container_width=True, disabled=(card_index == 0)):
                st.session_state.flashcard_index -= 1
                st.rerun()
            col2.markdown(f"<p style='text-align: center; font-weight: bold;'>Card {card_index + 1} of {total_cards}</p>", unsafe_allow_html=True)
            if col3.button("Next ➡️", use_container_width=True, disabled=(card_index == total_cards - 1)):
                st.session_state.flashcard_index += 1
                st.rerun()
        else:
            st.warning("Flashcards could not be generated for this topic.")

    # --- History tab (unchanged) ---
    with tab5:
        st.header("Your Performance History")
        if st.session_state.history:
            st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)
        else:
            st.info("No quiz attempts recorded yet.")
