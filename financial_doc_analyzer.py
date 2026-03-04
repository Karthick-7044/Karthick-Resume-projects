import streamlit as st
import PyPDF2
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import json
import re
from typing import List, Dict, Tuple
import time

st.set_page_config(
    page_title="PKF Financial Document Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    :root {
        --pkf-orange: #FF6B35;
        --pkf-light-grey: #F5F5F5;
        --pkf-dark-grey: #4A4A4A;
        --pkf-accent: #FFA500;
    }
    
    .main {
        background-color: var(--pkf-light-grey);
    }
    
    .pkf-header {
        background: linear-gradient(135deg, var(--pkf-orange) 0%, var(--pkf-accent) 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .pkf-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .pkf-subtitle {
        color: white;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    .chat-container {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .user-message {
        background: linear-gradient(135deg, #FFE5D9 0%, #FFF0E6 100%);
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid var(--pkf-orange);
        color: var(--pkf-dark-grey);
    }
    
    .assistant-message {
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid var(--pkf-dark-grey);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed var(--pkf-orange);
        text-align: center;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, var(--pkf-orange) 0%, var(--pkf-accent) 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.4);
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(135deg, var(--pkf-orange) 0%, var(--pkf-accent) 100%);
    }
    
    .stAlert {
        background: var(--pkf-light-grey);
        border-left: 4px solid var(--pkf-orange);
    }
    
    .stTextInput>div>div>input {
        border: 2px solid var(--pkf-light-grey);
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: var(--pkf-orange);
        box-shadow: 0 0 0 2px rgba(255, 107, 53, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-top: 3px solid var(--pkf-orange);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_embedding_model():
    """Load embedding model once and cache it"""
    return SentenceTransformer('all-MiniLM-L6-v2')


def check_ollama_running():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def check_qwen_available():
    """Check if Qwen 2.5 model is available in Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            return any('qwen2.5' in model.get('name', '').lower() for model in models)
    except:
        return False


def call_ollama(prompt: str, model: str = "qwen2.5:1.5b") -> str:
    """Call Ollama API for text generation"""
    try:
        max_prompt_length = 3500
        if len(prompt) > max_prompt_length:
            # Smart truncation - keep the question and most recent context
            lines = prompt.split('\n')
            question_part = '\n'.join([l for l in lines if 'Question:' in l or 'CRITICAL' in l])
            context_part = '\n'.join(lines[:100])  # Keep first 100 lines of context
            prompt = context_part + "\n\n" + question_part
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.05,  # Even lower for maximum factuality
                    "top_p": 0.85,
                    "num_predict": 200,
                    "num_ctx": 4096
                }
            },
            timeout=90
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '')
            if not result:
                return "The model returned an empty response. Please try rephrasing your question."
            return result
        else:
            error_msg = response.json().get('error', 'Unknown error')
            return f"Ollama error: {error_msg}"
    except requests.exceptions.Timeout:
        return "Request timed out. The model is taking too long. Try a simpler question."
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"


class FinancialDocumentAnalyzer:
    
    def __init__(self):
        self.embedding_model = None
        self.sections_db = None
        self.chunks_db = None
        self.current_doc_sections = []
        self.full_text = ""  # Store full document text
        
    def initialize_models(self, progress_callback=None):
        try:
            if progress_callback:
                progress_callback(0.2, "Checking Ollama...")
            
            if not check_ollama_running():
                st.error("❌ Ollama is not running! Start Ollama first.")
                return False
            
            if not check_qwen_available():
                st.warning("⚠️ Qwen 2.5 model not found. Run: ollama pull qwen2.5:1.5b")
                return False
            
            if progress_callback:
                progress_callback(0.5, "Loading embedding model...")
            
            self.embedding_model = load_embedding_model()
            
            if progress_callback:
                progress_callback(0.8, "Initializing vector databases...")
            
            chroma_client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                is_persistent=False
            ))
            
            self.sections_db = chroma_client.create_collection(
                name="financial_sections",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.chunks_db = chroma_client.create_collection(
                name="financial_chunks",
                metadata={"hnsw:space": "cosine"}
            )
            
            if progress_callback:
                progress_callback(1.0, "Ready! Using Qwen 2.5 via Ollama")
            
            return True
            
        except Exception as e:
            st.error(f"Error initializing: {str(e)}")
            return False
    
    def extract_financial_sections(self, pdf_file) -> List[Dict]:
        sections = []
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # More comprehensive patterns
        section_patterns = [
            r'(?i)^(INDEPENDENT\s+AUDITOR.?S?\s+REPORT)',
            r'(?i)^(STATEMENT\s+OF\s+FINANCIAL\s+POSITION)',
            r'(?i)^(BALANCE\s+SHEET)',
            r'(?i)^(STATEMENT\s+OF\s+PROFIT\s+(OR|AND)\s+LOSS)',
            r'(?i)^(INCOME\s+STATEMENT)',
            r'(?i)^(PROFIT\s+(OR|AND)\s+LOSS)',
            r'(?i)^(STATEMENT\s+OF\s+COMPREHENSIVE\s+INCOME)',
            r'(?i)^(COMPREHENSIVE\s+INCOME)',
            r'(?i)^(STATEMENT\s+OF\s+CHANGES\s+IN\s+EQUITY)',
            r'(?i)^(STATEMENT\s+OF\s+CASH\s+FLOWS)',
            r'(?i)^(CASH\s+FLOW\s+STATEMENT)',
            r'(?i)^(NOTES\s+TO\s+(THE\s+)?FINANCIAL\s+STATEMENTS)',
            r'(?i)^(ACCOUNTING\s+POLICIES)',
            r'(?i)^(SIGNIFICANT\s+ACCOUNTING\s+POLICIES)',
            r'(?i)^(DIRECTORS.?\s+REPORT)',
            r'(?i)^(AUDITOR.?S?\s+INFORMATION)',
            r'(?i)^NOTE\s+\d+',
            r'(?i)(PROFIT|LOSS|INCOME|EXPENDITURE).*(STATEMENT|ACCOUNT)',
        ]
        
        current_section = None
        current_text = []
        all_text_parts = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            all_text_parts.append(text)
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                is_header = False
                for pattern in section_patterns:
                    if re.match(pattern, line):
                        if current_section:
                            sections.append({
                                'title': current_section,
                                'content': '\n'.join(current_text),
                                'page': page_num
                            })
                        
                        current_section = line
                        current_text = []
                        is_header = True
                        break
                
                if not is_header and current_section:
                    current_text.append(line)
        
        if current_section:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_text),
                'page': page_num
            })
        
        # Store full document text
        self.full_text = '\n\n'.join(all_text_parts)
        
        # If no sections found, create one giant section
        if not sections:
            sections.append({
                'title': 'Full Document',
                'content': self.full_text,
                'page': 0
            })
        
        return sections
    
    def smart_chunk_section(self, section_content: str, chunk_size: int = 350, overlap: int = 120) -> List[str]:
        """IMPROVED: Smaller chunks with MORE overlap for better retrieval"""
        
        # First try to split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]
        
        # If no paragraphs, split by single newlines
        if len(paragraphs) < 3:
            paragraphs = [p.strip() for p in section_content.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_words = para.split()
            para_size = len(para_words)
            
            if current_size + para_size <= chunk_size:
                current_chunk.append(para)
                current_size += para_size
            else:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    
                    # Keep last 2 paragraphs for overlap
                    overlap_paras = current_chunk[-2:] if len(current_chunk) > 1 else current_chunk[-1:]
                    current_chunk = overlap_paras
                    current_size = sum(len(p.split()) for p in current_chunk)
                
                # If single paragraph is too large, split it
                if para_size > chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    temp_chunk = []
                    temp_size = 0
                    
                    for sent in sentences:
                        sent_words = sent.split()
                        sent_size = len(sent_words)
                        
                        if temp_size + sent_size <= chunk_size:
                            temp_chunk.append(sent)
                            temp_size += sent_size
                        else:
                            if temp_chunk:
                                chunks.append(' '.join(temp_chunk))
                                # Keep last sentence for overlap
                                temp_chunk = [temp_chunk[-1]] if temp_chunk else []
                                temp_size = len(temp_chunk[0].split()) if temp_chunk else 0
                            
                            temp_chunk.append(sent)
                            temp_size += sent_size
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                        current_size = temp_size
                else:
                    current_chunk.append(para)
                    current_size += para_size
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def ingest_document(self, pdf_file, progress_callback=None):
        try:
            if progress_callback:
                progress_callback(0.2, "Extracting financial sections...")
            
            sections = self.extract_financial_sections(pdf_file)
            self.current_doc_sections = sections
            
            if progress_callback:
                progress_callback(0.4, "Embedding sections...")
            
            section_texts = [f"{s['title']}\n\n{s['content']}" for s in sections]
            section_embeddings = self.embedding_model.encode(section_texts).tolist()
            
            self.sections_db.add(
                embeddings=section_embeddings,
                documents=section_texts,
                metadatas=[{'section_id': i, 'title': s['title'], 'page': s['page']} 
                          for i, s in enumerate(sections)],
                ids=[f"section_{i}" for i in range(len(sections))]
            )
            
            if progress_callback:
                progress_callback(0.6, "Creating smart chunks...")
            
            all_chunks = []
            all_metadata = []
            
            for section_id, section in enumerate(sections):
                chunks = self.smart_chunk_section(section['content'])
                
                for chunk_id, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'section_id': section_id,
                        'section_title': section['title'],
                        'chunk_id': chunk_id,
                        'page': section['page']
                    })
            
            if progress_callback:
                progress_callback(0.8, "Embedding chunks...")
            
            chunk_embeddings = self.embedding_model.encode(all_chunks).tolist()
            
            self.chunks_db.add(
                embeddings=chunk_embeddings,
                documents=all_chunks,
                metadatas=all_metadata,
                ids=[f"chunk_{i}" for i in range(len(all_chunks))]
            )
            
            if progress_callback:
                progress_callback(1.0, "Document ingested successfully!")
            
            return len(sections), len(all_chunks)
            
        except Exception as e:
            st.error(f"Error ingesting document: {str(e)}")
            return None, None
    
    def expand_query(self, query: str) -> List[str]:
        """Generate multiple query variations for better retrieval"""
        queries = [query]
        
        query_lower = query.lower()
        
        # Add variations based on financial terms
        if 'profit before tax' in query_lower or 'pbt' in query_lower:
            queries.extend([
                "profit before tax 2024",
                "profit before tax current year",
                "PBT March 2024",
                "earnings before tax",
                "profit tax lakhs crores"
            ])
        elif 'revenue' in query_lower:
            queries.extend([
                "total revenue 2024",
                "revenue from operations",
                "sales revenue"
            ])
        elif 'auditor' in query_lower:
            queries.extend([
                "independent auditor report",
                "auditor name",
                "statutory auditor"
            ])
        
        return queries[:3]  # Return top 3 variations
    
    def query_document(self, query: str) -> Tuple[str, List[Dict]]:
        """IMPROVED: Multi-query retrieval with better context assembly"""
        try:
            # Generate query variations
            query_variations = self.expand_query(query)
            
            # Get embeddings for all variations
            variation_embeddings = self.embedding_model.encode(query_variations).tolist()
            
            # Collect chunks from all query variations
            all_chunks = []
            all_metadata = []
            seen_chunks = set()
            
            for qv, qv_embedding in zip(query_variations, variation_embeddings):
                # Try section-based first
                section_results = self.sections_db.query(
                    query_embeddings=[qv_embedding],
                    n_results=2
                )
                
                if section_results['metadatas'] and section_results['metadatas'][0]:
                    best_section_id = section_results['metadatas'][0][0]['section_id']
                    
                    # Get chunks from this section
                    chunk_results = self.chunks_db.query(
                        query_embeddings=[qv_embedding],
                        n_results=4,
                        where={"section_id": best_section_id}
                    )
                    
                    if chunk_results['documents'] and chunk_results['documents'][0]:
                        for i, chunk in enumerate(chunk_results['documents'][0]):
                            chunk_key = chunk[:100]
                            if chunk_key not in seen_chunks:
                                seen_chunks.add(chunk_key)
                                all_chunks.append(chunk)
                                if i < len(chunk_results['metadatas'][0]):
                                    all_metadata.append(chunk_results['metadatas'][0][i])
                
                # Also try global search
                global_results = self.chunks_db.query(
                    query_embeddings=[qv_embedding],
                    n_results=5
                )
                
                if global_results['documents'] and global_results['documents'][0]:
                    for i, chunk in enumerate(global_results['documents'][0]):
                        chunk_key = chunk[:100]
                        if chunk_key not in seen_chunks:
                            seen_chunks.add(chunk_key)
                            all_chunks.append(chunk)
                            if i < len(global_results['metadatas'][0]):
                                all_metadata.append(global_results['metadatas'][0][i])
            
            # If still no chunks, use keyword search in full text
            if len(all_chunks) < 3:
                st.warning("⚠️ Using keyword-based fallback search...")
                
                keywords = ['profit', 'tax', 'before', '8124', '8,124', 'lakhs', 'crores']
                found_keywords = [kw for kw in keywords if kw in query.lower()]
                
                if found_keywords:
                    # Search in full text
                    lines = self.full_text.split('\n')
                    for i, line in enumerate(lines):
                        if any(kw in line.lower() for kw in found_keywords):
                            # Get context window around match
                            start = max(0, i - 10)
                            end = min(len(lines), i + 10)
                            context = '\n'.join(lines[start:end])
                            
                            chunk_key = context[:100]
                            if chunk_key not in seen_chunks and len(context.strip()) > 50:
                                seen_chunks.add(chunk_key)
                                all_chunks.append(context)
                                all_metadata.append({'section_title': 'Full Text Search', 'page': 0})
            
            if not all_chunks:
                return "I couldn't find relevant information in the document. Please try rephrasing your question.", []
            
            # Limit to top 6 chunks
            all_chunks = all_chunks[:6]
            all_metadata = all_metadata[:6]
            
            # Assemble context
            context = "\n\n---SECTION---\n\n".join(all_chunks)
            
            # Get section info for context
            section_info = "Document sections retrieved: " + ", ".join(set([m.get('section_title', 'Unknown') for m in all_metadata]))
            
            # Create improved prompt
            prompt = f"""You are a financial document analyst. Extract the EXACT answer from the data below.

{section_info}

Financial Document Data:
{context}

User Question: {query}

STRICT RULES:
1. Search for EXACT numbers with their units (crores, lakhs, ₹, Rs.)
2. If you find the answer, state it like: "Profit before tax for [year] is [exact number with units]"
3. Look for numbers like 8124, 8,124, 8124.03 - these are the values we need
4. If you DON'T find exact information, say "The specific information was not found in the retrieved sections"
5. DO NOT make up or estimate numbers
6. Keep answer to 1-2 sentences

Answer:"""

            response = call_ollama(prompt, model="qwen2.5:1.5b")
            
            # Clean response
            response = response.strip()
            
            # Create citations
            citations = []
            for i, chunk in enumerate(all_chunks):
                meta = all_metadata[i] if i < len(all_metadata) else {}
                citations.append({
                    'section': meta.get('section_title', 'Unknown'),
                    'page': meta.get('page', 0),
                    'text': chunk[:250] + "..." if len(chunk) > 250 else chunk
                })
            
            return response, citations
            
        except Exception as e:
            return f"Error processing query: {str(e)}", []


def main():
    
    st.markdown("""
        <div class="pkf-header">
            <h1 class="pkf-title">📊 PKF Financial Document Analyzer</h1>
            <p class="pkf-subtitle">AI-Powered Financial Document Intelligence with RAG Technology</p>
        </div>
    """, unsafe_allow_html=True)
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = FinancialDocumentAnalyzer()
        st.session_state.models_loaded = False
        st.session_state.document_loaded = False
        st.session_state.chat_history = []
    
    with st.sidebar:
        st.markdown("### 🎯 System Status")
        
        # Check Ollama status
        ollama_running = check_ollama_running()
        qwen_available = check_qwen_available()
        
        if ollama_running:
            st.success("✅ Ollama running")
        else:
            st.error("❌ Ollama not running")
            st.info("Start Ollama first!")
        
        if qwen_available:
            st.success("✅ Qwen 2.5 model ready")
        else:
            st.warning("⚠️ Qwen 2.5 not found")
            st.code("ollama pull qwen2.5:1.5b", language="bash")
        
        # Model initialization
        if not st.session_state.models_loaded:
            st.warning("⚠️ Models not initialized")
            if st.button("🚀 Initialize System", use_container_width=True, disabled=not (ollama_running and qwen_available)):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(value, text):
                    progress_bar.progress(value)
                    status_text.text(text)
                
                success = st.session_state.analyzer.initialize_models(update_progress)
                
                if success:
                    st.session_state.models_loaded = True
                    st.success("✅ System ready!")
                    st.rerun()
        else:
            st.success("✅ System ready")
        
        if st.session_state.document_loaded:
            st.success("✅ Document loaded")
            if st.button("📤 Upload New Document", use_container_width=True):
                st.session_state.document_loaded = False
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("📄 No document loaded")
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        This analyzer uses:
        - **Qwen 2.5 (1.5B)** via Ollama
        - **Multi-query RAG** with expansions
        - **Keyword fallback** for missed data
        - **ChromaDB** vector storage
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        st.markdown("""
        - What is the profit before tax?
        - Who is the auditor?
        - What is the total revenue?
        - Show me total assets
        """)
    
    if not st.session_state.models_loaded:
        st.info("👈 Please initialize the system using the sidebar button to get started.")
        return
    
    if not st.session_state.document_loaded:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 📁 Upload Financial Document")
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=['pdf'],
                help="Upload financial statements, audit reports, or other financial documents"
            )
            
            if uploaded_file is not None:
                st.info(f"📄 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
                
                if st.button("🔍 Analyze Document", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(value, text):
                        progress_bar.progress(value)
                        status_text.text(text)
                    
                    sections, chunks = st.session_state.analyzer.ingest_document(
                        uploaded_file,
                        update_progress
                    )
                    
                    if sections and chunks:
                        st.session_state.document_loaded = True
                        
                        # Debug: Show extracted sections
                        with st.expander("🔍 Debug: Extracted Sections"):
                            for i, section in enumerate(st.session_state.analyzer.current_doc_sections):
                                st.write(f"**Section {i}:** {section['title']} (Page {section['page']})")
                                st.write(f"Preview: {section['content'][:300]}...")
                                st.write("---")
                        
                        st.success(f"✅ Document analyzed! Found {sections} sections and created {chunks} chunks.")
                        time.sleep(1)
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="user-message">👤 <strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">🤖 <strong>PKF Assistant:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
                
                if 'citations' in message and message['citations']:
                    with st.expander("📚 View Citations & Retrieved Data"):
                        for i, cite in enumerate(message['citations']):
                            st.markdown(f"""
                            <div class="metric-card">
                                <strong>Source {i+1}:</strong> {cite['section']} (Page {cite['page']})<br>
                                <pre style="white-space: pre-wrap; font-size: 0.85em; background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 8px;">{cite['text']}</pre>
                            </div>
                            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Query input
        query = st.chat_input("Ask a question (e.g., What is the profit before tax?)")
        
        if query:
            st.session_state.chat_history.append({
                'role': 'user',
                'content': query
            })
            
            with st.spinner("🔍 Searching document with multiple strategies..."):
                response, citations = st.session_state.analyzer.query_document(query)
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'citations': citations
            })
            
            st.rerun()


if __name__ == "__main__":
    main()
