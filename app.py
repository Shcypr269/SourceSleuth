"""
SourceSleuth Web UI — Streamlit Frontend.

A premium, modern web interface for searching orphaned quotes across
your academic PDFs using local semantic search.

Deployment: Run with `streamlit run app.py --server.port 8501`
Health Check: The app responds to GET requests at /healthz
"""

import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

from src.config import PDF_DIR, DATA_DIR, EMBEDDING_MODEL, TOP_K, MIN_SCORE, SEARCH_MODE

# Lazy imports - import heavy ML libraries only when needed
# This avoids loading PyTorch/SentenceTransformers on every Streamlit rerun

# --- Page Configuration ---
st.set_page_config(
    page_title="SourceSleuth | Citation Recovery",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Query Expansion Helper (Domain-Agnostic, POS-Tagged WordNet)
# ---------------------------------------------------------------------------

def _get_wordnet_pos(treebank_tag: str) -> str | None:
    """
    Map Penn Treebank POS tags to WordNet POS tags.
    
    Args:
        treebank_tag: POS tag from nltk.pos_tag (e.g., 'NN', 'VB', 'JJ')
        
    Returns:
        WordNet POS constant or None if not a content word.
    """
    if treebank_tag.startswith('NN'):
        return 'n'  # Noun
    elif treebank_tag.startswith('VB'):
        return 'v'  # Verb
    elif treebank_tag.startswith('JJ'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('RB'):
        return 'r'  # Adverb
    return None


def expand_query_keywords(query: str, max_synonyms: int = 2) -> list[str]:
    """
    Expand a query with domain-agnostic synonyms using POS-tagged WordNet.
    
    This uses NLTK's POS tagger to identify nouns and verbs, then retrieves
    synonyms ONLY from the matching WordNet POS category. This prevents
    cross-POS semantic noise (e.g., expanding noun "force" with verb synonyms
    like "coerce", "ram").
    
    Returns a list of DISTINCT query variations for separate embedding,
    NOT a concatenated bloated string.
    
    Args:
        query: The original search query.
        max_synonyms: Maximum synonyms per content word.
        
    Returns:
        List of distinct query strings to search:
        - [0]: Original query
        - [1:]: Synonym-substituted variations
    """
    # Lazy import NLTK only when expansion is needed
    try:
        from nltk.corpus import wordnet as wn
        from nltk import pos_tag, word_tokenize
        import nltk
        
        # Download NLTK data on first run (cached locally)
        try:
            wn.ensure_loaded()
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('punkt', quiet=True)
            
    except ImportError:
        # NLTK not installed — return original query only
        return [query]
    
    # Tokenize and POS tag
    try:
        tokens = word_tokenize(query)
        tagged = pos_tag(tokens)
    except Exception:
        # Fallback if tokenization fails
        return [query]
    
    # Extract content words (nouns, verbs, adjectives) with POS and INDEX
    # Store index to enable precise token replacement
    content_words = []
    for idx, (word, tag) in enumerate(tagged):
        pos = _get_wordnet_pos(tag)
        # Only process content words: has POS tag, length > 2, is alphabetic
        if pos and len(word) > 2 and word.isalpha():
            content_words.append((idx, word.lower(), pos))
    
    if not content_words:
        return [query]
    
    # Build distinct query variations
    variations = [query]  # Always include original
    
    # Generate synonym-substituted queries
    # For each content word, create ONE variation with its primary synonym
    for token_idx, word, pos in content_words[:4]:  # Limit to first 4 for speed
        # Get synsets for this specific POS
        synsets = wn.synsets(word, pos=pos)
        if not synsets:
            continue
        
        # FIXED Issue #2: Get primary synonym directly (deterministic)
        # synsets[0] = most common sense, lemmas()[0] = most common lemma
        primary_lemma = synsets[0].lemmas()[0]
        primary_synonym = primary_lemma.name()
        
        # FIXED Issue #3: Sanitize underscores from multi-word lemmas
        primary_synonym = primary_synonym.replace('_', ' ')
        
        # Skip if synonym is same as original word
        if primary_synonym.lower() == word:
            continue
        
        # FIXED Issue #1: Token-level replacement (not substring replacement)
        # Create new token list with synonym at exact index
        new_tokens = tokens.copy()
        new_tokens[token_idx] = primary_synonym
        
        # Reconstruct query from tokens
        variation = ' '.join(new_tokens)
        
        # Clean up punctuation spacing (word_tokenize separates punctuation)
        variation = variation.replace(' .', '.').replace(' ,', ',').replace(' ?', '?')
        
        if variation != query:
            variations.append(variation)
    
    return variations


def expand_query_simple(query: str) -> list[str]:
    """
    Fallback query expansion when NLTK/WordNet is unavailable.
    
    Returns distinct query variations with generic academic context.
    
    Args:
        query: The original search query.
        
    Returns:
        List of distinct query strings.
    """
    return [
        query,
        query + " academic paper scholarly source",
        query + " research study citation reference",
    ]

# --- Premium CSS Styling ---
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    * { font-family: 'Inter', sans-serif; }

    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0a0a 40%, #2a1a0a 100%);
        color: #f0f0f0;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0a0a 0%, #0a0a0a 100%);
        border-right: 1px solid rgba(255, 50, 50, 0.15);
    }

    [data-testid="stSidebar"] * {
        color: #f5d5c5 !important;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Hero title */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff3333 0%, #ff6600 50%, #ffcc00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #c5a595;
        font-weight: 400;
        margin-bottom: 2rem;
        letter-spacing: 0.01em;
    }

    /* Stat cards */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }

    .stat-card {
        background: rgba(255, 50, 50, 0.05);
        border: 1px solid rgba(255, 50, 50, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .stat-card:hover {
        border-color: rgba(255, 100, 50, 0.5);
        background: rgba(255, 50, 50, 0.1);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 50, 50, 0.15);
    }

    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ff3333, #ffcc00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }

    .stat-label {
        font-size: 0.8rem;
        color: #a58575;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.4rem;
    }

    /* Citation result card */
    .result-card {
        background: rgba(255, 50, 50, 0.05);
        border: 1px solid rgba(255, 50, 50, 0.2);
        border-left: 4px solid;
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .result-card:hover {
        border-color: rgba(255, 100, 50, 0.4);
        background: rgba(255, 50, 50, 0.08);
        box-shadow: 0 8px 30px rgba(255, 50, 50, 0.1);
    }

    .result-card.high { border-left-color: #ff3333; }
    .result-card.medium { border-left-color: #ff9900; }
    .result-card.low { border-left-color: #ffcc00; }

    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
    }

    .result-filename {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f5d5c5;
    }

    .result-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-high { background: rgba(255, 50, 50, 0.15); color: #ff3333; }
    .badge-medium { background: rgba(255, 150, 50, 0.15); color: #ff9900; }
    .badge-low { background: rgba(255, 200, 50, 0.15); color: #ffcc00; }

    .result-context {
        font-size: 0.92rem;
        color: #d5b5a5;
        line-height: 1.7;
        font-style: italic;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 3px solid rgba(255, 100, 50, 0.3);
    }

    .result-meta {
        display: flex;
        gap: 1.5rem;
        font-size: 0.82rem;
        color: #a58575;
        margin-top: 0.6rem;
    }

    .result-meta span {
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }

    /* Search input area */
    .stTextArea textarea {
        background: rgba(255, 50, 50, 0.05) !important;
        border: 1px solid rgba(255, 100, 50, 0.3) !important;
        border-radius: 12px !important;
        color: #f0f0f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
        transition: border-color 0.3s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: rgba(255, 150, 50, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(255, 100, 50, 0.1) !important;
    }

    .stTextArea textarea::placeholder {
        color: #6a5a50 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #ff3333 0%, #ff6600 100%) !important;
        color: #0a0a0a !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.8rem 2rem !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 50, 50, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 100, 50, 0.4) !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f5d5c5;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 100, 50, 0.2);
    }

    /* Examples styling */
    .example-card {
        background: rgba(255, 100, 50, 0.05);
        border: 1px dashed rgba(255, 100, 50, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .example-item {
        color: #c5a595;
        font-style: italic;
        padding: 0.5rem 0;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #6a5a50;
        font-size: 0.8rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(255, 100, 50, 0.1);
        margin-top: 3rem;
    }

    .app-footer a {
        color: #ff6600;
        text-decoration: none;
    }

    .app-footer a:hover {
        text-decoration: underline;
    }

    /* Sidebar specific */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 50, 50, 0.1) !important;
        border: 1px solid rgba(255, 100, 50, 0.3) !important;
        box-shadow: none !important;
        font-size: 0.85rem !important;
        padding: 0.6rem 1rem !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 50, 50, 0.2) !important;
        border-color: rgba(255, 150, 50, 0.4) !important;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        margin-top: 0.5rem;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border-color: rgba(255, 100, 50, 0.2) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 50, 50, 0.05) !important;
        border-radius: 8px !important;
    }

    /* Dividers */
    hr {
        border-color: rgba(255, 100, 50, 0.15) !important;
    }

    /* Success/Warning/Error messages */
    .stAlert {
        border-radius: 12px !important;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Animated glow on search */
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(255, 100, 50, 0); }
        50% { box-shadow: 0 0 20px 5px rgba(255, 100, 50, 0.15); }
    }

    .search-active {
        animation: pulse-glow 2s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
@st.cache_resource
def get_vector_store():
    """Load or initialize the vector store."""
    # Lazy import to avoid loading PyTorch on every Streamlit rerun
    from src.vector_store import VectorStore
    
    store = VectorStore(model_name=EMBEDDING_MODEL, data_dir=DATA_DIR)
    store.load()
    return store


def format_confidence(score):
    """Format confidence score with tier."""
    if score >= 0.75:
        return "high", f"High ({score:.3f})"
    elif score >= 0.50:
        return "medium", f"Medium ({score:.3f})"
    else:
        return "low", f"Low ({score:.3f})"


# --- Sidebar ---
with st.sidebar:
    st.markdown("### Settings")
    st.caption("Configure search parameters")

    st.divider()

    # Search settings
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=TOP_K)
    
    # FIXED: Lower default threshold from 0.65 to 0.35
    # all-MiniLM-L6-v2 produces compressed cosine scores
    # Relevant paraphrases often score 0.45-0.55
    min_score = st.slider(
        "Minimum similarity",
        min_value=0.0, max_value=1.0, value=0.35, step=0.05,
        help="Lower threshold (0.35) captures paraphrased concepts. "
             "Dense embeddings compress scores - relevant matches often score 0.4-0.6."
    )
    
    # NEW: Search mode toggle for hybrid/dense/sparse
    search_mode = st.selectbox(
        "Search mode",
        options=["hybrid", "dense", "sparse"],
        index=0 if SEARCH_MODE == "hybrid" else (1 if SEARCH_MODE == "dense" else 2),
        help="Hybrid (default): Combines semantic + keyword matching. "
             "Dense: Semantic only. Sparse: Keyword (BM25) only."
    )

    st.divider()

    # OCR Settings
    st.markdown("### OCR Settings")
    st.caption("For scanned documents")
    
    enable_ocr = st.checkbox(
        "Enable OCR for scanned PDFs",
        value=False,
        help="Enable OCR (Optical Character Recognition) for scanned documents. "
             "Slower processing but extracts text from image-only PDFs."
    )
    
    ocr_language = st.selectbox(
        "OCR Language",
        options=["eng", "fra", "deu", "spa", "ita", "por", "rus"],
        index=0,
        help="OCR language code. Note: Only English ('eng') is installed by default. "
             "Other languages require manual installation of Tesseract language packs."
    )
    
    st.divider()

    # File upload section
    st.markdown("### Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload academic PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF files to add to the search index",
    )

    # Initialize session state for tracking processed files
    # Use composite key: filename + ocr settings to allow re-processing with different params
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}  # Dict: {composite_key: True}

    if uploaded_files:
        # Build composite keys for already processed files
        def get_composite_key(filename: str) -> str:
            return f"{filename}_ocr{enable_ocr}_lang{ocr_language}"
        
        # Filter to only files that haven't been processed with CURRENT settings
        new_files = [
            f for f in uploaded_files 
            if get_composite_key(f.name) not in st.session_state.processed_files
        ]
        
        if new_files and st.button("Process Uploaded PDFs", width="stretch"):
            with st.spinner("Processing PDFs ..."):
                temp_dir = tempfile.mkdtemp()
                saved_paths = []

                for uploaded_file in new_files:
                    save_path = Path(temp_dir) / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_paths.append(save_path)

                # Use centralized PDF processor with OCR support (DRY principle)
                try:
                    from src.pdf_processor import process_pdf_directory
                    
                    # Process PDFs using centralized logic (includes OCR fallback)
                    chunks = process_pdf_directory(
                        directory=Path(temp_dir),
                        use_ocr=enable_ocr,
                        ocr_language=ocr_language,
                    )
                    
                    if chunks:
                        # Add to vector store and persist
                        store = get_vector_store()
                        added = store.add_chunks(chunks)
                        store.save()
                        
                        # Track processed files with composite key to allow re-processing
                        for f in new_files:
                            key = get_composite_key(f.name)
                            st.session_state.processed_files[key] = True
                        
                        st.success(f"Added {added} chunks from {len(new_files)} PDF(s)")
                    else:
                        st.warning("No text could be extracted from uploaded PDFs.")
                        
                except Exception as e:
                    st.error(f"Failed to process PDFs: {e}")
        
        elif not new_files:
            st.info(f"All {len(uploaded_files)} uploaded file(s) have already been processed with current settings.")

    st.divider()

    # Maintenance section
    st.markdown("### Maintenance")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh", width="stretch"):
            st.cache_resource.clear()
            st.rerun()

    with col2:
        if st.button("Clear Index", width="stretch"):
            store = get_vector_store()
            store.clear()
            index_path = DATA_DIR / "sourcesleuth.index"
            meta_path = DATA_DIR / "sourcesleuth_metadata.json"
            if index_path.exists():
                index_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
            st.cache_resource.clear()
            st.success("Index cleared!")
            st.rerun()

    st.divider()

    st.markdown("### About")
    st.markdown("""
    **SourceSleuth** recovers citations for orphaned quotes using local semantic search.

    - All data stays on your machine
    - No API keys required
    - Powered by FAISS + SentenceTransformers
    - Licensed under Apache 2.0
    """)

    st.caption("v1.0.0 | Apache 2.0")


# --- Main UI ---
st.markdown('<div class="hero-title">SourceSleuth</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Recover citations for orphaned quotes using local semantic search — powered by MCP'
    '</div>',
    unsafe_allow_html=True,
)

# Load vector store and get stats
store = get_vector_store()
stats = store.get_stats()

# Display stats
st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-value">{stats['total_chunks']:,}</div>
        <div class="stat-label">Total Chunks</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{stats['num_files']}</div>
        <div class="stat-label">Documents</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{stats['embedding_dim']}</div>
        <div class="stat-label">Dimensions</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">CPU</div>
        <div class="stat-label">Runtime</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Input Area
st.markdown(
    '<div class="section-header">Search for Orphaned Quote</div>',
    unsafe_allow_html=True,
)

query = st.text_area(
    "Paste the quote or paraphrase you want to find the source for:",
    placeholder=(
        "e.g., 'The attention mechanism allows models to focus on "
        "specific parts of the input sequence...'"
    ),
    height=120,
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    search_clicked = st.button("Find Sources", width="stretch")

# --- Results Section ---
if search_clicked and query:
    st.divider()
    st.markdown(
        '<div class="section-header">Search Results</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Searching your documents ..."):
        # FIXED: Wire search_mode parameter to enable hybrid search
        # FIXED: Try query expansion with POS-tagged WordNet synonyms
        results = store.search(query=query, top_k=top_k, mode=search_mode)
        
        # Track seen chunk indices to deduplicate results
        seen_chunk_indices = set()
        if results:
            for r in results:
                seen_chunk_indices.add((r.get('filename'), r.get('page'), r.get('start_char')))
        
        # If no results or few results, try expanded queries
        if len(results) < 3 and search_mode in ["hybrid", "dense"]:
            st.info("Trying query expansion with synonyms...")
            
            # Get POS-tagged WordNet expansions (distinct query variations)
            expansions = expand_query_keywords(query)
            
            # Fallback to simple expansion if WordNet returned only original
            if len(expansions) <= 1:
                expansions = expand_query_simple(query)
            
            # Search each variation separately and deduplicate
            for expanded_query in expansions[1:]:  # Skip original (already searched)
                if expanded_query == query:
                    continue
                    
                expanded_results = store.search(query=expanded_query, top_k=top_k, mode=search_mode)
                
                # Deduplicate: only add results we haven't seen
                new_results = []
                for r in expanded_results:
                    key = (r.get('filename'), r.get('page'), r.get('start_char'))
                    if key not in seen_chunk_indices:
                        new_results.append(r)
                        seen_chunk_indices.add(key)
                
                if new_results:
                    st.success(f"Found {len(new_results)} additional matches with expanded query!")
                    results.extend(new_results)
                    
                    # Stop after finding new results
                    break
        
        # Sort all results by score descending
        if results:
            results = sorted(results, key=lambda x: -x.get('score', 0))

    if not results:
        st.warning(
            "No matching sources found. Try uploading more PDFs "
            "or adjusting your search query."
        )
    else:
        filtered_results = [r for r in results if r["score"] >= min_score]

        if not filtered_results:
            st.warning(
                f"No results above the minimum score threshold ({min_score}). "
                "Try lowering the threshold in the sidebar."
            )
        else:
            st.success(f"Found {len(filtered_results)} potential match(es)!")

            for i, result in enumerate(filtered_results, start=1):
                tier, badge_text = format_confidence(result["score"])
                context_preview = result["text"][:400].replace("\n", " ")
                if len(result["text"]) > 400:
                    context_preview += " ..."

                st.markdown(f"""
                <div class="result-card {tier}">
                    <div class="result-header">
                        <span class="result-filename">
                            #{i} &mdash; {result['filename']}
                        </span>
                        <span class="result-badge badge-{tier}">{badge_text}</span>
                    </div>
                    <div class="result-context">
                        "{context_preview}"
                    </div>
                    <div class="result-meta">
                        <span>Page {result['page']}</span>
                        <span>Chunk #{result.get('chunk_index', 'N/A')}</span>
                        <span>Chars {result.get('start_char', '?')}&ndash;{result.get('end_char', '?')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"View full context for result #{i}"):
                    st.text(result["text"])

elif search_clicked and not query:
    st.warning("Please enter a quote to search for.")

else:
    # Example queries
    st.markdown("""
    <div class="example-card">
        <div class="section-header" style="border: none; margin-bottom: 0.5rem;">
            Example Queries
        </div>
        <div class="example-item">
            "Attention is all you need for sequence transduction"
        </div>
        <div class="example-item">
            "Wave interference produces a pattern of bright and dark fringes"
        </div>
        <div class="example-item">
            "The photoelectric effect demonstrates the particle nature of light"
        </div>
        <div class="example-item">
            "Transformers have replaced recurrent models in most NLP tasks"
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Document List ---
if stats["num_files"] > 0:
    st.divider()
    st.markdown(
        '<div class="section-header">Indexed Documents</div>',
        unsafe_allow_html=True,
    )

    df = pd.DataFrame({
        "Filename": stats["ingested_files"],
        "Status": ["Indexed"] * len(stats["ingested_files"]),
    })
    st.dataframe(df, width="stretch", hide_index=True)

# --- Footer ---
st.markdown("""
<div class="app-footer">
    Powered by <strong>Model Context Protocol (MCP)</strong>
    &nbsp;&bull;&nbsp;
    <a href="https://github.com/Ishwarpatra/OpenSourceSleuth" target="_blank">GitHub</a>
    &nbsp;&bull;&nbsp;
    Apache 2.0
</div>
""", unsafe_allow_html=True)
