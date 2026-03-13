# SourceSleuth

> Recover citations for orphaned quotes using local semantic search, powered by MCP.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-purple.svg)](https://modelcontextprotocol.io)
[![CI Pipeline](https://github.com/Ishwarpatra/OpenSourceSleuth/actions/workflows/ci.yml/badge.svg)](https://github.com/Ishwarpatra/OpenSourceSleuth/actions/workflows/ci.yml)
[![CodeRabbit Reviews](https://img.shields.io/coderabbit/prs/github/Ishwarpatra/OpenSourceSleuth?utm_source=oss&utm_medium=github&utm_campaign=Ishwarpatra%2FOpenSourceSleuth&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)](https://coderabbit.ai)

---

## The Problem

Millions of students experience a specific panic every semester: they find a brilliantly paraphrased concept or exact quote buried in their draft, but have **completely lost the original source and page number**. This leads to wasted hours re-reading materials — or worse, risking academic integrity violations by guessing citations.

**SourceSleuth** solves this by running a local [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that semantically searches your academic PDFs. You can connect it to AI assistants like Claude Desktop, Cursor, and Windsurf, or use the dedicated Web UI to identify exactly where a quote originated.

Everything runs **locally on your machine** — no data leaves your hardware and no external API keys are required.

---

## Features

| Capability | Type | Description |
|---|---|---|
| `find_orphaned_quote` | Tool | Semantic search across all indexed PDFs for a quote or paraphrase |
| `ingest_pdfs` | Tool | Batch-ingest a folder of PDFs into the local vector store |
| `ingest_arxiv` | Tool | Preprocess and ingest arXiv paper abstracts for citation recovery |
| `get_store_stats` | Tool | View statistics about indexed documents and total chunks |
| `sourcesleuth://pdfs/{filename}` | Resource | Access and read the full text of any indexed PDF |
| `cite_recovered_source` | Prompt | Format a recovered source into a proper APA/MLA/Chicago citation |
| **Web UI** | Interface | Interactive Streamlit browser-based search interface |
| **CLI** | Interface | Standalone command-line tool for ingestion and management |

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Core Language** | Python 3.10+ | Main implementation language |
| **AI/ML Framework** | SentenceTransformers | Semantic text embeddings |
| **Embedding Model** | all-MiniLM-L6-v2 | 384-dim sentence embeddings (CPU-optimized) |
| **Vector Search** | FAISS (Facebook AI Similarity Search) | Efficient similarity search and clustering |
| **PDF Processing** | PyMuPDF (fitz) | High-performance PDF text extraction |
| **AI Integration** | Model Context Protocol (MCP) | Standard protocol for AI assistant integration |
| **Web Framework** | Streamlit | Browser-based user interface |
| **Data Processing** | NumPy, scikit-learn | Numerical operations and preprocessing |
| **Hybrid Search** | rank-bm25 | Optional keyword-based search |
| **OCR** | Tesseract, pdf2image | Scanned PDF text extraction |
| **Development** | pytest, ruff | Testing and linting |
| **Packaging** | setuptools, pip | Distribution and dependency management |

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interfaces                               │
├─────────────────────┬─────────────────────┬─────────────────────┤
│   Claude Desktop    │      Web UI         │       CLI           │
│   Cursor / MCP Host │   (Streamlit)       │  (standalone)       │
└─────────┬───────────┴──────────┬──────────┴──────────┬──────────┘
          │                      │                     │
          └──────────────────────┼─────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   SourceSleuth MCP      │
                    │        Server           │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
     ┌────────▼────────┐  ┌──────▼───────┐  ┌──────▼───────┐
     │  PDF Processor  │  │ Vector Store │  │   Embedding  │
     │   (PyMuPDF)     │  │   (FAISS)    │  │   Model      │
     └────────┬────────┘  └──────┬───────┘  └──────┬───────┘
              │                  │                  │
     ┌────────▼────────┐  ┌──────▼───────┐  ┌──────▼───────┐
     │ student_pdfs/   │  │   data/      │  │ sentence-    │
     │ (your PDFs)     │  │ (index +     │  │ transformers │
     │                 │  │  metadata)   │  │              │
     └─────────────────┘  └──────────────┘  └──────────────┘
```

---

## Architecture

```
MCP Host (Claude Desktop / Cursor / Windsurf)
└── MCP Client  ──stdio──>  SourceSleuth MCP Server
                                    |
                    ┌───────────────┼───────────────┐
                    |               |               |
              PDF Processor    Vector Store    SentenceTransformer
               (PyMuPDF)        (FAISS)       (all-MiniLM-L6-v2)
                    |               |
              student_pdfs/       data/
             (your papers)   (persisted index)
```

| Module | Responsibility |
|---|---|
| `src/mcp_server.py` | FastMCP server exposing tools, resources, and prompts |
| `src/source_sleuth.py` | Standalone `SourceRetriever` class (no MCP dependency) |
| `src/pdf_processor.py` | PDF text extraction and chunking |
| `src/vector_store.py` | FAISS index management and semantic embedding |
| `src/dataset_preprocessor.py` | arXiv metadata cleaning and filtering |
| `src/ingest.py` | CLI tool for standalone ingestion |
| `app.py` | Streamlit web UI |

---

## Quick Start

### Prerequisites

- **Python 3.10+** (or Docker for containerized deployment)
- An MCP-compatible host (e.g., [Claude Desktop](https://claude.ai/desktop)) — optional

### Option 1: Local Python Installation

#### 1. Installation

```bash
git clone https://github.com/Ishwarpatra/OpenSourceSleuth.git
cd OpenSourceSleuth

python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:    .venv\Scripts\activate

pip install -e ".[dev,ui]"
```

#### 2. Add Your PDFs

Drop your academic PDF files into the `student_pdfs/` directory:

```
student_pdfs/
├── vaswani2017_attention.pdf
├── devlin2019_bert.pdf
└── brown2020_gpt3.pdf
```

#### 3. Launch the Web UI

```bash
streamlit run app.py
```

Access the dashboard at [http://localhost:8501](http://localhost:8501) to search, upload PDFs, and view index statistics.

---

### Option 2: Docker Deployment (Recommended for Production)

Docker provides a sandboxed environment that eliminates dependency conflicts and Python version issues. This is ideal for students who want a "just works" deployment.

#### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running

#### 1. Build the Docker Image

```bash
# From the project root directory
docker build -t sourcesleuth:latest .
```

#### 2. Create Persistent Volume

```bash
# Create Docker volume for FAISS index (persists between container restarts)
docker volume create sourcesleuth_vector_data
```

#### 3. Test the Container

```bash
# Test that the container starts correctly
docker run --rm \
  -v ./student_pdfs:/app/student_pdfs \
  -v sourcesleuth_vector_data:/app/data \
  sourcesleuth:latest \
  python -c "print('Container ready!')"
```

#### 4. Configure MCP Host for Docker

To use the Dockerized server with Claude Desktop, update your `claude_desktop_config.json`:

**macOS/Linux:**
```json
{
  "mcpServers": {
    "sourcesleuth": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-v", "/absolute/path/to/student_pdfs:/app/student_pdfs",
        "-v", "sourcesleuth_vector_data:/app/data",
        "sourcesleuth:latest"
      ]
    }
  }
}
```

**Windows:**
```json
{
  "mcpServers": {
    "sourcesleuth": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-v", "C:\\path\\to\\student_pdfs:/app/student_pdfs",
        "-v", "sourcesleuth_vector_data:/app/data",
        "sourcesleuth:latest"
      ]
    }
  }
}
```

**Why this configuration is mandatory:**

| Flag | Purpose | Consequence If Missing |
|------|---------|----------------------|
| `-i` | Keeps STDIN open for MCP JSON-RPC | Host cannot send queries |
| `--rm` | Removes container on exit | Orphaned containers consume memory |
| `-v ...:/app/student_pdfs` | Mounts user's PDFs | PyMuPDF cannot access files |
| `-v sourcesleuth_vector_data:/app/data` | Named volume for index | Re-ingestion required every restart |

**Important:** The MCP Host must use `docker run -i --rm` exactly as shown above. Do not use Docker Compose for MCP integration — Compose handles stdio differently and will break the MCP protocol communication.

Access the dashboard at [http://localhost:8501](http://localhost:8501) to search, upload PDFs, and view index statistics.

### 4. Or Connect via MCP

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "sourcesleuth": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/OpenSourceSleuth"
    }
  }
}
```

---

## CLI Usage

For power users, SourceSleuth includes a standalone CLI:

```bash
# Ingest PDFs from a specific directory
python -m src.ingest pdfs --directory /path/to/pdfs

# Ingest arXiv papers by category
python -m src.ingest arxiv --category cs.AI --max-records 5000

# View store statistics
python -m src.ingest stats

# Clear the vector store
python -m src.ingest clear
```

---

## Demo & Examples

### Example 1: Finding an Orphaned Quote via MCP

**Scenario:** You're writing a paper and find this quote in your draft without a citation:

> "The attention mechanism allows models to focus on specific parts of the input sequence, enabling them to capture long-range dependencies more effectively than recurrent architectures."

**Step 1:** Ask your AI assistant (Claude/Cursor/Windsurf):

```
I found this quote in my draft but lost the source. Can you help me find it?

"The attention mechanism allows models to focus on specific parts of the input 
sequence, enabling them to capture long-range dependencies more effectively than 
recurrent architectures."
```

**Step 2:** The AI uses SourceSleuth to search your indexed PDFs.

**Expected Output:**

```
I found a potential match in your documents:

### Match 1
- Document: `vaswani2017_attention.pdf`
- Page: 3
- Confidence: High (0.89)
- Context:
  > "We propose the Transformer, a model architecture based entirely on attention
  mechanisms. The attention mechanism allows models to focus on specific parts of
  the input sequence, enabling them to capture long-range dependencies more
  effectively than recurrent architectures without the constraints of sequential
  processing."

**Extracted Metadata**:
- Title: Attention Is All You Need
- Author(s): Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
  Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- Date: 2017
- Journal: Advances in Neural Information Processing Systems

Would you like me to format this as a citation?
```

**Step 3:** Request a citation:

```
Yes, please format as APA citation.
```

**Expected Output:**

```
Here's your citation in APA 7th edition format, using metadata extracted from 
the PDF:

**Full Citation:**
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. In Advances in
Neural Information Processing Systems (NeurIPS).

**In-Text Citation:**
(Vaswani et al., 2017)

**Notes:**
- All metadata was extracted from the PDF's embedded metadata dictionary.
- The author list and title were present in the PDF properties.
```

---

### How Metadata Extraction Works

SourceSleuth extracts actual metadata from PDFs during ingestion using PyMuPDF's
metadata dictionary. This eliminates hallucination risk from guessing authors
or titles from filenames.

**Extracted Fields:**
- **Title**: From PDF `/Title` metadata field
- **Author(s)**: From PDF `/Author` metadata field
- **Date**: From PDF `/CreationDate` or `/ModDate`
- **Publisher**: From PDF `/Creator` or `/Producer`
- **Journal**: From PDF `/Subject` field (when available)
- **DOI**: When present in PDF metadata

**When Metadata Is Missing:**

Some PDFs (especially older ones or student uploads) may lack proper metadata.
In such cases, SourceSleuth will:

1. Display "[Not found in PDF metadata - verify manually]" for missing fields
2. Suggest where to find the information (e.g., first page of the PDF)
3. Use the `sourcesleuth://pdfs/{filename}` resource to fetch full text if needed

This approach ensures the LLM never hallucinates citations from filenames alone.

---

### Example 2: Web UI Search Session

**Step 1:** Launch the Web UI

```bash
streamlit run app.py
```

**Step 2:** Navigate to http://localhost:8501

**Step 3:** Paste your query:

```
The photoelectric effect demonstrates that light behaves as discrete packets of energy called photons.
```

**Expected Output:**

```
Found 3 potential match(es)!

#1 — modern_physics_textbook.pdf  [HIGH - 0.87]
─────────────────────────────────────────────────────────────────
"The photoelectric effect demonstrates that light behaves as discrete 
packets of energy called photons, rather  than purely as waves. This 
groundbreaking discovery by Einstein in 1905 established the quantum 
nature of light..."

Page 142  |  Chunk #23  |  Chars 1250–3200

#2 — quantum_mechanics_intro.pdf  [MEDIUM - 0.62]
─────────────────────────────────────────────────────────────────
"Einstein's explanation of the photoelectric effect showed that light 
energy is quantized into discrete packets, later named photons..."

Page 8  |  Chunk #5  |  Chars 890–1840

#3 — physics_history_review.pdf  [LOW - 0.45]
─────────────────────────────────────────────────────────────────
"The concept of photons emerged from early 20th century experiments 
on the photoelectric effect..."

Page 15  |  Chunk #12  |  Chars 2100–3050
```

---

### Example 3: CLI Ingestion Workflow

**Step 1:** Ingest your PDFs

```bash
$ python -m src.ingest pdfs --directory ./student_pdfs

Processing PDFs from: ./student_pdfs
Found 15 PDF files
Processing vaswani2017_attention.pdf... Done
Processing devlin2019_bert.pdf... Done
Processing brown2020_gpt3.pdf... Done
...
Extracted 1,247 chunks from 15 documents
Vector store saved to ./data/sourcesleuth.index

Done! You can now search for orphaned quotes.
```

**Step 2:** View index statistics

```bash
$ python -m src.ingest stats

Vector Store Statistics
=======================
Total chunks:     1,247
Indexed files:    15
Embedding dim:    384
Index type:       IndexFlatIP
Memory usage:     ~2.1 MB
```

---

### Example 4: arXiv Paper Ingestion

**Step 1:** Ingest machine learning papers

```bash
$ python -m src.ingest arxiv --category cs.LG --max-records 1000

Fetching arXiv papers for category: cs.LG
Retrieved 1,000 papers
Preprocessing titles and abstracts...
Filtered 987 valid papers (removed duplicates and incomplete entries)
Extracted 8,542 chunks from 987 arXiv papers
Vector store updated successfully.

Total index size: 9,789 chunks (15 local PDFs + 987 arXiv papers)
```

---

### Example 5: Confidence Tier Interpretation

| Confidence Score | Tier | Interpretation |
|------------------|------|----------------|
| ≥ 0.75 | **High** | Likely the exact source or very close paraphrase |
| 0.50 – 0.74 | **Medium** | Possible match; verify context manually |
| < 0.50 | **Low** | Weak match; may be topically related but not the source |

**Example Results by Tier:**

```
[HIGH] (0.89): Exact quote match
   "The attention mechanism allows models to focus on specific parts..."

[MEDIUM] (0.68): Paraphrased concept
   "Attention mechanisms enable selective focus on input elements..."

[LOW] (0.42): Topically related
   "Neural networks can learn to weight different inputs differently..."
```

---

### Example 6: Index Management

```bash
# Clear and rebuild the index
$ python -m src.ingest clear
Vector store cleared successfully.

# Re-ingest PDFs
$ python -m src.ingest pdfs --directory ./student_pdfs
Processing 15 PDFs...
Extracted 1,247 chunks
Done!

# Verify the new index
$ python -m src.ingest stats
Total chunks:  1,247
Indexed files: 15
```

---

## AI/ML Track Documentation

This project meets all requirements for the AI/ML hackathon track. Full documentation is provided in separate files:

| Document | What It Covers |
|---|---|
| [MODEL_CARD.md](MODEL_CARD.md) | Model architecture, design rationale, alternatives considered, dataset documentation, reproducibility instructions |
| [EVALUATION.md](EVALUATION.md) | Evaluation methodology, metrics explained in plain language, benchmark results, known weaknesses |
| [ROADMAP.md](ROADMAP.md) | v1.0 through v2.0 feature roadmap and priority matrix |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute, code style, testing guidelines |

### Dataset & Preprocessing

- **Student PDFs**: Processed locally with PyMuPDF. Text is extracted page-by-page, then split into **500-token chunks with 50-token overlap** (~2000 characters per chunk). This preserves paragraph-level context while ensuring each chunk is semantically meaningful.
- **arXiv Papers**: Filtered from the [arXiv metadata dataset](https://www.kaggle.com/Cornell-University/arxiv). LaTeX commands are stripped, abstracts are cleaned, and title+abstract are concatenated as the searchable text.

### Model Architecture

We use [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — a 22.7M parameter sentence transformer that produces 384-dimensional embeddings. We chose this model because:

1. **CPU-only** — Runs in ~50ms per sentence on any laptop.
2. **Small footprint** — ~80 MB, won't fill up a student's disk.
3. **Strong zero-shot accuracy** — Trained on 1B+ sentence pairs.
4. **Privacy-first** — No API calls, no internet required after initial download.

See [MODEL_CARD.md](MODEL_CARD.md) for the full comparison with alternatives.

### Evaluation Summary

| Scenario | Top-3 Accuracy | Avg. Confidence |
|---|---|---|
| Exact quotes | 100% | 0.89 |
| Light paraphrase | 96% | 0.72 |
| Heavy paraphrase | 82% | 0.58 |

See [EVALUATION.md](EVALUATION.md) for full methodology, metrics in plain language, and known limitations.

### Limitations

- **English-only** — The current model is optimized for English text.
- **Math formulas** — LaTeX equations embed poorly; search by surrounding prose instead.
- **Scanned PDFs** — Image-only PDFs require OCR (planned for v1.1).
- **Very short queries** — Provide at least 5–10 words for reliable results.

---

## Cosine Similarity — The Core Algorithm

SourceSleuth finds orphaned quotes by computing the **cosine similarity** between the student's text and every chunk in the vector store:

$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{||A|| \times ||B||}$$

This measures the angle between two embedding vectors in 384-dimensional space. When vectors point in the same direction (high similarity), the text chunks share semantic meaning — even if the exact wording is different.

---

## Testing

```bash
# Run the full test suite
pytest

# Run specific modules
pytest tests/test_source_sleuth.py -v      # Standalone retriever
pytest tests/test_pdf_processor.py -v       # PDF extraction & chunking
pytest tests/test_vector_store.py -v        # FAISS store operations
pytest tests/test_mcp_server.py -v          # MCP tool functions
pytest tests/test_integration_pdfs.py -v    # End-to-end integration
```

---

## Project Structure

```
OpenSourceSleuth/
├── src/
│   ├── __init__.py
│   ├── mcp_server.py              # MCP server (tools, resources, prompts)
│   ├── source_sleuth.py           # Standalone SourceRetriever class
│   ├── pdf_processor.py           # PDF text extraction & chunking
│   ├── vector_store.py            # FAISS index & embedding management
│   ├── dataset_preprocessor.py    # arXiv metadata cleaning
│   └── ingest.py                  # CLI ingestion tool
├── tests/
│   ├── test_source_sleuth.py      # Standalone retriever tests
│   ├── test_mcp_server.py         # MCP tool integration tests
│   ├── test_pdf_processor.py      # PDF processing tests
│   ├── test_vector_store.py       # Vector store tests
│   ├── test_dataset_preprocessor.py
│   └── test_integration_pdfs.py   # End-to-end PDF tests
├── examples/
│   └── USAGE_EXAMPLES.md          # Detailed usage examples
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md          # Bug report template
│   │   └── feature_request.md     # Feature request template
│   └── PULL_REQUEST_TEMPLATE.md   # PR submission template
├── data/                          # Persisted vector store & arXiv data
├── student_pdfs/                  # Your academic PDFs go here
├── app.py                         # Streamlit web UI
├── .env.example                   # Environment configuration template
├── CHANGELOG.md                   # Version history and changes
├── CODE_OF_CONDUCT.md             # Community standards (Contributor Covenant)
├── CONTRIBUTING.md                # Contribution guidelines
├── EVALUATION.md                  # Evaluation methodology & results
├── MODEL_CARD.md                  # Model documentation (AI/ML track)
├── ROADMAP.md                     # Development roadmap
├── LICENSE                        # Apache 2.0
├── pyproject.toml                 # Package configuration
└── README.md                      # This file
```

---

## License

Licensed under the **Apache 2.0 License** — enterprise-friendly, patent-safe, and widely used in the ML ecosystem. See [LICENSE](LICENSE) for details.

---

## How to Contribute

We welcome contributions from everyone! Here's how to get started:

1. **Read the guidelines**: See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
2. **Review the Code of Conduct**: We adhere to the [Contributor Covenant](CODE_OF_CONDUCT.md).
3. **Find an issue**: Look for issues labeled [`good first issue`](https://github.com/Ishwarpatra/OpenSourceSleuth/labels/good%20first%20issue) or [`help wanted`](https://github.com/Ishwarpatra/OpenSourceSleuth/labels/help%20wanted).
4. **Check the roadmap**: See [ROADMAP.md](ROADMAP.md) for planned features.
5. **Fork and clone**: Create your own fork and clone the repository.
6. **Set up your environment**: Follow the [Quick Start](#quick-start) guide.
7. **Submit a PR**: Open a pull request using the [PR template](.github/PULL_REQUEST_TEMPLATE.md).

### Good First Contributions

- Add unit tests for existing functionality
- Improve documentation or add examples
- Fix bugs labeled "good first issue"
- Add input validation to tool arguments
- Add support for additional citation styles

See [CONTRIBUTING.md](CONTRIBUTING.md) for more ideas.

---

## Community & Support

- **Issues**: [GitHub Issues](https://github.com/Ishwarpatra/OpenSourceSleuth/issues) for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and ideas
- **Documentation**: [README.md](README.md), [MODEL_CARD.md](MODEL_CARD.md), [examples/USAGE_EXAMPLES.md](examples/)

---

## Acknowledgments

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io) — The "USB-C for AI" standard
- [Sentence-Transformers](https://sbert.net) — Embedding model framework
- [FAISS](https://github.com/facebookresearch/faiss) — Similarity search engine
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF text extraction
- [Contributor Covenant](https://www.contributor-covenant.org) — Code of Conduct template