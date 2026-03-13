# SourceSleuth Roadmap

This document outlines the development roadmap for SourceSleuth, from initial release to future enhancements.

---

## v1.0 — Initial Release (Current)

**Status:** Complete

### Core Features
- [x] MCP server with stdio transport
- [x] `find_orphaned_quote` tool — semantic search for orphaned quotes
- [x] `ingest_pdfs` tool — batch PDF ingestion (with OCR support for scanned PDFs)
- [x] `ingest_arxiv` tool — arXiv abstract ingestion
- [x] `get_store_stats` tool — vector store statistics
- [x] `sourcesleuth://pdfs/{filename}` resource — read full PDF text
- [x] `cite_recovered_source` prompt — citation formatting
- [x] CLI ingestion tool (`python -m src.ingest`)

### Technical Stack
- [x] PDF parsing with PyMuPDF (fitz)
- [x] FAISS `IndexFlatIP` for exact similarity search
- [x] SentenceTransformers (`all-MiniLM-L6-v2`) for embeddings
- [x] Local persistence of vector store
- [x] Environment-based configuration
- [x] OCR support with Tesseract for scanned documents
- [x] Hybrid search (FAISS + BM25 + RRF fusion)
- [x] POS-tagged query expansion with WordNet

### Documentation
- [x] README.md with installation and usage guide
- [x] AI/ML track documentation (models, datasets, evaluation)
- [x] Configuration documentation
- [x] Contributing guidelines
- [x] Docker deployment documentation

---

## v1.1 — Enhanced Document Support (Q2 2026)

**Status:** Planned

### Features
- [ ] **GraphRAG Integration** — Build citation graphs between indexed papers
- [ ] **Multi-modal Understanding** — Extract and index figures, charts, and tables
- [ ] **Automatic Citation Generation** — Generate BibTeX/EndNote from recovered sources
- [ ] **Table Extraction** — Preserve table structure during parsing
- [ ] **Figure Caption Recovery** — Extract and index figure captions
- [ ] **Multi-language Support** — Embeddings for non-English papers

### Technical Improvements
- [ ] Configurable chunk size via `.env`
- [ ] Progress bars for CLI ingestion
- [ ] Batch query support for `find_orphaned_quote`
- [ ] Cross-encoder re-ranking for improved accuracy

---

## v1.2 — Advanced Search Features (Q3 2026)

**Status:** Planned

### Features
- [ ] **Hybrid Search** — Combine keyword (BM25) + semantic search
- [ ] **Date Filtering** — Filter results by publication date
- [ ] **Author Filtering** — Search by specific authors
- [ ] **Citation Graph** — Track citation relationships between papers
- [ ] **Re-ranking** — Cross-encoder re-ranking for top-k results

### Configuration
- [ ] Configurable chunk size and overlap via CLI flags
- [ ] Support for multiple embedding models
- [ ] Custom stopword lists

---

## v1.5 — Format Expansion (Q4 2026)

**Status:** Planned

### New Formats
- [ ] **EPUB Support** — Ingest e-books
- [ ] **DOCX Support** — Microsoft Word documents
- [ ] **LaTeX Source** — Parse `.tex` files directly
- [ ] **HTML/Web Pages** — Archive and search web content

### PDF Enhancements
- [ ] **Layout-Aware Parsing** — Use `pdfplumber` for two-column papers
- [ ] **Math Formula Handling** — Better LaTeX expression extraction
- [ ] **Reference Section Parsing** — Auto-extract bibliography

### Metadata
- [ ] **CrossRef API Integration** — Automatic metadata lookup
- [ ] **DOI Resolution** — Fetch metadata from DOI
- [ ] **BibTeX Export** — Generate citations directly

---

## v2.0 — Scalability (Q1 2027)

**Status:** Planned

### Performance
- [ ] **Disk-Backed Index** — FAISS `IndexIVFFlat` for large corpora
- [ ] **Incremental Indexing** — Add new PDFs without re-ingestion
- [ ] **Parallel Processing** — Multi-threaded PDF ingestion
- [ ] **Memory Mapping** — Load large indices without full RAM usage

### Scale Targets
- Support 1M+ chunks (~10,000 PDFs)
- Query latency < 1s for 100k chunks
- Index size < 10GB for 100k chunks

---

## Future Considerations

### AI/ML Improvements
- [ ] **Domain-Specific Embeddings** — Fine-tune on academic text
- [ ] **Multi-Modal Search** — Include figure/image embeddings
- [ ] **Query Expansion** — Automatic synonym/hyponym expansion
- [ ] **Zero-Shot Classification** — Auto-categorize papers by topic

### User Experience
- [ ] **Web UI** — Browser-based search interface
- [ ] **Desktop App** — Native GUI for non-technical users
- [ ] **VS Code Extension** — Integrated citation search
- [ ] **Zotero/Mendeley Plugin** — Direct integration with reference managers

### Collaboration
- [ ] **Shared Vector Stores** — Team-based document collections
- [ ] **Cloud Sync** — Optional cloud backup (opt-in)
- [ ] **Public Corpus** — Pre-indexed arXiv subset for instant search

---

## Priority Matrix

| Feature | Impact | Effort | Priority |
|---|---|---|---|
| OCR Integration | High | Medium | P0 |
| Configurable Chunking | High | Low | P0 |
| Hybrid Search | Medium | Medium | P1 |
| Layout-Aware PDF | High | High | P1 |
| Disk-Backed Index | Medium | High | P2 |
| Web UI | Medium | High | P2 |
| Multi-Modal Search | Low | High | P3 |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Good First Issues
- Add unit tests for new features
- Improve error messages and logging
- Add support for additional embedding models
- Write tutorials and examples

### Areas We Need Help
- **Document parsers** — EPUB, DOCX, LaTeX
- **ML models** — Fine-tuning embeddings for academic text
- **UI/UX** — Web and desktop interfaces
- **Testing** — Increase test coverage
- **Documentation** — Tutorials, examples, API docs

---

## Changelog

### v1.0.0 (March 2026)
- Initial release
- MCP server with tools, resources, and prompts
- PDF and arXiv ingestion
- FAISS vector store with persistence
- CLI ingestion tool

---

## Acknowledgments

This roadmap was inspired by feedback from the open-source community and requirements from the AI/ML hackathon track.

Special thanks to:
- [Model Context Protocol](https://modelcontextprotocol.io) team
- [Sentence-Transformers](https://sbert.net) maintainers
- [FAISS](https://github.com/facebookresearch/faiss) contributors
