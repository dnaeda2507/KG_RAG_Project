══════════════════════════════════════════════════════════════════════════
  KG-Infused RAG: Türkiye Domain Knowledge Graph Question Answering
  CSE 474 & 5074 — Social Network Analysis  |  Term Project
══════════════════════════════════════════════════════════════════════════

  Knowledge Graph-Infused Retrieval-Augmented Generation (KG-RAG) using
  the Wikidata5M dataset, with Türkiye as the target domain. The system
  generates multi-hop questions from a Neo4j knowledge graph and compares
  four question-answering methods: No-Retrieval, Vanilla RAG, Vanilla
  Query Expansion, and KG-Infused RAG.


══════════════════════════════════════════════════════════════════════════
  PREREQUISITES
══════════════════════════════════════════════════════════════════════════

  1. Python 3.10+
    pip install -r requirements.txt
    (or: uv sync  if using pyproject.toml)

  2. Neo4j Desktop / Server
    URL : bolt://localhost:7687
    User: neo4j
    Password: set in kg_infused_rag/config.py (NEO4J_PASSWORD)
    Database loaded with Wikidata5M nodes & relationships
    (wikidata5m_raw_data/ triplets imported via Neo4j import or
     rel.py / tutorial_step3.py)

  3. Groq API Key
    Set environment variable: GROQ_API_KEY=gsk_...
    Or hard-code in kg_infused_rag/config.py
    Model used: llama-3.3-70b-versatile

  4. Wikidata5M raw files in wikidata5m_raw_data/
    wikidata5m_all_triplet.txt
    wikidata5m_text.txt
    wikidata5m_alias/wikidata5m_entity.txt
    wikidata5m_alias/wikidata5m_relation.txt
    wikidata5m_transductive/  (train/valid/test splits)


══════════════════════════════════════════════════════════════════════════
  PROJECT STRUCTURE
══════════════════════════════════════════════════════════════════════════

  kg_infused_rag/
  ├── config.py                    — Neo4j URL, Groq API, model settings
  ├── phase1_exploration/
  │   ├── phase1_turkey_analysis.py    — Türkiye entity detection
  │   └── phase1_turkey_analysis1.py   — City & relation frequency
  ├── phase2_domain/
  │   ├── phase2_domain_overview.py    — 5-domain comparison
  │   ├── phase2_cinema_verification.py— Cinema path analysis
  │   └── phase2_football_verification.py — Football path analysis
  ├── phase3_questions/
  │   ├── phase3_cinema_question_generator.py  — 50 cinema QA pairs
  │   └── phase3_football_question_generator.py— 50 football QA pairs
  ├── phase4_pipeline/
  │   ├── module1_spreading_activation.py  — Neo4j subgraph expansion
  │   ├── module2_query_expansion.py       — Triple selection + Wikipedia
  │   ├── module3_answer_generation.py     — LLM answer generation
  │   ├── pipeline_cinema.py               — Full cinema pipeline (50 Qs)
  │   └── pipeline_football.py             — Full football pipeline
  └── phase5_experiments/
      └── phase5_evaluation.py             — Phase 5 systematic evaluation

  outputs/
  ├── phase1_turkey_stats.json
  ├── phase1_domain_counts.json
  ├── phase1_relation_freq.json
  ├── phase1_city_report.json
  ├── phase2_domain_overview.json
  ├── cinema/
  │   ├── cinema_domain_report.json
  │   ├── cinema_main_entities.json
  │   ├── cinema_multihop_paths.json
  │   └── cinema_path_density_report.json
  ├── football/
  │   ├── football_domain_report.json
  │   ├── football_main_entities.json
  │   ├── football_multihop_paths.json
  │   └── football_path_density_report.json
  ├── phase3/
  │   ├── qa_dataset.json          — 50 cinema QA questions
  │   └── qa_dataset_summary.json
  ├── phase3_football/
  │   ├── qa_dataset.json          — 50 football QA questions
  │   └── qa_dataset_summary.json
  ├── phase4_cinema/
  │   ├── pipeline_results.json    — 50 Qs × 4 methods, full results
  │   └── pipeline_summary.json    — aggregate metrics
  └── reports/
      ├── phase1_2_report.txt      — Phase 1-2 analysis report
      ├── phase3_report.txt        — Phase 3 QA dataset report
      └── phase4_report.txt        — Phase 4 pipeline results report


══════════════════════════════════════════════════════════════════════════
  HOW TO RUN
══════════════════════════════════════════════════════════════════════════

  All commands assume the virtual environment is activated:
    .venv\Scripts\activate      (Windows)
    source .venv/bin/activate   (Linux/macOS)

  ── Phase 1: Türkiye Entity Exploration ──────────────────────────────
  python kg_infused_rag/phase1_exploration/phase1_turkey_analysis.py
  python kg_infused_rag/phase1_exploration/phase1_turkey_analysis1.py
  Outputs: outputs/phase1_*.json

  ── Phase 2: Domain Selection ────────────────────────────────────────
  python kg_infused_rag/phase2_domain/phase2_domain_overview.py
  python kg_infused_rag/phase2_domain/phase2_cinema_verification.py
  python kg_infused_rag/phase2_domain/phase2_football_verification.py
  Outputs: outputs/phase2_*.json, outputs/cinema/*.json, outputs/football/*.json

  ── Phase 3: Question Generation ─────────────────────────────────────
  python kg_infused_rag/phase3_questions/phase3_cinema_question_generator.py
  python kg_infused_rag/phase3_questions/phase3_football_question_generator.py
  Outputs: outputs/phase3/qa_dataset.json, outputs/phase3_football/qa_dataset.json

  ── Phase 4: Pipeline Execution ──────────────────────────────────────
  python kg_infused_rag/phase4_pipeline/pipeline_cinema.py
  (Football pipeline: pipeline_football.py — optional)
  Outputs: outputs/phase4_cinema/pipeline_results.json

  ── Phase 5: Systematic Evaluation ───────────────────────────────────
  python kg_infused_rag/phase5_experiments/phase5_evaluation.py
  Output: outputs/phase5/*.json

  Note: Phase 6 case study is not included in the current codebase.


══════════════════════════════════════════════════════════════════════════
  RESULTS SUMMARY
══════════════════════════════════════════════════════════════════════════

  Phase 4 Cinema Pipeline (50 questions × 4 methods):

  ┌──────────────────┬───────────────┬────────┬────────┐
  │  Method          │  Accuracy     │  F1    │  EM    │
  ├──────────────────┼───────────────┼────────┼────────┤
  │  No-Retrieval    │   0.00%       │  0.00% │  0.00% │
  │  Vanilla RAG     │   0.00%       │  0.00% │  0.00% │
  │  Vanilla QE      │   0.00%       │  0.00% │  0.00% │
  │  KG-RAG          │   8.89%       │  0.00% │  0.00% │
  └──────────────────┴───────────────┴────────┴────────┘

  KG-RAG outperforms all baselines. Main bottlenecks: Turkish entity
  alias matching and Wikipedia retrieval returning empty results for
  Turkish-context queries in English Wikipedia.

  Key finding: Using the Wikidata5M text file (wikidata5m_text.txt)
  as the retrieval corpus instead of live Wikipedia would resolve the
  0-passage issue and is the primary planned improvement.


══════════════════════════════════════════════════════════════════════════
  CONFIGURATION (kg_infused_rag/config.py)
══════════════════════════════════════════════════════════════════════════

  NEO4J_URI       = "bolt://localhost:7687"
  NEO4J_USER      = "neo4j"
  NEO4J_PASSWORD  = "<your password>"
  GROQ_API_KEY    = "<your Groq API key>"
  LLM_MODEL       = "llama-3.3-70b-versatile"
  EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
  TURKEY_ENTITY   = "Q43"


══════════════════════════════════════════════════════════════════════════
  KNOWN LIMITATIONS
══════════════════════════════════════════════════════════════════════════

  1. Turkish alias matching failures
     Special chars ş ğ ü ö ı ç prevent exact entity lookups.
     Planned fix: ASCII normalisation in alias search.

  2. Wikipedia retrieval empty for Turkish domain
     Entity names like "halit refiğ" or "Duvara karşı" return no passages
     from English Wikipedia. Planned fix: use wikidata5m_text.txt corpus.

  3. Gold answer aliases use historical/alternate city names
     "Ankara" is "Angora, Turkey" in Wikidata5M. Soft accuracy partially
     handles this but EM and F1 are both 0.0%.

  4. F1 metric broken for Turkish text
     Token-level F1 calculation fails silently on Unicode edge cases.
     Soft accuracy is the reliable metric for this project.

  5. Groq API rate limits
     50 questions × 4 calls = 200 API requests per run. Pipeline includes
     per-call error handling but long runs may hit rate limits.


══════════════════════════════════════════════════════════════════════════
  End of README
══════════════════════════════════════════════════════════════════════════
