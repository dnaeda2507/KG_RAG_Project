"""
config.py — Merkezi Sabitler
=============================
Tüm modüller bu dosyadan import eder.
Değiştirmek için tek yer burasıdır.

Kullanım:
    from kg_infused_rag.config import NEO4J_URI, GROQ_MODEL, K_E
    # veya proje kökünden çalıştırırken:
    from config import NEO4J_URI, GROQ_MODEL, K_E
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Neo4j Bağlantısı ──────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

# ── Groq LLM ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"

# ── Embedding Modeli ──────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── KG Domain ─────────────────────────────────────────────────────────────────
TURKEY_ID = "Q43"

# ── Module 1: Spreading Activation ───────────────────────────────────────────
K_E                     = 3    # Başlangıç seed entity sayısı
MAX_ROUNDS              = 6    # Maksimum aktivasyon turu
MAX_ENTITIES_PER_ROUND  = 3    # Tur başına eklenen entity sayısı
MAX_TRIPLES_PER_ENTITY  = 10   # Entity başına maksimum triple
KEYWORD_CANDIDATE_LIMIT = 150  # Keyword arama aday sınırı

# ── Module 2: Query Expansion & Wikipedia Retrieval ───────────────────────────
K_P               = 6     # Toplam passage sayısı
MAX_PASSAGE_CHARS = 400   # Passage başına maksimum karakter (pipeline & module3)
MAX_PASSAGE_LEN   = 500   # Passage başına maksimum karakter (module2)
WIKI_SEARCH_DELAY = 0.4   # Wikipedia arama sonrası bekleme (saniye)
WIKI_PAGE_DELAY   = 0.2   # Wikipedia sayfa çekimi sonrası bekleme (saniye)
WIKI_LANGUAGE     = "en"

# ── Retry Mekanizması ─────────────────────────────────────────────────────────
RETRY_MAX_RETRIES = 3
RETRY_DELAY       = 2.0   # saniye

# ── Çıktı Klasörleri ─────────────────────────────────────────────────────────
OUTPUT_ROOT          = "outputs"
OUTPUT_PHASE1        = "outputs"
OUTPUT_PHASE2_CINEMA = "outputs/cinema"
OUTPUT_PHASE2_FOOTBALL = "outputs/football"
OUTPUT_PHASE3_CINEMA   = "outputs/phase3"
OUTPUT_PHASE3_FOOTBALL = "outputs/phase3_football"
OUTPUT_PHASE4          = "outputs/phase4"
OUTPUT_PHASE5          = "outputs/phase5"
OUTPUT_PHASE6          = "outputs/phase6"