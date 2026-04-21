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

GROQ_API_KEY   = os.getenv("GROQ_API_KEY",  "")
GROQ_API_KEY1  = os.getenv("GROQ_API_KEY1", "")
GROQ_API_KEY2  = os.getenv("GROQ_API_KEY2", "")
GROQ_API_KEY3 = os.getenv("GROQ_API_KEY3", "")
GROQ_API_KEY4 = os.getenv("GROQ_API_KEY4", "")
GROQ_API_KEY5 = os.getenv("GROQ_API_KEY5", "")
# 429 rate limit gelince sırayla denenen key listesi (yeni key önce)
GROQ_API_KEYS = [k for k in [GROQ_API_KEY5, GROQ_API_KEY4, GROQ_API_KEY3, GROQ_API_KEY2, GROQ_API_KEY1, GROQ_API_KEY] if k]
GROQ_MODEL    = "llama-3.1-8b-instant"


class GroqKeyRotator:
    """
    Birden fazla Groq API key'ini yönetir.
    429 rate limit hatası gelince bir sonraki key'e geçer.
    self.groq.chat.completions.create(...) arayüzünü sunar.
    """
    def __init__(self, keys: list):
        from groq import Groq
        self._keys   = keys if keys else [GROQ_API_KEY]
        self._idx    = 0
        self._client = Groq(api_key=self._keys[0])
        self.chat    = self._client.chat

    def rotate(self):
        """Bir sonraki key'e geç ve client'ı yenile."""
        from groq import Groq
        self._idx    = (self._idx + 1) % len(self._keys)
        self._client = Groq(api_key=self._keys[self._idx])
        self.chat    = self._client.chat
        print(f"  [KeyRotator] Key #{self._idx + 1} kullanılıyor.")

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
MAX_PASSAGE_CHARS = 600    # 400 → 600
MAX_PASSAGE_LEN   = 800
WIKI_SEARCH_DELAY = 0.4   # Wikipedia arama sonrası bekleme (saniye)
WIKI_PAGE_DELAY   = 0.2   # Wikipedia sayfa çekimi sonrası bekleme (saniye)
WIKI_LANGUAGE     = "en"  # Sorular İngilizce olduğu için English Wikipedia

# ── Retry Mekanizması ─────────────────────────────────────────────────────────
RETRY_MAX_RETRIES = 3
RETRY_DELAY       = 15.0   # saniye (Groq 429 rate limit için yeterli süre)

# ── Çıktı Klasörleri ─────────────────────────────────────────────────────────
OUTPUT_ROOT          = "outputs"

OUTPUT_PHASE1        = "outputs/phase1_dataset_exploration"

OUTPUT_PHASE2        = "outputs/phase2_domain_selection"
OUTPUT_PHASE2_CINEMA    = "outputs/phase2_domain_selection/cinema"
OUTPUT_PHASE2_FOOTBALL  = "outputs/phase2_domain_selection/football"
OUTPUT_PHASE2_ACADEMIA  = "outputs/phase2_domain_selection/academia"

OUTPUT_PHASE3          = "outputs/phase3_question_generation"
OUTPUT_PHASE3_CINEMA   = "outputs/phase3_question_generation/cinema"
OUTPUT_PHASE3_FOOTBALL = "outputs/phase3_question_generation/football"
OUTPUT_PHASE3_ACADEMIA = "outputs/phase3_question_generation/academia"

OUTPUT_PHASE4 = "outputs/phase4_kg_infused_rag"
OUTPUT_PHASE5 = "outputs/phase5_experiments"
OUTPUT_PHASE6 = "outputs/phase6_case_study"


