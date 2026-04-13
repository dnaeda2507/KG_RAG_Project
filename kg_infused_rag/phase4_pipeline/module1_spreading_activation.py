"""
Module 1 - KG-Guided Spreading Activation
==========================================
Sorguya en alakalı seed entity'leri Neo4j'den bulur,
iteratif olarak KG üzerinde yayılım yapar ve
toplanan triple'lardan doğal dil özeti üretir.

Düzeltmeler:
  - Seed seçimi: CANDIDATE_POOL kaldırıldı, sadece keyword match kullanılıyor
  - rounds_completed: gerçek tur sayısı döndürülüyor
  - LLM retry mekanizması eklendi
  - driver.close() güvenli hale getirildi
"""

import os
import json
import time
import functools
from typing import Optional
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq

load_dotenv()

# ── Parametreler ──────────────────────────────────────────────────────────────
K_E                    = 3    # Başlangıç seed entity sayısı
MAX_ROUNDS             = 6    # Maksimum yayılım turu
MAX_ENTITIES_PER_ROUND = 3    # Tur başına maksimum yeni entity
MAX_TRIPLES_PER_ENTITY = 10   # Entity başına maksimum triple
KEYWORD_CANDIDATE_LIMIT = 150 # Her keyword için Neo4j'den çekilecek max entity
GROQ_MODEL             = "llama-3.1-8b-instant"

# ── Neo4j & LLM bağlantıları ──────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")


# ── Retry decorator ───────────────────────────────────────────────────────────
def retry_on_failure(max_retries: int = 3, delay: float = 2.0):
    """LLM/API çağrıları için üstel geri çekilmeli retry decorator."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    wait = delay * (attempt + 1)
                    print(f"  [Retry] {func.__name__} hata (deneme {attempt+1}/{max_retries}): "
                          f"{e} — {wait:.1f}s bekleniyor.")
                    time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator


class SpreadingActivation:
    """
    KG-Infused RAG — Modül 1
    Spreading Activation üzerinden KG subgraph özeti üretir.
    """

    def __init__(self):
        print("[Module1] Başlatılıyor...")
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.groq        = Groq(api_key=GROQ_API_KEY)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Module1] ✔ Neo4j, Groq ve Embedding modeli hazır.")

    # ──────────────────────────────────────────────────────────────────────────
    # NEO4J YARDIMCI FONKSİYONLAR
    # ──────────────────────────────────────────────────────────────────────────

    def _run_cypher(self, cypher: str, params: dict = None) -> list:
        with self.driver.session() as session:
            return [r.data() for r in session.run(cypher, params or {})]

    def _get_candidate_entities(self, query: str) -> list:
        """
        Sorgu kelimelerini Neo4j description'da arar (keyword match).
        Genel yüksek-dereceli entity havuzu KULLANILMIYOR — sorguyla alakasız
        entity'lerin seed seçimine karışmasını önlemek için.
        """
        if not query:
            return []

        # 4+ karakter uzunluğundaki kelimeleri filtrele (stop-word benzeri)
        keywords = [w for w in query.lower().split() if len(w) > 3]
        results  = []

        for kw in keywords[:6]:
            hits = self._run_cypher("""
                MATCH (e:Entity)
                WHERE toLower(e.description) CONTAINS $kw
                  AND e.description IS NOT NULL
                  AND size(e.description) > 10
                RETURN e.entityId AS id, e.description AS desc
                LIMIT $limit
            """, {"kw": kw, "limit": KEYWORD_CANDIDATE_LIMIT})
            results.extend(hits)

        # Tekrarları temizle
        seen, unique = set(), []
        for r in results:
            if r["id"] not in seen:
                seen.add(r["id"])
                unique.append(r)

        return unique

    def _get_one_hop_triples(self, entity_id: str) -> list:
        """Bir entity'nin 1-hop komşularını döndürür."""
        return self._run_cypher("""
            MATCH (e:Entity {entityId: $eid})-[r]->(target:Entity)
            WHERE target.description IS NOT NULL
            RETURN type(r)            AS relation,
                   target.entityId    AS target_id,
                   target.description AS target_desc
            LIMIT $limit
        """, {"eid": entity_id, "limit": MAX_TRIPLES_PER_ENTITY})

    def _get_entity_description(self, entity_id: str) -> str:
        """Entity description'ını döndürür."""
        results = self._run_cypher("""
            MATCH (e:Entity {entityId: $eid})
            RETURN e.description AS desc
        """, {"eid": entity_id})
        return results[0]["desc"] if results else ""

    # ──────────────────────────────────────────────────────────────────────────
    # ADIM 1: SEED ENTITY SEÇİMİ
    # ──────────────────────────────────────────────────────────────────────────

    def select_seed_entities(self, query: str) -> list:
        """
        Sorgu ile entity description'ları arasındaki cosine benzerliğini
        hesaplar ve en yüksek K_E entity'yi seed olarak seçer.
        Aday havuzu sadece keyword-matched entity'lerden oluşur.
        """
        print(f"\n[Step 1] Seed entity seçiliyor (k_e={K_E})...")
        candidates = self._get_candidate_entities(query)

        if not candidates:
            print("[Step 1] ⚠ Aday entity bulunamadı.")
            return []

        print(f"  {len(candidates)} keyword-matched aday entity bulundu.")

        # Embedding hesapla
        query_emb      = self.embed_model.encode([query])
        desc_texts     = [c["desc"] for c in candidates]
        candidate_embs = self.embed_model.encode(
            desc_texts, batch_size=64, show_progress_bar=False
        )

        # Cosine benzerlik
        similarities = cosine_similarity(query_emb, candidate_embs)[0]
        top_k        = min(K_E, len(candidates))
        top_indices  = np.argsort(similarities)[::-1][:top_k]

        seeds = []
        for idx in top_indices:
            seeds.append({
                "id":         candidates[idx]["id"],
                "desc":       candidates[idx]["desc"],
                "similarity": float(similarities[idx])
            })
            print(f"  ✔ Seed: {candidates[idx]['id']:12} | "
                  f"sim={similarities[idx]:.3f} | "
                  f"{str(candidates[idx]['desc'])[:60]}")

        return seeds

    # ──────────────────────────────────────────────────────────────────────────
    # ADIM 2: İTERATİF YAYILIM
    # ──────────────────────────────────────────────────────────────────────────

    @retry_on_failure(max_retries=3, delay=2.0)
    def _llm_select_relevant_triples(
        self, query: str, entity_desc: str, triples: list
    ) -> list:
        """
        LLM'e sorguyu ve triple listesini verir,
        sorguyla alakalı triple'ları seçmesini ister.
        JSON liste olarak döner.
        """
        if not triples:
            return []

        triples_text = "\n".join(
            f"{i+1}. [{t['relation']}] -> {t['target_desc']}"
            for i, t in enumerate(triples)
        )

        prompt = f"""You are a knowledge graph reasoning assistant.

Query: {query}
Current Entity: {entity_desc}

Available triples from this entity:
{triples_text}

Select ONLY the triples that are relevant to answering the query.
Return a JSON list of the relevant triple numbers (1-based index).
Example: [1, 3, 5]
Return ONLY the JSON list, nothing else."""

        response = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()

        try:
            indices  = json.loads(raw)
            selected = [triples[i-1] for i in indices if 1 <= i <= len(triples)]
            return selected
        except json.JSONDecodeError:
            print(f"  [LLM] JSON parse hatası: '{raw}' — ilk 3 triple alınıyor.")
            return triples[:3]

    def iterative_spreading(
        self, query: str, seeds: list
    ) -> tuple[list, list, int]:
        """
        Seed entity'lerden başlayarak iteratif yayılım yapar.

        Returns:
            (activated_triples, visited_entity_ids, actual_rounds_completed)
        """
        print(f"\n[Step 2] Iteratif spreading activation (max_rounds={MAX_ROUNDS})...")

        visited        = set()
        activated      = []
        current_queue  = [s["id"] for s in seeds]
        actual_rounds  = 0

        for seed in seeds:
            visited.add(seed["id"])

        for round_num in range(1, MAX_ROUNDS + 1):
            actual_rounds = round_num

            if not current_queue:
                print(f"  [Round {round_num}] Kuyruk boş, durduruluyor.")
                break

            print(f"\n  [Round {round_num}] {len(current_queue)} entity işleniyor...")
            next_queue    = []
            round_triples = []

            for entity_id in current_queue:
                entity_desc = self._get_entity_description(entity_id)
                triples     = self._get_one_hop_triples(entity_id)

                if not triples:
                    continue

                try:
                    selected = self._llm_select_relevant_triples(
                        query, entity_desc, triples
                    )
                except Exception as e:
                    print(f"  [Round {round_num}] LLM hatası {entity_id}: {e} — atlanıyor.")
                    selected = triples[:2]  # son fallback

                for t in selected:
                    triple_record = {
                        "source_id":   entity_id,
                        "source_desc": entity_desc,
                        "relation":    t["relation"],
                        "target_id":   t["target_id"],
                        "target_desc": t["target_desc"],
                        "round":       round_num,
                    }
                    activated.append(triple_record)
                    round_triples.append(triple_record)

                    if (t["target_id"] not in visited and
                            len(next_queue) < MAX_ENTITIES_PER_ROUND):
                        next_queue.append(t["target_id"])
                        visited.add(t["target_id"])

                print(f"    Entity {entity_id[:12]:12} → {len(selected)} triple seçildi")
                time.sleep(0.3)

            print(f"  [Round {round_num}] {len(round_triples)} triple aktivasyona eklendi.")

            if not next_queue:
                print(f"  [Round {round_num}] Yeni entity yok, durduruluyor.")
                break

            current_queue = next_queue

        print(f"\n  Toplam aktivasyon: {len(activated)} triple, "
              f"{len(visited)} entity ziyaret edildi, "
              f"{actual_rounds} tur tamamlandı.")
        return activated, list(visited), actual_rounds

    # ──────────────────────────────────────────────────────────────────────────
    # ADIM 3: KG SUBGRAPH ÖZETİ
    # ──────────────────────────────────────────────────────────────────────────

    @retry_on_failure(max_retries=3, delay=2.0)
    def summarize_subgraph(self, query: str, activated_triples: list) -> str:
        """
        Aktivasyon sırasında toplanan triple'ları
        LLM kullanarak doğal dil özetine dönüştürür.
        """
        print("\n[Step 3] KG subgraph özeti üretiliyor...")

        if not activated_triples:
            return "No relevant knowledge graph information found."

        triples_text = "\n".join(
            f"- {t['source_desc']} --[{t['relation']}]--> {t['target_desc']}"
            for t in activated_triples[:30]
        )

        prompt = f"""You are a knowledge graph summarization assistant.

Query: {query}

The following triples were collected from the knowledge graph:
{triples_text}

Write a concise natural language summary of this knowledge graph information
that is relevant to the query. Focus on facts that help answer the question.
Write 3-5 sentences maximum. Be specific and factual."""

        response = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        summary = response.choices[0].message.content.strip()
        print(f"  ✔ KG Özeti ({len(summary)} karakter)")
        return summary

    # ──────────────────────────────────────────────────────────────────────────
    # ANA FONKSİYON
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, query: str) -> dict:
        """
        Spreading activation pipeline'ını çalıştırır.

        Returns:
            {
                "seed_entities":      list,
                "activated_triples":  list,
                "kg_summary":         str,
                "visited_entities":   list,
                "rounds_completed":   int   # gerçek tamamlanan tur sayısı
            }
        """
        print("\n" + "=" * 60)
        print(f"[Module 1] Spreading Activation")
        print(f"Query: {query}")
        print("=" * 60)

        seeds = self.select_seed_entities(query)

        if not seeds:
            return {
                "seed_entities":     [],
                "activated_triples": [],
                "kg_summary":        "No seed entities found.",
                "visited_entities":  [],
                "rounds_completed":  0,
            }

        # gerçek tur sayısını al (3. dönüş değeri)
        activated, visited, actual_rounds = self.iterative_spreading(query, seeds)

        try:
            kg_summary = self.summarize_subgraph(query, activated)
        except Exception as e:
            print(f"  [Module1] Özetleme başarısız: {e}")
            kg_summary = " ".join(
                f"{t['source_desc']} {t['relation'].lower()} {t['target_desc']}."
                for t in activated[:5]
            )

        result = {
            "seed_entities":     seeds,
            "activated_triples": activated,
            "kg_summary":        kg_summary,
            "visited_entities":  visited,
            "rounds_completed":  actual_rounds,  # ✔ Düzeltildi
        }

        print("\n[Module 1] ✅ Tamamlandı.")
        return result

    def close(self):
        self.driver.close()


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "Galatasaray'ın teknik direktörünün doğum yeri neresidir?",
        "Bir Türk filminin yönetmeni hangi ödülü almıştır?",
        "Fenerbahçe'de oynayan futbolcunun doğduğu şehir hangisidir?",
    ]

    sa = SpreadingActivation()
    try:
        for query in test_queries[:1]:
            result = sa.run(query)

            print("\n" + "─" * 60)
            print("SONUÇ:")
            print(f"  Seed entity sayısı    : {len(result['seed_entities'])}")
            print(f"  Aktive triple sayısı  : {len(result['activated_triples'])}")
            print(f"  Ziyaret edilen entity : {len(result['visited_entities'])}")
            print(f"  Tamamlanan tur        : {result['rounds_completed']}")
            print(f"\nKG Özeti:\n{result['kg_summary']}")

            os.makedirs("outputs/phase4", exist_ok=True)
            with open("outputs/phase4/module1_test_result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print("\n  ✔ Test sonucu: outputs/phase4/module1_test_result.json")
    finally:
        sa.close()
