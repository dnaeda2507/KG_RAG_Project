"""
Module 1 - KG-Guided Spreading Activation
==========================================
Sorguya en alakalı seed entity'leri Neo4j'den bulur,
iteratif olarak KG üzerinde yayılım yapar ve
toplanan triple'lardan doğal dil özeti üretir.

Düzeltme:
  - _normalize_turkish(): Türkçe iyelik/hal ekleri temizlenir.
    Örnek: "galatasaray'ın" → "galatasaray", "fenerbahçe'de" → "fenerbahçe"
    Bu sayede keyword match Neo4j'de çok daha iyi sonuç verir.
  - Diğer düzeltmeler korundu (retry, rounds_completed, driver.close in finally).
"""

import os
import re
import json
import time
import functools
from typing import Optional
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    K_E, MAX_ROUNDS, MAX_ENTITIES_PER_ROUND, MAX_TRIPLES_PER_ENTITY,
    KEYWORD_CANDIDATE_LIMIT, EMBED_MODEL, GROQ_MODEL, GROQ_API_KEY,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
)


def retry_on_failure(max_retries: int = 3, delay: float = 2.0):
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


# ✔ DÜZELTME: Türkçe morfoloji temizleyici
# Türkçe'de sözcükler kesme işareti ile ekler alır: galatasaray'ın, fenerbahçe'de
# Neo4j'de bu biçimde entity bulunmaz, bu yüzden temizlenmesi gerekir.
_TR_SUFFIXES = re.compile(
    r"'(ın|in|un|ün|nın|nin|nun|nün|da|de|ta|te|dan|den|tan|ten|"
    r"a|e|ya|ye|na|ne|ı|i|u|ü|nı|ni|nu|nü|"
    r"yla|yle|la|le|ca|ce|ça|çe|ki|"
    r"dır|dir|dur|dür|tır|tir|tur|tür|"
    r"nın|nda|nde|ndaki|ndeki|ndan|nden)$",
    re.IGNORECASE
)

def _normalize_turkish(word: str) -> str:
    """
    Türkçe iyelik/hal eklerini kesme işareti öncesinde keser.
    'galatasaray'ın'  → 'galatasaray'
    'fenerbahçe'de'   → 'fenerbahçe'
    'direktörü'       → değişmez (kesme işareti yok)
    """
    cleaned = _TR_SUFFIXES.sub("", word)
    # Geriye kalan apostrophe'u da temizle
    cleaned = cleaned.rstrip("'")
    return cleaned if cleaned else word


class SpreadingActivation:

    def __init__(self):
        print("[Module1] Başlatılıyor...")
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.groq        = Groq(api_key=GROQ_API_KEY)
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        print("[Module1] ✔ Neo4j, Groq ve Embedding modeli hazır.")

    def _run_cypher(self, cypher: str, params: dict = None) -> list:
        with self.driver.session() as session:
            return [r.data() for r in session.run(cypher, params or {})]

    def _get_candidate_entities(self, query: str) -> list:
        """
        Sorgu kelimelerini Neo4j description'da arar (keyword match).
        ✔ Türkçe morfoloji temizleme eklendi: kelimeler normalize edilir.
        """
        if not query:
            return []

        raw_words = query.lower().split()
        # ✔ Her kelimeyi Türkçe normalize et, 3+ karakter olanları al
        keywords = []
        for w in raw_words:
            normalized = _normalize_turkish(w)
            if len(normalized) > 3:
                keywords.append(normalized)
                # Orijinal farklıysa onu da ekle (ikisi de denensin)
                if normalized != w and len(w) > 3:
                    keywords.append(w)

        # Tekrarları koru ama sıralamayı koruyarak benzersiz yap
        seen_kw, unique_kw = set(), []
        for kw in keywords[:10]:
            if kw not in seen_kw:
                seen_kw.add(kw)
                unique_kw.append(kw)

        results = []
        for kw in unique_kw[:6]:
            hits = self._run_cypher("""
                MATCH (e:Entity)
                WHERE toLower(e.description) CONTAINS $kw
                  AND e.description IS NOT NULL
                  AND size(e.description) > 10
                RETURN e.entityId AS id, e.description AS desc
                LIMIT $limit
            """, {"kw": kw, "limit": KEYWORD_CANDIDATE_LIMIT})
            results.extend(hits)

        seen, unique = set(), []
        for r in results:
            if r["id"] not in seen:
                seen.add(r["id"])
                unique.append(r)

        return unique

    def _get_one_hop_triples(self, entity_id: str) -> list:
        return self._run_cypher("""
            MATCH (e:Entity {entityId: $eid})-[r]->(target:Entity)
            WHERE target.description IS NOT NULL
            RETURN type(r)            AS relation,
                   target.entityId    AS target_id,
                   target.description AS target_desc
            LIMIT $limit
        """, {"eid": entity_id, "limit": MAX_TRIPLES_PER_ENTITY})

    def _get_entity_description(self, entity_id: str) -> str:
        results = self._run_cypher("""
            MATCH (e:Entity {entityId: $eid})
            RETURN e.description AS desc
        """, {"eid": entity_id})
        return results[0]["desc"] if results else ""

    def select_seed_entities(self, query: str) -> list:
        print(f"\n[Step 1] Seed entity seçiliyor (k_e={K_E})...")
        candidates = self._get_candidate_entities(query)

        if not candidates:
            print("[Step 1] ⚠ Aday entity bulunamadı.")
            return []

        print(f"  {len(candidates)} keyword-matched aday entity bulundu.")

        query_emb      = self.embed_model.encode([query])
        desc_texts     = [c["desc"] for c in candidates]
        candidate_embs = self.embed_model.encode(
            desc_texts, batch_size=64, show_progress_bar=False
        )

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

    @retry_on_failure(max_retries=3, delay=2.0)
    def _llm_select_relevant_triples(
        self, query: str, entity_desc: str, triples: list
    ) -> list:
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
                    selected = triples[:2]

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

    @retry_on_failure(max_retries=3, delay=2.0)
    def summarize_subgraph(self, query: str, activated_triples: list) -> str:
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

    def run(self, query: str) -> dict:
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
            "rounds_completed":  actual_rounds,
        }

        print("\n[Module 1] ✅ Tamamlandı.")
        return result

    def close(self):
        self.driver.close()


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
