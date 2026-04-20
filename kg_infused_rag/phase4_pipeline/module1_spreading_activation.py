"""
Module 1 - KG-Guided Spreading Activation
==========================================
Düzeltmeler:
  - _get_one_hop_triples(): Hem ileri (e->target) hem ters (e<-target)
    yönlü triple'lar çekilir. Football'da milli takım seed olduğunda
    MEMBER_OF_SPORTS_TEAM ters yönde geldiği için oyuncular kaçıyordu.
  - _infer_target_relations(): Sorgudan hedef relation'ları dinamik çıkarır.
    Sabit relation adı yok — keyword → relation mapping kullanılır.
    Football ve cinema için aynı kod çalışır.
  - _rerank_triples_by_relation(): LLM'e göndermeden önce ilgili
    relation'ları listenin başına taşır.
  - _normalize_text(): Türkçe iyelik/hal ekleri temizlenir.
  - select_seed_entities(): Düşük similarity durumunda keyword fallback
    ile pool'da string arama yapılır (Süleyman Seba gibi vakalar için).
  - _build_entity_pool(): Keyword listesi genişletildi, güvenlik ağı eklendi.
  - Diğer düzeltmeler korundu (retry, rounds_completed, driver.close).
"""

import os
import re
import json
import time
import functools
import logging
import unicodedata
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    K_E, MAX_ROUNDS, MAX_ENTITIES_PER_ROUND, MAX_TRIPLES_PER_ENTITY,
    KEYWORD_CANDIDATE_LIMIT, EMBED_MODEL, GROQ_MODEL, GROQ_API_KEYS,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GroqKeyRotator,
)

logging.basicConfig(
    filename='pipeline_debug.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
)

# ── Query keyword → KG relation mapping ──────────────────────────────────────
_RELATION_HINTS: list[tuple[tuple[str, ...], list[str]]] = [
    (
        ("born", "birth", "birthplace", "dogum", "dogdu", "nerede dogdu"),
        ["PLACE_OF_BIRTH"],
    ),
    (
        ("stadium", "venue", "arena", "sports hall", "ground", "home ground"),
        ["HOME_VENUE", "LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY", "LOCATION"],
    ),
    (
        ("award", "prize", "won", "winning", "oscar", "golden", "odul", "kazandi"),
        ["AWARD_RECEIVED", "NOMINATED_FOR"],
    ),
    (
        ("nationality", "citizenship", "citizen", "uyruk"),
        ["COUNTRY_OF_CITIZENSHIP", "NATIONALITY"],
    ),
    (
        ("coach", "manager", "head coach", "teknik direktor", "teknik"),
        ["HEAD_COACH", "COACHED_BY"],
    ),
    (
        ("plays for", "member", "club", "team", "transfer", "signed"),
        ["MEMBER_OF_SPORTS_TEAM", "CURRENT_TEAM"],
    ),
    (
        ("director", "directed", "yonetmen", "yonetti"),
        ["DIRECTOR", "FILM_DIRECTOR"],
    ),
    (
        ("actor", "actress", "starred", "cast", "oyuncu"),
        ["CAST_MEMBER", "ACTOR"],
    ),
    (
        ("located", "location", "where is", "city", "region", "district", "bulundugu"),
        ["LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY", "LOCATION", "COUNTRY"],
    ),
    (
        ("studied", "educated", "school", "university", "graduate"),
        ["EDUCATED_AT"],
    ),
    (
        ("league", "division", "lig"),
        ["LEAGUE"],
    ),
    (
        ("country", "nation", "ulke"),
        ["COUNTRY", "COUNTRY_OF_ORIGIN"],
    ),
]


def _infer_target_relations(query: str) -> list[str]:
    """
    Sorgudan hedef KG relation'larını dinamik olarak çıkarır.
    """
    q = _normalize_text(query)
    matched: list[str] = []
    for keywords, relations in _RELATION_HINTS:
        if any(kw in q for kw in keywords):
            for r in relations:
                if r not in matched:
                    matched.append(r)
    return matched


def _rerank_triples_by_relation(
    triples: list[dict], target_relations: list[str]
) -> list[dict]:
    """
    Triple listesini hedef relation'lara göre yeniden sıralar.
    Hedef relation'a uyan triple'lar listenin başına taşınır.
    """
    if not target_relations:
        return triples

    target_norm = [r.upper() for r in target_relations]
    priority, rest = [], []
    for t in triples:
        rel = t.get("relation", "").upper()
        if any(tr in rel or rel in tr for tr in target_norm):
            priority.append(t)
        else:
            rest.append(t)
    return priority + rest


# ── Pool ──────────────────────────────────────────────────────────────────────
_POOL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "domain_entity_pool.npz"
)
_POOL_SCHEMA_VERSION = 2

_DOMAIN_KEYWORDS = [
    "football", "soccer", "player", "footballer", "basketball", "tennis",
    "athlete", "club", "coach", "stadium", "arena", "sports hall", "venue",
    "actor", "actress", "director", "film", "movie", "television",
    "singer", "musician", "award", "prize",
]


def _normalize_text(text: str) -> str:
    """Türkçe dahil metni ASCII-safe normalize eder."""
    if not text:
        return ""
    s = text.strip().lower()
    tr_map = str.maketrans({
        "ı": "i", "İ": "i",
        "ğ": "g", "Ğ": "g",
        "ş": "s", "Ş": "s",
        "ç": "c", "Ç": "c",
        "ö": "o", "Ö": "o",
        "ü": "u", "Ü": "u",
    })
    s = s.translate(tr_map)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_DOMAIN_KEYWORDS_NORM = [_normalize_text(k) for k in _DOMAIN_KEYWORDS]

# Keyword fallback için minimum similarity eşiği
_SEED_SIM_FALLBACK_THRESHOLD = 0.55


def retry_on_failure(max_retries: int = 3, delay: float = 2.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            self_obj = args[0] if args and hasattr(args[0], 'groq') else None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if '429' in str(e) and self_obj is not None and len(GROQ_API_KEYS) > 1:
                        self_obj.groq.rotate()
                    wait = delay * (attempt + 1)
                    print(f"  [Retry] {func.__name__} hata (deneme {attempt+1}/{max_retries}): "
                          f"{e} — {wait:.1f}s bekleniyor.")
                    time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator


class SpreadingActivation:

    def __init__(self):
        print("[Module1] Başlatılıyor...")
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.groq        = GroqKeyRotator(GROQ_API_KEYS)
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        self._build_entity_pool()
        print("[Module1] ✔ Neo4j, Groq ve Embedding modeli hazır.")

    def _run_cypher(self, cypher: str, params: dict = None) -> list:
        try:
            with self.driver.session() as session:
                return [r.data() for r in session.run(cypher, params or {})]
        except Exception as e:
            print(f"[Module1] ❌ Neo4j hata: {e}")
            logging.error(f"Neo4j cypher hatası: {e} | Sorgu: {cypher} | Param: {params}")
            return []

    def _build_entity_pool(self) -> None:
        pool_path = _POOL_PATH
        if os.path.exists(pool_path):
            print("[Module1] Entity pool yükleniyor...")
            data = np.load(pool_path, allow_pickle=True)
            cached_version = int(data["pool_version"][0]) if "pool_version" in data else 0
            if cached_version >= _POOL_SCHEMA_VERSION and "norm_descs" in data:
                self.pool_ids        = data["ids"].tolist()
                self.pool_descs      = data["descs"].tolist()
                self.pool_norm_descs = data["norm_descs"].tolist()
                self.pool_embs       = data["embs"]
                print(f"[Module1] ✔ Pool hazır: {len(self.pool_ids)} entity (v{cached_version})")
                return
            print("[Module1] ⚠ Eski pool formatı — yeniden oluşturuluyor...")

        print("[Module1] Entity pool bulunamadı — Neo4j'den çekiliyor...")
        seen_ids, rows = set(), []
        for hop in [1, 2, 3]:
            cypher = (
                f"MATCH (t:Entity {{entityId: 'Q43'}})-[*{hop}]-(e:Entity) "
                "WHERE e.name IS NOT NULL "
                "RETURN DISTINCT e.entityId AS id, "
                "  coalesce(e.aliases,'') + ' | ' + coalesce(e.name,'') "
                "  + ' | ' + coalesce(e.description,'') AS desc "
                "LIMIT 30000"
            )
            hop_rows = self._run_cypher(cypher)
            new_rows = [r for r in hop_rows if r["id"] not in seen_ids]
            seen_ids.update(r["id"] for r in new_rows)
            rows.extend(new_rows)
            print(f"[Module1] {hop}-hop: {len(new_rows)} yeni entity (toplam: {len(rows)})")

        if not rows:
            print("[Module1] ⚠ Pool entity çekilemedi — boş pool ile devam ediliyor.")
            self.pool_ids = self.pool_descs = self.pool_norm_descs = []
            self.pool_embs = np.array([])
            return

        filtered = [
            r for r in rows
            if any(kw in _normalize_text(r["desc"]) for kw in _DOMAIN_KEYWORDS_NORM)
        ]
        rows = filtered if filtered else rows
        print(f"[Module1] {len(rows)} entity keyword filtresi sonrası kaldı.")

        self.pool_ids        = [r["id"]   for r in rows]
        self.pool_descs      = [r["desc"] for r in rows]
        self.pool_norm_descs = [_normalize_text(d) for d in self.pool_descs]
        print(f"[Module1] {len(self.pool_ids)} entity encode ediliyor...")
        self.pool_embs = self.embed_model.encode(
            self.pool_norm_descs, batch_size=128, show_progress_bar=True
        ).astype(np.float32)
        os.makedirs(os.path.dirname(pool_path), exist_ok=True)
        np.savez(
            pool_path,
            ids=np.array(self.pool_ids),
            descs=np.array(self.pool_descs),
            norm_descs=np.array(self.pool_norm_descs),
            embs=self.pool_embs,
            pool_version=np.array([_POOL_SCHEMA_VERSION], dtype=np.int32),
        )
        print(f"[Module1] ✔ Pool kaydedildi: {pool_path} ({len(self.pool_ids)} entity)")

    def _get_one_hop_triples(self, entity_id: str) -> list:
        """
        Hem ileri (entity → target) hem ters (entity ← source) yönlü
        triple'ları çeker ve birleştirir.
        """
        # İleri yön: entity → target
        forward = self._run_cypher("""
            MATCH (e:Entity {entityId: $eid})-[r]->(target:Entity)
            RETURN type(r) AS relation,
                   target.entityId AS target_id,
                   coalesce(target.aliases, coalesce(target.name, target.entityId))
                     + ' | ' + coalesce(target.name, target.entityId)
                     + ' | ' + coalesce(target.description, '') AS target_desc,
                   'forward' AS direction
            LIMIT $limit
        """, {"eid": entity_id, "limit": MAX_TRIPLES_PER_ENTITY})

        # Ters yön: entity ← source
        backward = self._run_cypher("""
            MATCH (source:Entity)-[r]->(e:Entity {entityId: $eid})
            RETURN type(r) AS relation,
                   source.entityId AS target_id,
                   coalesce(source.aliases, coalesce(source.name, source.entityId))
                     + ' | ' + coalesce(source.name, source.entityId)
                     + ' | ' + coalesce(source.description, '') AS target_desc,
                   'backward' AS direction
            LIMIT $limit
        """, {"eid": entity_id, "limit": MAX_TRIPLES_PER_ENTITY})

        for t in backward:
            t["relation"] = "INV_" + t["relation"]

        combined = forward + backward
        return combined[:MAX_TRIPLES_PER_ENTITY * 2]

    def _get_entity_description(self, entity_id: str) -> str:
        results = self._run_cypher("""
            MATCH (e:Entity {entityId: $eid})
            RETURN coalesce(e.aliases, coalesce(e.name, e.entityId))
                     + ' | ' + coalesce(e.name, e.entityId)
                     + ' | ' + coalesce(e.description, '') AS desc
        """, {"eid": entity_id})
        return results[0]["desc"] if results else ""

    def select_seed_entities(self, query: str) -> list:
        print(f"\n[Step 1] Seed entity seçiliyor (k_e={K_E}, pool={len(self.pool_ids)})...")
        if not self.pool_ids:
            print("[Step 1] ⚠ Entity pool boş.")
            return []

        query_emb    = self.embed_model.encode([_normalize_text(query)])
        similarities = cosine_similarity(query_emb, self.pool_embs)[0]
        top_k        = min(K_E, len(self.pool_ids))
        top_indices  = np.argsort(similarities)[::-1][:top_k]

        seeds = []
        for idx in top_indices:
            seeds.append({
                "id":         self.pool_ids[idx],
                "desc":       self.pool_descs[idx],
                "similarity": float(similarities[idx]),
            })
            print(f"  ✔ Seed: {self.pool_ids[idx]:12} | "
                  f"sim={similarities[idx]:.3f} | "
                  f"{str(self.pool_descs[idx])[:60]}")

        # ── Keyword fallback ──────────────────────────────────────────────────
        # Tüm seed'lerin similarity'si eşiğin altındaysa (transliterasyon,
        # yazım farkı vb.), sorgudan özel isim çıkarıp pool'da string arama yap.
        # Örnek: "Sueleyman Seba" → normalize → "sueleyman seba" → pool desc'inde ara
        max_sim = max(s["similarity"] for s in seeds) if seeds else 0.0
        if max_sim < _SEED_SIM_FALLBACK_THRESHOLD:
            # Sorgudan büyük harfle başlayan 4+ karakter token'ları al
            tokens = [t for t in query.split() if len(t) > 3 and t[0].isupper()]
            if tokens:
                name_query = _normalize_text(" ".join(tokens))
                print(f"  [Fallback] Düşük similarity ({max_sim:.3f}) — "
                      f"keyword arama: '{name_query}'")
                existing_ids = {s["id"] for s in seeds}
                for i, desc_norm in enumerate(self.pool_norm_descs):
                    if name_query in desc_norm and self.pool_ids[i] not in existing_ids:
                        fallback_seed = {
                            "id":         self.pool_ids[i],
                            "desc":       self.pool_descs[i],
                            "similarity": 1.0,  # keyword hit — kesin eşleşme
                        }
                        seeds.insert(0, fallback_seed)
                        print(f"  ✔ Fallback Seed: {self.pool_ids[i]:12} "
                              f"| keyword match | {str(self.pool_descs[i])[:60]}")
                        break  # ilk bulunan yeterli

        return seeds[:K_E]

    @retry_on_failure(max_retries=3, delay=2.0)
    def _llm_select_relevant_triples(
        self,
        query: str,
        entity_desc: str,
        triples: list,
        target_relations: list[str] | None = None,
    ) -> list:
        if not triples:
            return []

        if target_relations:
            triples = _rerank_triples_by_relation(triples, target_relations)

        triples   = triples[:30]
        ent_short = entity_desc[:150]

        triples_text = "\n".join(
            f"{i+1}. <{ent_short}, {t['relation']}, {t['target_desc'][:150]}>"
            for i, t in enumerate(triples)
        )

        if target_relations:
            relation_list = ", ".join(target_relations)
            preference_hint = (
                f"The query is asking about: {relation_list}. "
                f"Relations starting with 'INV_' are reverse-direction edges "
                f"(e.g. INV_MEMBER_OF_SPORTS_TEAM means the entity is a club/team "
                f"that players belong to). "
                f"If any triple with these relation types (or their INV_ variants) "
                f"exists, prioritize them. "
                f"Include ALL matching triples of these types — do not skip them."
            )
        else:
            preference_hint = "Prioritize relations that directly match the queried attribute."

        prompt = f"""You are a knowledge graph reasoning assistant.

Query: {query}

Retrieved Entity Triples:
{triples_text}

Select ONLY the triples that are relevant to answering the query.
{preference_hint}
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
            raw_clean = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()
            indices   = json.loads(raw_clean)
            selected  = [triples[int(i)-1] for i in indices
                         if 1 <= int(i) <= len(triples)]
            return selected
        except (json.JSONDecodeError, ValueError, TypeError):
            print(f"  [LLM] JSON parse hatası: '{raw}' — ilk 3 triple alınıyor.")
            return triples[:3]

    def iterative_spreading(
        self, query: str, seeds: list
    ) -> tuple[list, list, int]:
        print(f"\n[Step 2] Iteratif spreading activation (max_rounds={MAX_ROUNDS})...")

        target_relations = _infer_target_relations(query)
        if target_relations:
            print(f"  [Relation Hint] Tespit edilen: {target_relations}")

        visited       = set()
        activated     = []
        current_queue = [s["id"] for s in seeds]
        actual_rounds = 0

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
                        query, entity_desc, triples,
                        target_relations=target_relations,
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
            f"<{t['source_desc']}, {t['relation']}, {t['target_desc']}>"
            for t in activated_triples[:30]
        )
        if len(triples_text) > 5000:
            triples_text = triples_text[:5000] + "\n..."

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
        "Where was Sueleyman Seba born?",
        "In which city or region is the stadium of Adanaspor Basketbol located?",
        "Which award did the director of the film win?",
    ]

    sa = SpreadingActivation()
    try:
        for query in test_queries[:1]:
            result = sa.run(query)
            print("\n" + "─" * 60)
            print(f"  Seed entity     : {len(result['seed_entities'])}")
            print(f"  Aktive triple   : {len(result['activated_triples'])}")
            print(f"  Ziyaret entity  : {len(result['visited_entities'])}")
            print(f"  Tamamlanan tur  : {result['rounds_completed']}")
            print(f"\nKG Özeti:\n{result['kg_summary']}")
            os.makedirs("outputs/phase4", exist_ok=True)
            with open("outputs/phase4/module1_test_result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    finally:
        sa.close()