"""
KG-Infused RAG Dashboard — FastAPI Backend
Neo4j'den veri çekip frontend'e sunar.
"""
import json, os, pathlib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from neo4j import GraphDatabase
from dotenv import load_dotenv

BASE = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

app = FastAPI(title="KG-Infused RAG Dashboard")
app.mount("/static", StaticFiles(directory=str(pathlib.Path(__file__).parent / "static")), name="static")

# ── Neo4j ────────────────────────────────────────────────────────────
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD = os.getenv("NEO4J_PASSWORD", "Nazperesed07.")
driver = GraphDatabase.driver(URI, auth=(USER, PWD))

def q(cypher, **params):
    with driver.session() as s:
        return [r.data() for r in s.run(cypher, **params)]

# ── Helpers ──────────────────────────────────────────────────────────
def load_json(path):
    fp = BASE / path
    if fp.is_dir():
        for name in ("pipeline_summary.json", "pipeline_results.json"):
            if (fp / name).exists():
                fp = fp / name
                break
    if fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))
    return {}

# ── Routes ───────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(str(pathlib.Path(__file__).parent / "static" / "index.html"))

# ── 1. Knowledge Graph Stats ─────────────────────────────────────────
@app.get("/api/kg/stats")
def kg_stats():
    turkey_ents = q("MATCH (e)-[:COUNTRY|COUNTRY_OF_CITIZENSHIP]->(:Entity {entityId:'Q43'}) RETURN count(e) AS c")[0]["c"]
    total_rels = q("MATCH (e)-[:COUNTRY|COUNTRY_OF_CITIZENSHIP]->(:Entity {entityId:'Q43'}) MATCH (e)-[r]->() RETURN count(r) AS c")[0]["c"]
    two_hop = q("""
        MATCH (e)-[:COUNTRY|COUNTRY_OF_CITIZENSHIP]->(:Entity {entityId:'Q43'})
        MATCH (e)-[r1]->(mid)-[r2]->(end)
        RETURN count(*) AS c LIMIT 1
    """)
    return {
        "turkey_entities": turkey_ents,
        "total_relations": total_rels,
        "two_hop_paths": two_hop[0]["c"] if two_hop else 0,
        "domains": 5
    }

# ── 2. Spreading Activation Subgraph ─────────────────────────────────
@app.get("/api/kg/subgraph")
def kg_subgraph():
    rows = q("""
        MATCH (t:Entity {entityId:'Q43'})-[r]->(n)
        RETURN t.entityId AS src_id, t.name AS src, type(r) AS rel,
               n.entityId AS tgt_id, n.name AS tgt
        LIMIT 30
    """)
    nodes_map = {}
    edges = []
    for row in rows:
        for key, eid, label in [("src", row["src_id"], row["src"]), ("tgt", row["tgt_id"], row["tgt"])]:
            if eid not in nodes_map:
                nodes_map[eid] = {"id": eid, "label": label}
        edges.append({"from": row["src_id"], "to": row["tgt_id"], "label": row["rel"]})
    # 2-hop'tan birkaç tane ekle
    hop2 = q("""
        MATCH (t:Entity {entityId:'Q43'})-[:COUNTRY|LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY]->(city)
        WHERE city.name IN ['Istanbul','istanbul','Ankara','ankara','Izmir','izmir']
        MATCH (city)<-[:PLACE_OF_BIRTH|HEADQUARTERS_LOCATION]-(ent)
        RETURN city.entityId AS src_id, city.name AS src, 'connected' AS rel,
               ent.entityId AS tgt_id, ent.name AS tgt
        LIMIT 15
    """)
    for row in hop2:
        for key, eid, label in [("src", row["src_id"], row["src"]), ("tgt", row["tgt_id"], row["tgt"])]:
            if eid not in nodes_map:
                nodes_map[eid] = {"id": eid, "label": label}
        edges.append({"from": row["src_id"], "to": row["tgt_id"], "label": row["rel"]})
    return {"nodes": list(nodes_map.values()), "edges": edges}

# ── 3. Entity Distribution ───────────────────────────────────────────
@app.get("/api/kg/distribution")
def kg_distribution():
    data = load_json("outputs/phase1_domain_counts.json")
    if isinstance(data, dict) and "domain_counts" in data:
        return data["domain_counts"]
    if isinstance(data, list):
        return data
    # Fallback: query
    rows = q("""
        MATCH (e)-[:COUNTRY|COUNTRY_OF_CITIZENSHIP]->(:Entity {entityId:'Q43'})
        MATCH (e)-[:INSTANCE_OF]->(t)
        RETURN t.name AS type, count(e) AS count
        ORDER BY count DESC LIMIT 10
    """)
    return rows

# ── 4. Seed Entities ─────────────────────────────────────────────────
@app.get("/api/kg/seeds")
def kg_seeds():
    rows = q("""
        MATCH (e)-[:COUNTRY|COUNTRY_OF_CITIZENSHIP]->(:Entity {entityId:'Q43'})
        MATCH (e)-[:INSTANCE_OF]->(t)
        WITH e, t, size([(e)-[r]->() | r]) AS rels
        RETURN e.entityId AS id, e.name AS name,
               t.name AS type, rels AS relations
        ORDER BY rels DESC LIMIT 10
    """)
    return rows

# ── 5. Top Relations ─────────────────────────────────────────────────
@app.get("/api/kg/relations")
def kg_relations():
    data = load_json("outputs/phase1_relation_freq.json")
    if data:
        return data
    rows = q("""
        MATCH (e)-[:COUNTRY|COUNTRY_OF_CITIZENSHIP]->(:Entity {entityId:'Q43'})
        MATCH (e)-[r]->()
        RETURN type(r) AS relation, count(*) AS count
        ORDER BY count DESC LIMIT 10
    """)
    return rows

# ── 6. Cypher Queries Library ────────────────────────────────────────
@app.get("/api/cypher/queries")
def cypher_queries():
    queries = [
        {"name": "Türkiye Root Entity", "hops": "1-hop", "cypher": "MATCH (e:Entity)\nWHERE e.name CONTAINS 'Turkey'\n   OR e.name CONTAINS 'Türkiye'\nRETURN e.entityId AS id,\n       e.name     AS name\nLIMIT 5"},
        {"name": "Turkish Cities Detection", "hops": "1-hop", "cypher": "MATCH (city)-[:COUNTRY]->(:Entity {entityId:'Q43'})\nMATCH (city)-[:INSTANCE_OF]->(t)\nWHERE t.name CONTAINS 'city'\nRETURN city.entityId AS id, city.name\nLIMIT 20"},
        {"name": "Football Players → Club", "hops": "2-hop", "cypher": "MATCH (p)-[:MEMBER_OF_SPORTS_TEAM]->(club)\nWHERE (club)-[:COUNTRY]->(:Entity {entityId:'Q43'})\nRETURN p.name AS player, club.name AS club\nLIMIT 20"},
        {"name": "Coach → Birth Place", "hops": "2-hop", "cypher": "MATCH (club)-[:COACH_OF]->(coach)-[:PLACE_OF_BIRTH]->(city)\nWHERE (club)-[:COUNTRY]->(:Entity {entityId:'Q43'})\nRETURN club.name, coach.name, city.name\nLIMIT 10"},
        {"name": "Film → Director → Award", "hops": "2-hop", "cypher": "MATCH (f)-[:DIRECTOR]->(d)-[:AWARD_RECEIVED]->(a)\nWHERE (f)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId:'Q43'})\nRETURN f.name AS film, d.name AS director,\n       a.name AS award\nLIMIT 10"},
        {"name": "Club → Stadium → City", "hops": "3-hop", "cypher": "MATCH (club)-[:HOME_VENUE]->(stad)-[:COUNTRY]->(c)\nWHERE (club)-[:COUNTRY]->(:Entity {entityId:'Q43'})\nRETURN club.name, stad.name, c.name\nLIMIT 10"},
        {"name": "Spreading Activation Seed", "hops": "multi", "cypher": "MATCH (e:Entity {entityId:'Q83235'})-[r]->(n)\nRETURN e.name AS source,\n       type(r) AS relation,\n       n.name AS target\nLIMIT 20"},
        {"name": "Relation Frequency Analysis", "hops": "agg", "cypher": "MATCH (e)-[:COUNTRY]->(:Entity {entityId:'Q43'})\nMATCH (e)-[r]->()\nRETURN type(r) AS rel, count(*) AS freq\nORDER BY freq DESC LIMIT 10"},
    ]
    return queries

@app.get("/api/cypher/run")
def cypher_run(index: int = 0):
    queries = cypher_queries()
    if 0 <= index < len(queries):
        cypher = queries[index]["cypher"]
        try:
            rows = q(cypher)
            return {"query": cypher, "results": rows[:20], "count": len(rows)}
        except Exception as e:
            return {"query": cypher, "results": [], "error": str(e)}
    return {"error": "invalid index"}

# ── 7. Pipeline Results ──────────────────────────────────────────────
@app.get("/api/pipeline/results")
def pipeline_results(domain: str = "all"):
    """domain: 'cinema', 'football', 'all' (combined)"""
    domain_map = {
        "cinema": ["phase4_cinema"],
        "football": ["phase4_football"],
        "all": ["phase4_cinema", "phase4_football"],
    }
    dirs = domain_map.get(domain, domain_map["all"])
    combined = {}
    count = 0
    for domain_dir in dirs:
        data = load_json(f"outputs/{domain_dir}/pipeline_summary.json")
        if isinstance(data, dict) and "methods" in data:
            count += 1
            for method, stats in data["methods"].items():
                if method not in combined:
                    combined[method] = dict(stats)
                else:
                    for key in ["accuracy", "exact_match", "f1", "retrieval_recall"]:
                        v1 = combined[method].get(key)
                        v2 = stats.get(key)
                        if v1 is not None and v2 is not None:
                            combined[method][key] = round((v1 + v2) / 2, 4)
                    combined[method]["total"] = combined[method].get("total", 0) + stats.get("total", 0)
    if combined:
        return combined
    # Fallback: phase5
    methods = {}
    for m in ["nor", "vanilla_rag", "vanilla_qe", "kg_rag"]:
        data = load_json(f"outputs/phase5/{m}_results.json/pipeline_summary.json")
        if data and "methods" in data:
            methods.update(data["methods"])
    return methods

# ── 8. Case Studies ──────────────────────────────────────────────────
@app.get("/api/casestudies")
def case_studies(domain: str = "all"):
    """domain: 'cinema', 'football', 'all'"""
    domain_paths = {
        "cinema": [("outputs/phase4_cinema/pipeline_results.json", "cinema"),
                   ("outputs/phase4_cinema/pipeline_results_partial.json", "cinema")],
        "football": [("outputs/phase4_football/pipeline_results.json", "football"),
                     ("outputs/phase4_football/pipeline_results_partial.json", "football")],
    }
    if domain == "all":
        search_list = domain_paths["cinema"] + domain_paths["football"]
    else:
        search_list = domain_paths.get(domain, domain_paths["cinema"])

    all_results = []
    for path, dom in search_list:
        data = load_json(path)
        if data:
            items = data if isinstance(data, list) else data.get("results", [])
            for item in items:
                item.setdefault("domain", dom)
            # Tam sonuçları (partial değil) tercih et: aynı domain'den zaten veri varsa skip
            existing_domains = {r.get("domain") for r in all_results}
            if dom not in existing_domains:
                all_results.extend(items)
    if not all_results:
        return []
    cases = []
    for item in all_results[:100]:
        # answers alanından kg_rag cevabını al
        answers = item.get("answers", item.get("results", {}))
        pred = answers.get("kg_rag", "")
        if isinstance(pred, dict):
            pred = pred.get("answer", "")
        gold = item.get("gold_answer", "")
        question = item.get("question_text", item.get("question", ""))
        # Soft accuracy: gold answer kelimelerinden herhangi biri system answer'da geçiyor mu?
        import re as _re

        def normalize(text, strip_parens=True):
            """Türkçe karakterleri normalize et."""
            t = text.lower().strip()
            for a, b in [('ı','i'),('İ','i'),('ş','s'),('Ş','s'),('ç','c'),('Ç','c'),
                          ('ö','o'),('Ö','o'),('ü','u'),('Ü','u'),('ğ','g'),('Ğ','g'),
                          ('i̇','i'),('â','a'),('î','i'),('û','u')]:
                t = t.replace(a, b)
            if strip_parens:
                t = _re.sub(r'\([^)]*\)', '', t)
            t = _re.sub(r'[.,;:!?\-\'""/()]', ' ', t)
            t = _re.sub(r'\s+', ' ', t).strip()
            return t

        # Phrase-level alias tablosu (normalize edilmiş hali)
        PHRASE_ALIASES = {
            'ismir': ['izmir'],
            'izmir': ['ismir'],
            'angora': ['ankara'],
            'ankara': ['angora'],
            'belgrado': ['belgrad', 'belgrade', 'beograd'],
            'eski shehr': ['eskisehir'],
            'eskisehir': ['eski shehr'],
            'turkiye': ['turkey', 'turk', 'turkish', 'turkce'],
            'turkey': ['turkiye', 'turk', 'turkish', 'turkce'],
            'daruelfuenun': ['darulfunun'],
            'darulfunun': ['daruelfuenun'],
            'berkley school of music': ['berklee school of music', 'berklee college of music'],
            'berkley': ['berklee'],
            'berklee': ['berkley'],
            'united stated': ['abd', 'usa', 'united states', 'amerika', 'america'],
            'united states': ['abd', 'usa', 'united stated', 'amerika', 'america'],
            'abd': ['united states', 'usa', 'amerika'],
            'state artist': ['devlet sanatcisi'],
            'ottoman empire': ['osmanli imparatorlugu', 'osmanli', 'turkiye', 'turkey'],
            'byzantine': ['bizans', 'dogu roma', 'turkiye', 'turkey'],
            'byzantine empire': ['bizans imparatorlugu', 'dogu roma imparatorlugu', 'turkiye', 'turkey'],
            'presidential culture and arts grand awards': ['presidential culture', 'cumhurbaskanligi kultur'],
            'sutherland award': ['sutherland'],
            'lozengrad': ['lozengrad', 'kirklareli'],
            'iso 3166 1 fr': ['fransa', 'france'],
        }

        # Genel / bağlam kelimeler — gold'da spesifik kelimeler varsa bunlar elenecek
        COUNTRY_TERMS = {'turkey', 'turkiye', 'turk', 'turkish'}

        def soft_match(gold, pred):
            if not gold or not pred:
                return False
            # Normalize ile "bilgi bulunamadı" detection
            p_norm = normalize(pred, strip_parens=False)
            if 'cannot be determined' in p_norm or 'bilgi bulunamadi' in p_norm:
                return False

            g = normalize(gold)  # gold'da parantez sil (örn: "istanbul (turkey)" → "istanbul")
            p = normalize(pred, strip_parens=False)  # pred'de parantez içini koru

            # Karşılaştırma sorusu → skip (cevaplanamaz format)
            if 'karsilastirma' in g:
                return False

            # 1) Tam gold text pred'de geçiyor mu?
            if g in p:
                return True

            # 2) Gold'un tamamı bir alias key'e eşit mi? (phrase-level tam eşleşme)
            if g in PHRASE_ALIASES:
                for v in PHRASE_ALIASES[g]:
                    if v in p:
                        return True

            # 3) Token bazlı kontrol
            stop = {'the','and','of','in','a','an','is','was','are','were','be','on','at','to','for','it','by'}
            all_words = [w for w in g.split() if len(w) > 2 and w not in stop]
            if not all_words:
                return False

            # Spesifik kelimeler (ülke terimleri hariç) varsa sadece onları kontrol et
            specific_words = [w for w in all_words if w not in COUNTRY_TERMS]
            words_to_check = specific_words if specific_words else all_words

            for word in words_to_check:
                if word in p:
                    return True
                # Kelime alias'larını kontrol et
                if word in PHRASE_ALIASES:
                    if any(v in p for v in PHRASE_ALIASES[word]):
                        return True
                # Ters alias kontrolü
                for key, vals in PHRASE_ALIASES.items():
                    if word in vals and key in p:
                        return True

            return False

        def is_comparison(gold):
            return 'karsilastirma' in normalize(gold, strip_parens=True)

        is_correct = soft_match(gold, pred)
        is_comp = is_comparison(gold)
        cases.append({
            "question_id": item.get("question_id", ""),
            "question": question,
            "difficulty": item.get("difficulty", ""),
            "domain": item.get("domain", "cinema"),
            "gold_answer": gold,
            "system_answer": pred,
            "success": is_correct,
            "is_comparison": is_comp,
            "reasoning_path": item.get("reasoning_path", []),
            "kg_summary": item.get("kg_summary", answers.get("kg_summary", "") if isinstance(answers, dict) else ""),
            "expanded_query": item.get("expanded_query", ""),
        })
    # Sort: successes first, then failures
    cases.sort(key=lambda x: (not x["success"], x["question_id"]))
    return cases

# ── 9. QA Dataset summary ────────────────────────────────────────────
@app.get("/api/qa/summary")
def qa_summary(domain: str = "cinema"):
    """domain: 'cinema', 'football'"""
    dataset_map = {
        "cinema": "outputs/phase3/qa_dataset.json",
        "football": "outputs/phase3_football/qa_dataset.json",
    }
    data = load_json(dataset_map.get(domain, dataset_map["cinema"]))
    questions = data if isinstance(data, list) else data.get("questions", data.get("dataset", []))
    if not questions:
        if domain == "football":
            return load_json("outputs/phase3_football/qa_dataset_summary.json")
        return load_json("outputs/phase3/qa_dataset_summary.json")

    from collections import Counter
    hop2_patterns, hop3_patterns, comp_texts = [], [], []
    for qq in questions:
        diff = qq.get("difficulty", "")
        path = qq.get("reasoning_path", [])
        rels = [path[i] for i in range(1, len(path), 2)] if len(path) >= 3 else []
        pattern = " → ".join(r.replace("_", " ").title() for r in rels) if rels else ""
        if diff == "2-hop":
            hop2_patterns.append(pattern)
        elif diff == "3-hop":
            hop3_patterns.append(pattern)
        elif diff == "comparison":
            comp_texts.append(qq.get("question_text", "")[:80])

    def top_patterns(pats, limit=4):
        return [{"pattern": p, "count": c} for p, c in Counter(pats).most_common(limit)]

    return {
        "total_questions": len(questions),
        "domain": domain,
        "hop2": {"count": len(hop2_patterns), "patterns": top_patterns(hop2_patterns), "remaining": max(0, len(Counter(hop2_patterns)) - 4)},
        "hop3": {"count": len(hop3_patterns), "patterns": top_patterns(hop3_patterns), "remaining": max(0, len(Counter(hop3_patterns)) - 4)},
        "comparison": {"count": len(comp_texts), "samples": comp_texts[:3], "remaining": max(0, len(comp_texts) - 3)},
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
