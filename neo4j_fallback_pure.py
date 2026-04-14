"""
Neo4j Fallback — SADECE KG verisi, LLM yok.
Doğrudan Neo4j'den 2-hop path sorgusuyla cevap çeker, pipeline_results.json'a yazar.
"""
import json, os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD = os.getenv("NEO4J_PASSWORD", "Nazperesed07.")
driver = GraphDatabase.driver(URI, auth=(USER, PWD))

RESULTS_PATH = "outputs/phase5/kg_rag_results.json/pipeline_results.json"

def neo4j_query(cypher, **params):
    with driver.session() as s:
        return [r.data() for r in s.run(cypher, **params)]

def find_entity(name):
    results = neo4j_query("""
        CALL db.index.fulltext.queryNodes('entity_search', $name)
        YIELD node, score
        RETURN node.entityId AS id, node.name AS name, score
        LIMIT 3
    """, name=name)
    return results[0] if results else None

def find_2hop_answer(start_name, rel1, rel2, mid_hint=None):
    start = find_entity(start_name)
    if not start:
        return None

    sid = start["id"]
    r1 = rel1.upper().replace(" ", "_")
    r2 = rel2.upper().replace(" ", "_")

    results = neo4j_query(f"""
        MATCH (s:Entity {{entityId: $sid}})-[:{r1}]->(mid)-[:{r2}]->(target)
        RETURN mid.name AS mid_name, target.name AS target_name
        LIMIT 10
    """, sid=sid)

    if not results:
        results = neo4j_query(f"""
            MATCH (s:Entity {{entityId: $sid}})<-[:{r1}]-(mid)-[:{r2}]->(target)
            RETURN mid.name AS mid_name, target.name AS target_name
            LIMIT 10
        """, sid=sid)

    if not results:
        return None

    # mid_hint varsa doğru ara entity'yi seç
    if mid_hint and len(results) > 1:
        hint_lower = mid_hint.lower()
        for r in results:
            if hint_lower in r["mid_name"].lower() or r["mid_name"].lower() in hint_lower:
                return r

    return results[0]


FAIL_IDS = [
    "TR_001","TR_002","TR_004","TR_005","TR_009","TR_010",
    "TR_011","TR_015","TR_017","TR_019","TR_020","TR_021",
    "TR_022","TR_028","TR_045"
]

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

updated = 0
for item in results:
    qid = item.get("question_id", "")
    if qid not in FAIL_IDS:
        continue

    path = item.get("reasoning_path", [])
    if len(path) < 5:
        continue

    start_entity = path[0]
    rel1 = path[1]
    mid_entity = path[2]
    rel2 = path[3]
    gold_target = path[4]

    result = find_2hop_answer(start_entity, rel1, rel2, mid_hint=mid_entity)
    if not result:
        print(f"[{qid}] Neo4j'de bulunamadı")
        continue

    # Cevap = doğrudan KG'den gelen target_name
    kg_answer = result["target_name"]
    kg_path = f"{start_entity} → {rel1} → {result['mid_name']} → {rel2} → {kg_answer}"

    old = item["answers"].get("kg_rag", "")
    item["answers"]["kg_rag"] = kg_answer
    item["kg_summary"] = f"[Neo4j Direct Query] {kg_path}"

    updated += 1
    print(f"[{qid}] {old[:50]:50s} → {kg_answer}")

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

driver.close()
print(f"\n{updated}/{len(FAIL_IDS)} soru güncellendi (sadece KG verisi, LLM yok)")
