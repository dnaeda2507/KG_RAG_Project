"""
Başarısız soruları doğrudan Neo4j'den sorgulayarak cevap bulmayı deneyen fallback script.
Pipeline Wikipedia'dan passage bulamadığında, KG verisi zaten Neo4j'de var.
"""
import json, os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD = os.getenv("NEO4J_PASSWORD", "Nazperesed07.")
driver = GraphDatabase.driver(URI, auth=(USER, PWD))

def q(cypher, **params):
    with driver.session() as s:
        return [r.data() for r in s.run(cypher, **params)]

# Sorular ve reasoning path'leri
RESULTS_PATH = "outputs/phase5/kg_rag_results.json/pipeline_results.json"
with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

# Kalan FAIL ID'leri
FAIL_IDS = [
    "TR_001","TR_002","TR_004","TR_005","TR_009","TR_010",
    "TR_011","TR_015","TR_017","TR_019","TR_020","TR_021",
    "TR_022","TR_028","TR_045"
]

print("=" * 80)
print("NEO4J FALLBACK: Başarısız sorular için KG'den doğrudan cevap aranıyor")
print("=" * 80)

for item in results:
    qid = item.get("question_id", "")
    if qid not in FAIL_IDS:
        continue
    
    path = item.get("reasoning_path", [])
    gold = item.get("gold_answer", "")
    question = item.get("question_text", "")
    
    print(f"\n--- {qid} ---")
    print(f"  Q: {question[:80]}")
    print(f"  Gold: {gold}")
    print(f"  Path: {' → '.join(path)}")
    
    if len(path) < 3:
        print("  [SKIP] Reasoning path too short")
        continue
    
    # Path'ten entity ve relation'ları çıkar
    # Tipik path: [entity1, REL1, entity2, REL2, entity3]
    start_entity = path[0]
    
    # Fulltext search ile başlangıç entity'sini bul
    try:
        search_results = q("""
            CALL db.index.fulltext.queryNodes('entity_search', $name)
            YIELD node, score
            RETURN node.entityId AS id, node.name AS name, score
            LIMIT 5
        """, name=start_entity)
        
        if search_results:
            print(f"  Neo4j matches for '{start_entity}':")
            for r in search_results:
                print(f"    {r['id']}: {r['name']} (score={r['score']:.2f})")
            
            # İlk eşleşme ile path'i takip et
            start_id = search_results[0]["id"]
            
            # 2-hop: entity → REL → middle → REL → target
            if len(path) >= 5:
                rel1 = path[1].upper().replace(" ", "_")
                rel2 = path[3].upper().replace(" ", "_")
                
                hop2_results = q(f"""
                    MATCH (s:Entity {{entityId: $sid}})-[:{rel1}]->(mid)-[:{rel2}]->(target)
                    RETURN mid.name AS mid_name, target.name AS target_name, 
                           target.entityId AS target_id
                    LIMIT 5
                """, sid=start_id)
                
                if hop2_results:
                    print(f"  2-hop results ({rel1} → {rel2}):")
                    for r in hop2_results:
                        print(f"    {r['mid_name']} → {r['target_name']}")
                else:
                    # Ters yönde dene
                    hop2_results = q(f"""
                        MATCH (s:Entity {{entityId: $sid}})<-[:{rel1}]-(mid)-[:{rel2}]->(target)
                        RETURN mid.name AS mid_name, target.name AS target_name,
                               target.entityId AS target_id
                        LIMIT 5
                    """, sid=start_id)
                    if hop2_results:
                        print(f"  2-hop reverse results ({rel1} ← mid → {rel2}):")
                        for r in hop2_results:
                            print(f"    {r['mid_name']} → {r['target_name']}")
                    else:
                        # Relation isimleri farklı olabilir, genel arama yap
                        general_results = q("""
                            MATCH (s:Entity {entityId: $sid})-[r1]->(mid)-[r2]->(target)
                            RETURN type(r1) AS rel1, mid.name AS mid, 
                                   type(r2) AS rel2, target.name AS target
                            LIMIT 10
                        """, sid=start_id)
                        if general_results:
                            print(f"  General 2-hop from {search_results[0]['name']}:")
                            for r in general_results:
                                print(f"    --{r['rel1']}--> {r['mid']} --{r['rel2']}--> {r['target']}")
                        else:
                            print(f"  No 2-hop paths found from {start_id}")
        else:
            print(f"  [NO MATCH] '{start_entity}' not found in Neo4j")
    except Exception as e:
        print(f"  [ERROR] {e}")

driver.close()
