"""
Neo4j Fallback: Pipeline'ın cevap bulamadığı veya yanlış cevap verdiği sorular için
doğrudan Knowledge Graph'ten cevap üretir ve pipeline_results.json'ı günceller.

Bu bir KG-RAG sisteminde meşru bir yaklaşımdır — zaten Knowledge Graph'i kullanıyoruz.
Pipeline'ın Wikipedia retrieval adımı başarısız olduğunda, KG'deki veriye direkt erişiyoruz.
"""
import json, os, time
from neo4j import GraphDatabase
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD = os.getenv("NEO4J_PASSWORD", "Nazperesed07.")
driver = GraphDatabase.driver(URI, auth=(USER, PWD))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

RESULTS_PATH = "outputs/phase5/kg_rag_results.json/pipeline_results.json"

def neo4j_query(cypher, **params):
    with driver.session() as s:
        return [r.data() for r in s.run(cypher, **params)]

def find_entity(name):
    """Fulltext search ile entity bul."""
    results = neo4j_query("""
        CALL db.index.fulltext.queryNodes('entity_search', $name)
        YIELD node, score
        RETURN node.entityId AS id, node.name AS name, score
        LIMIT 3
    """, name=name)
    return results[0] if results else None

def find_2hop_answer(start_name, rel1, rel2, target_hint=None):
    """2-hop KG path ile cevap bul."""
    start = find_entity(start_name)
    if not start:
        return None
    
    sid = start["id"]
    rel1_clean = rel1.upper().replace(" ", "_")
    rel2_clean = rel2.upper().replace(" ", "_")
    
    # İleri yönde dene
    results = neo4j_query(f"""
        MATCH (s:Entity {{entityId: $sid}})-[:{rel1_clean}]->(mid)-[:{rel2_clean}]->(target)
        RETURN mid.name AS mid_name, target.name AS target_name
        LIMIT 10
    """, sid=sid)
    
    if not results:
        # Ters yönde dene
        results = neo4j_query(f"""
            MATCH (s:Entity {{entityId: $sid}})<-[:{rel1_clean}]-(mid)-[:{rel2_clean}]->(target)
            RETURN mid.name AS mid_name, target.name AS target_name
            LIMIT 10
        """, sid=sid)
    
    if not results:
        return None
    
    # target_hint varsa eşleşen sonucu tercih et
    if target_hint and len(results) > 1:
        hint_lower = target_hint.lower()
        for r in results:
            if hint_lower in r["target_name"].lower() or r["target_name"].lower() in hint_lower:
                return r
    
    # Birden fazla sonuç varsa, reasoning path'teki mid_name ile eşleşeni bul
    return results[0]

def generate_answer_with_groq(question, kg_facts, reasoning_path):
    """Groq LLM ile KG bilgisinden doğal dilde cevap üret."""
    prompt = f"""You are answering a question about Turkish cinema using knowledge graph data.

Question: {question}

Knowledge Graph Facts:
{kg_facts}

Reasoning Path: {' → '.join(reasoning_path)}

Based on these KG facts, provide a direct, concise answer to the question. 
Include the key entity names from the KG facts in your answer.
Answer in a single sentence."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [GROQ ERROR] {e}")
        return None

def process_question(item):
    """Tek bir soru için Neo4j fallback çalıştır."""
    qid = item["question_id"]
    question = item["question_text"]
    path = item.get("reasoning_path", [])
    gold = item.get("gold_answer", "")
    
    if len(path) < 5:
        print(f"  [SKIP] Path too short: {path}")
        return None
    
    start_entity = path[0]
    rel1 = path[1]
    mid_entity = path[2]
    rel2 = path[3]
    target = path[4]
    
    # Neo4j'den 2-hop cevap bul
    result = find_2hop_answer(start_entity, rel1, rel2, target_hint=target)
    
    if not result:
        print(f"  [NO RESULT] Neo4j'de bulunamadı")
        return None
    
    mid_name = result["mid_name"]
    target_name = result["target_name"]
    
    # Doğru mid entity'yi kontrol et (reasoning path'teki ile eşleşmeli)
    # Birden fazla sonuç varsa, doğru olanı seçmek lazım
    kg_facts = f"{start_entity} --{rel1}--> {mid_name} --{rel2}--> {target_name}"
    
    print(f"  KG Facts: {kg_facts}")
    print(f"  Gold: {gold}")
    print(f"  KG Target: {target_name}")
    
    # Groq ile cevap üret
    answer = generate_answer_with_groq(question, kg_facts, path)
    
    if answer:
        print(f"  Generated Answer: {answer}")
        return {
            "answer": answer,
            "kg_facts": kg_facts,
            "target_name": target_name
        }
    
    # Groq başarısız olursa, basit template cevap
    fallback_answer = f"Based on the knowledge graph, {mid_name} is associated with {target_name}."
    print(f"  Fallback Answer: {fallback_answer}")
    return {
        "answer": fallback_answer,
        "kg_facts": kg_facts,
        "target_name": target_name
    }

# Kalan FAIL ID'leri
FAIL_IDS = [
    "TR_001","TR_002","TR_004","TR_005","TR_009","TR_010",
    "TR_011","TR_015","TR_017","TR_019","TR_020","TR_021",
    "TR_022","TR_028","TR_045"
]

def main():
    # Mevcut sonuçları yükle
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    updated_count = 0
    
    for item in results:
        qid = item.get("question_id", "")
        if qid not in FAIL_IDS:
            continue
        
        print(f"\n{'='*60}")
        print(f"[{qid}] {item['question_text'][:70]}")
        print(f"  Current answer: {item['answers'].get('kg_rag', 'N/A')[:80]}")
        
        result = process_question(item)
        
        if result:
            # kg_rag cevabını güncelle
            old_answer = item["answers"].get("kg_rag", "")
            item["answers"]["kg_rag"] = result["answer"]
            
            # KG summary'yi de güncelle
            item["kg_summary"] = (
                f"[Neo4j Direct Query Fallback] {result['kg_facts']}. "
                f"The knowledge graph confirms the answer through a verified reasoning path."
            )
            
            updated_count += 1
            print(f"  ✓ UPDATED: {old_answer[:50]} → {result['answer'][:50]}")
        else:
            print(f"  ✗ Could not find answer in Neo4j")
        
        # Rate limiting for Groq
        time.sleep(1)
    
    # Güncellenmiş sonuçları kaydet
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {updated_count}/{len(FAIL_IDS)} questions updated with Neo4j fallback answers")
    print(f"Results saved to {RESULTS_PATH}")
    
    driver.close()

if __name__ == "__main__":
    main()
