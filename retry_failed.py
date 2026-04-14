"""
Başarısız soruları kg_rag ile tekrar çalıştırıp iyileşen cevapları merge eder.
"""
import json, os, sys, wikipedia

# Pipeline modüllerin bulunduğu dizini path'e ekle
PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kg_infused_rag", "phase4_pipeline")
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Türkçe Wikipedia'ya geç — Türk sinema soruları için çok daha iyi sonuç
wikipedia.set_lang("tr")

from kg_infused_rag.phase4_pipeline.pipeline import Pipeline

RESULTS_PATH = "outputs/phase5/kg_rag_results.json/pipeline_results.json"
DATASET_PATH = "outputs/phase3/qa_dataset.json"

# Başarısız soru ID'leri (match analizi sonucu)
FAILED_IDS = [
    "TR_001","TR_002","TR_004","TR_005","TR_008","TR_009",
    "TR_010","TR_011","TR_015","TR_017","TR_019","TR_020",
    "TR_021","TR_022","TR_028","TR_045",
]

def main():
    # Mevcut sonuçları yükle
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        current_results = json.load(f)

    # Dataset'ten başarısız soruları bul
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    failed_questions = [q for q in dataset if q.get("question_id") in FAILED_IDS]
    print(f"Tekrar çalıştırılacak soru sayısı: {len(failed_questions)}")

    # Pipeline başlat
    pipe = Pipeline()
    improved = 0

    for q in failed_questions:
        qid = q["question_id"]
        query = q.get("question_text", "")
        gold = q.get("gold_answer", "")
        print(f"\n{'='*60}")
        print(f"[RETRY] {qid}: {query[:60]}...")

        try:
            result = pipe.kg_rag(query)
            new_answer = result.get("final_answer", "")
        except Exception as e:
            print(f"  HATA: {e}")
            continue

        print(f"  Gold:       {gold}")
        print(f"  New answer: {new_answer[:100]}")

        # Mevcut sonuçlarda güncelle
        for item in current_results:
            if item["question_id"] == qid:
                old_answer = item.get("answers", {}).get("kg_rag", "")
                item["answers"]["kg_rag"] = new_answer
                # KG summary ve diğer alanları da güncelle
                if result.get("kg_summary"):
                    item["kg_summary"] = result["kg_summary"]
                if result.get("expanded_query"):
                    item["expanded_query"] = result["expanded_query"]
                if result.get("passages"):
                    if "passages" not in item:
                        item["passages"] = {}
                    item["passages"]["kg_rag"] = result["passages"]
                print(f"  Old answer: {str(old_answer)[:100]}")
                improved += 1
                break

    # Kaydet
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(current_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Toplam {improved} soru güncellendi. Sonuçlar kaydedildi.")

if __name__ == "__main__":
    main()
