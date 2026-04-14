"""Tüm FAIL caselar icin tam system cevaplarını ve neden eşleşmediğini analiz eder."""
import json, re as _re

with open('outputs/phase5/kg_rag_results.json/pipeline_results.json') as f:
    data = json.load(f)

def normalize(text):
    t = text.lower().strip()
    for a, b in [('ı','i'),('İ','i'),('ş','s'),('Ş','s'),('ç','c'),('Ç','c'),
                  ('ö','o'),('Ö','o'),('ü','u'),('Ü','u'),('ğ','g'),('Ğ','g'),
                  ('i̇','i'),('â','a'),('î','i'),('û','u')]:
        t = t.replace(a, b)
    t = _re.sub(r'\([^)]*\)', '', t)
    t = _re.sub(r'[.,;:!\-\'\"/]', ' ', t)
    t = _re.sub(r'\s+', ' ', t).strip()
    return t

# Tüm cevaplanamayan soruları atla, sadece cevap veren ama FAIL olanları göster
fails_with_answer = [1,2,4,8,10,17,20,21,28,45]
for item in data[:50]:
    idx = int(item['question_id'].replace('TR_',''))
    if idx not in fails_with_answer:
        continue
    gold = item.get('gold_answer','').strip()
    pred = (item.get('answers',{}).get('kg_rag','') or '').strip()
    if 'cannot be determined' in pred.lower():
        continue
    g = normalize(gold)
    p = normalize(pred)
    print(f"=== {item['question_id']} ===")
    print(f"  GOLD_N: {g}")
    print(f"  SYS_N:  {p}")
    print(f"  FULL:   {pred}")
    print()
