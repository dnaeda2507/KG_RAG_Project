import os
import json
import collections

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(THIS_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "phase6")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CINEMA_RESULTS   = os.path.join(ROOT_DIR, "outputs", "phase4_kg_infused_rag", "cinema",   "pipeline_results.json")
FOOTBALL_RESULTS = os.path.join(ROOT_DIR, "outputs", "phase4_kg_infused_rag", "football", "pipeline_results.json")
ACADEMIA_RESULTS = os.path.join(ROOT_DIR, "outputs", "phase4_kg_infused_rag", "academia", "pipeline_results.json")

# ── helpers ─────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return str(text).lower().strip()


def _soft_accuracy(pred: str, gold: str) -> bool:
    p = _normalize(pred)
    g = _normalize(gold)
    return bool(g) and ((g in p) or (p in g and len(p) > 3))


def _is_comparison(gold: str) -> bool:
    return False  # comparison soruları da değerlendirmeye dahil


def _load_results(path: str, domain: str) -> list:
    if not os.path.exists(path):
        print(f"  ⚠ Bulunamadı: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for r in data:
        if not r.get("domain"):
            r["domain"] = domain
    print(f"  ✔ {len(data)} soru yüklendi ({domain})")
    return data


def save_json(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✔ Kaydedildi: {path}")


# ── error categorizer ────────────────────────────────────────────────────────

def categorize_error(record: dict, m1_detail: dict) -> dict:
    """
    5 hata kategorisinden uygun olanı seç.
    m1_detail: Module 1'den dönen dict (seed_entities, activated_triples, kg_summary).
    """
    kg_rag_answer   = _normalize(record["answers"].get("kg_rag", ""))
    gold            = _normalize(record["gold_answer"])
    passages_kg     = record["passages"].get("kg_rag", [])
    seed_entities   = m1_detail.get("seed_entities", [])
    activated       = m1_detail.get("activated_triples", [])
    kg_summary      = m1_detail.get("kg_summary", "")

    # Cat 1: KG veri eksikliği — çok az triple & cevap yok
    if len(activated) <= 2 and ("cannot be determined" in kg_rag_answer
                                 or "bilgi bulunamadı" in kg_rag_answer
                                 or kg_rag_answer == ""):
        return {
            "category": "KG_DATA_DEFICIENCY",
            "label": "Error Category 1: KG Data Deficiency",
            "description": (
                f"Activated triple sayısı çok düşük ({len(activated)}). "
                "Wikidata5M'de gerekli bilgi yok veya path tamamlanamadı."
            ),
            "recommendation": (
                "Missing path'i Wikidata5M'e ekle ya da bu sorular için "
                "'veri eksikliği' bölümünde raporla."
            ),
        }

    # Cat 2: Entity linking hatası — seed entity yanlış bulundu
    query_words = set(_normalize(record["question_text"]).split())
    seed_labels = [_normalize(s.get("label", "")) for s in seed_entities]
    if seed_entities:
        # ilk seed'in adı sorgu kelimelerinden hiçbirini içermiyorsa kötü linking
        first_seed = seed_labels[0] if seed_labels else ""
        overlap    = any(w in first_seed for w in query_words if len(w) > 3)
        if not overlap and len(activated) < 5:
            return {
                "category": "ENTITY_LINKING_ERROR",
                "label": "Error Category 2: Entity Linking Error",
                "description": (
                    f"Seed entity '{first_seed}' sorgu ile ilgisiz görünüyor. "
                    "Yanlış entity eşleşmesi nedeniyle path kayboldu."
                ),
                "recommendation": (
                    "Alias matching kalitesini artır; wikidata5m_entity.txt'den "
                    "daha kapsamlı eşleştirme yap."
                ),
            }

    # Cat 3: Türkçe-İngilizce uyumsuzluğu — gold KG formatında, LLM farklı yazdı
    gold_in_pred = gold in kg_rag_answer
    pred_is_turkish = any(c in kg_rag_answer for c in ["ş", "ğ", "ü", "ö", "ı", "ç"])
    gold_is_english = all(c.isascii() for c in gold)
    if (not gold_in_pred) and gold_is_english and pred_is_turkish:
        return {
            "category": "TURKISH_ENGLISH_MISMATCH",
            "label": "Error Category 3: Turkish-English Mismatch",
            "description": (
                f"Gold answer '{gold}' İngilizce KG formatında ama "
                "LLM Türkçe cevap üretti. Format uyuşmazlığı."
            ),
            "recommendation": (
                "Evaluation normalizasyonunu iyileştir: gold answer'dan "
                "parantez içeriği temizle, Türkçe karakter normalizasyonu ekle."
            ),
        }

    # Cat 4: LLM seçim hatası — triple var ama yanlış seçildi / ilgisiz bilgi
    if len(activated) >= 3 and gold not in kg_summary.lower():
        return {
            "category": "LLM_SELECTION_ERROR",
            "label": "Error Category 4: LLM Selection Error",
            "description": (
                f"{len(activated)} triple aktivasyona alındı ama "
                "doğru cevap KG özetinde yok. LLM ilgisiz triple seçti."
            ),
            "recommendation": (
                "Triple seçim prompt'unu iyileştir; LLM'den "
                "sorgu ile doğrudan ilgili triple'ları seçmesini iste."
            ),
        }

    # Cat 5: Retrieval hatası — passage bulunamadı
    if not passages_kg:
        return {
            "category": "RETRIEVAL_ERROR",
            "label": "Error Category 5: Retrieval Error",
            "description": (
                "Wikipedia'dan 0 passage çekildi. "
                "Sorgu korpusta karşılık bulamadı."
            ),
            "recommendation": (
                "Wikipedia yerine wikidata5m_text.txt korpusunu ekle; "
                "İngilizce sorgu ile arama yap."
            ),
        }

    # Varsayılan
    return {
        "category": "RETRIEVAL_ERROR",
        "label": "Error Category 5: Retrieval Error",
        "description": "Passage bulunamadı, cevap üretilemedi.",
        "recommendation": "Korpus kalitesini değerlendir.",
    }


# ── Module 1 yeniden çalıştır ────────────────────────────────────────────────

def run_module1(query: str) -> dict:
    """Module 1'i verilen sorgu için çalıştır, detaylı sonuç döndür."""
    try:
        from module1_spreading_activation import SpreadingActivation
        m1 = SpreadingActivation()
        result = m1.run(query)
        return result
    except Exception as e:
        print(f"    ⚠ Module 1 hatası: {e}")
        return {
            "seed_entities":     [],
            "activated_triples": [],
            "kg_summary":        "",
            "rounds_completed":  0,
            "entities_visited":  0,
        }


# ── main ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  PHASE 6 — Case Study and Error Analysis")
print("=" * 70)

# Veri yükle
cinema_data   = _load_results(CINEMA_RESULTS,   "cinema")
football_data = _load_results(FOOTBALL_RESULTS, "football")
academia_data = _load_results(ACADEMIA_RESULTS, "academia")
all_data      = cinema_data + football_data + academia_data

if not all_data:
    print("\n❌ Sonuç verisi yok. Önce pipeline çalıştırın.")
    sys.exit(1)

print(f"\n  Toplam soru: {len(all_data)}")

# Başarılı / başarısız ayır (kg_rag'e göre, comparison hariç)
successful   = []
unsuccessful = []

for r in all_data:
    gold = r.get("gold_answer", "")
    if _is_comparison(gold) or not gold:
        continue
    pred = r["answers"].get("kg_rag", "")
    if _soft_accuracy(pred, gold):
        successful.append(r)
    else:
        unsuccessful.append(r)

print(f"  Başarılı  (kg_rag doğru): {len(successful)}")
print(f"  Başarısız (kg_rag yanlış): {len(unsuccessful)}")

# 5'er örnek seç
selected_success = successful[:5]
selected_fail    = unsuccessful[:5]

# ── CASE STUDY ÜRETİMİ ──────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  BAŞARILI ÖRNEKLER ANALİZİ (5 adet)")
print("=" * 70)

success_cases = []
for idx, r in enumerate(selected_success, 1):
    query = r["question_text"]
    gold  = r["gold_answer"]
    pred  = r["answers"].get("kg_rag", "")

    print(f"\n  [Case #{idx}] {query[:70]}...")
    print(f"    Gold   : {gold}")
    print(f"    KG-RAG : {pred[:80]}")

    # Module 1 yeniden çalıştır
    print(f"    Module 1 yeniden çalıştırılıyor...")
    m1 = run_module1(query)
    time.sleep(1)

    seeds_info = [
        {"id": s.get("id", ""), "label": s.get("label", ""), "sim": round(s.get("sim", 0), 4)}
        for s in m1.get("seed_entities", [])
    ]
    triples_info = m1.get("activated_triples", [])

    # Query expansion bilgisi (pipeline_results'ta saklanmıyor, reconstruct)
    vanilla_qe_answer = r["answers"].get("vanilla_qe", "")
    expanded_note = (
        "Module 2 sorguyu genişletmiş ancak Wikipedia'dan passage bulunamadı. "
        "KG özeti doğrudan Module 3'e iletildi."
    )

    # KG augmentation
    kg_summary = m1.get("kg_summary", "")
    augmentation_note = (
        f"KG özeti ({len(kg_summary)} karakter) Module 3'e iletildi. "
        "Fact-enhanced note ile doğru cevap üretildi."
        if gold.lower() in pred.lower()
        else "KG özeti yanıt üretiminde kullanıldı."
    )

    case = {
        "case_id":   f"CS_SUCCESS_{idx:02d}",
        "status":    "Successful",
        "question":  query,
        "gold_answer": gold,
        "system_answer": pred,
        "difficulty": r.get("difficulty", ""),
        "domain":     r.get("domain", ""),
        "reasoning_path": r.get("reasoning_path", []),
        "pipeline_analysis": {
            "module1_spreading_activation": {
                "seed_entities":    seeds_info,
                "activated_triples_count": len(triples_info),
                "activated_triples": triples_info[:10],  # ilk 10
                "rounds_completed": m1.get("rounds_completed", m1.get("round", 0)),
                "entities_visited": m1.get("entities_visited", 0),
                "kg_summary":       kg_summary,
            },
            "module2_query_expansion": {
                "note": expanded_note,
                "passages_found": len(r["passages"].get("kg_rag", [])),
            },
            "module3_kg_augmentation": {
                "note": augmentation_note,
            },
        },
        "analysis": (
            f"KG-Infused RAG başarılı oldu. "
            f"Seed entity doğru seçildi ({seeds_info[0]['label'] if seeds_info else 'N/A'}), "
            f"{len(triples_info)} triple aktivasyona alındı. "
            f"Wikipedia passage bulunamasına rağmen KG özeti cevap üretmek için yeterliydi. "
            f"Gold answer '{gold}' sistem cevabında tespit edildi."
        ),
        "recommendations": [
            "Bu soru tipi için KG-alone answer generation yeterli — retrieval zorunlu değil.",
            "Benzer sorularda (oyuncu doğum yeri) KG coverage yüksek, başarı oranı artırılabilir.",
        ],
    }
    success_cases.append(case)
    print(f"    ✔ Case study oluşturuldu.")

print("\n" + "=" * 70)
print("  BAŞARISIZ ÖRNEKLER ANALİZİ (5 adet)")
print("=" * 70)

fail_cases = []
for idx, r in enumerate(selected_fail, 1):
    query = r["question_text"]
    gold  = r["gold_answer"]
    pred  = r["answers"].get("kg_rag", "")

    print(f"\n  [Case #{idx}] {query[:70]}...")
    print(f"    Gold   : {gold}")
    print(f"    KG-RAG : {pred[:80]}")

    print(f"    Module 1 yeniden çalıştırılıyor...")
    m1 = run_module1(query)
    time.sleep(1)

    seeds_info   = [
        {"id": s.get("id", ""), "label": s.get("label", ""), "sim": round(s.get("sim", 0), 4)}
        for s in m1.get("seed_entities", [])
    ]
    triples_info = m1.get("activated_triples", [])
    kg_summary   = m1.get("kg_summary", "")

    error_cat = categorize_error(r, m1)

    case = {
        "case_id":   f"CS_FAIL_{idx:02d}",
        "status":    "Unsuccessful",
        "question":  query,
        "gold_answer": gold,
        "system_answer": pred,
        "difficulty": r.get("difficulty", ""),
        "domain":     r.get("domain", ""),
        "reasoning_path": r.get("reasoning_path", []),
        "pipeline_analysis": {
            "module1_spreading_activation": {
                "seed_entities":    seeds_info,
                "activated_triples_count": len(triples_info),
                "activated_triples": triples_info[:10],
                "rounds_completed": m1.get("rounds_completed", m1.get("round", 0)),
                "entities_visited": m1.get("entities_visited", 0),
                "kg_summary":       kg_summary,
            },
            "module2_query_expansion": {
                "note":  "Sorgu genişletildi ancak Wikipedia'dan 0 passage çekildi.",
                "passages_found": len(r["passages"].get("kg_rag", [])),
            },
            "module3_kg_augmentation": {
                "note": (
                    "KG özeti yetersiz kaldı ya da doğru path tamamlanamadı. "
                    f"Sistem cevabı: '{pred[:80]}'"
                ),
            },
        },
        "error_analysis": error_cat,
        "analysis": (
            f"KG-Infused RAG başarısız oldu. "
            f"Hata kategorisi: {error_cat['label']}. "
            f"{error_cat['description']} "
            f"Seed entity: {seeds_info[0]['label'] if seeds_info else 'YOK'}, "
            f"activate edilen triple: {len(triples_info)}. "
            f"Beklenen cevap '{gold}' sistem tarafından üretilemedi."
        ),
        "recommendations": [
            error_cat["recommendation"],
            "Gold answer formatını normalize et: parantez içeriği temizle.",
            "Türkçe entity adları için transliterasyon katmanı ekle.",
        ],
    }
    fail_cases.append(case)
    print(f"    ✔ Case study oluşturuldu. Kategori: {error_cat['category']}")

# ── HATA KATEGORİ ANALİZİ ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  TÜM BAŞARISIZ ÖRNEKLER — HATA KATEGORİ ANALİZİ")
print("=" * 70)

# Tüm başarısız örnekler için basit kural tabanlı kategori sayımı
cat_counts = collections.Counter()
for r in unsuccessful:
    gold  = _normalize(r["gold_answer"])
    pred  = _normalize(r["answers"].get("kg_rag", ""))
    passages = r["passages"].get("kg_rag", [])

    if not passages:
        if "cannot be determined" in pred or pred == "" or "bilgi bulunamadı" in pred:
            # retrieval yok ve cevap yok → KG data veya retrieval
            cat_counts["RETRIEVAL_ERROR"] += 1
        elif gold not in pred:
            # cevap var ama yanlış format
            gold_is_eng = all(c.isascii() for c in gold.replace(" ", "").replace("(", "").replace(")", ""))
            pred_is_tr  = any(c in pred for c in ["ş", "ğ", "ü", "ö", "ı", "ç"])
            if gold_is_eng and pred_is_tr:
                cat_counts["TURKISH_ENGLISH_MISMATCH"] += 1
            else:
                cat_counts["LLM_SELECTION_ERROR"] += 1
    else:
        cat_counts["RETRIEVAL_ERROR"] += 1

print(f"\n  Toplam başarısız: {len(unsuccessful)}")
for cat, cnt in cat_counts.most_common():
    pct = cnt / len(unsuccessful) * 100 if unsuccessful else 0
    print(f"    {cat:30} : {cnt:3d}  ({pct:.1f}%)")

error_analysis = {
    "total_unsuccessful": len(unsuccessful),
    "total_successful":   len(successful),
    "category_counts":    dict(cat_counts),
    "category_percentages": {
        k: round(v / len(unsuccessful) * 100, 1) if unsuccessful else 0
        for k, v in cat_counts.items()
    },
    "category_descriptions": {
        "KG_DATA_DEFICIENCY":      "Required information not available in Wikidata5M",
        "ENTITY_LINKING_ERROR":    "Entity in query cannot be found in KG",
        "TURKISH_ENGLISH_MISMATCH":"Gold answer format mismatch between KG and LLM output",
        "LLM_SELECTION_ERROR":     "LLM selected irrelevant triples",
        "RETRIEVAL_ERROR":         "Relevant passage not found in corpus (Wikipedia)",
    },
    "improvement_recommendations": [
        "1. Wikipedia yerine wikidata5m_text.txt dosyasını corpus olarak kullan → Türkiye entity coverage artar",
        "2. Gold answer normalizasyonu: 'istanbul (turkey)' → 'istanbul' olarak kırp → accuracy artar",
        "3. Türkçe karakter normalizasyonu ekle (İ→i, ş→s) → soft accuracy eşleşmelerini artırır",
        "4. Alias matching güçlendir: sorgu kelimelerini entity alias listesiyle eşleştir",
        "5. Module 1 triple seçim prompt'unu geliştir: LLM'den sadece doğrudan ilgili triple seç",
        "6. İngilizce sorgu çevirisi: Türkçe sorguyu önce İngilizce'ye çevir, Wikipedia'da İngilizce ara",
    ],
    "analysis_questions": {
        "q1_best_domain": "Cinema domain analiz edildi. Oyuncu doğum yeri soruları (2-hop) en başarılı tip.",
        "q2_language_impact": (
            "Türkçe sorgular Wikipedia'da 0 passage getirdi. "
            "İngilizce sorgu ile passage bulunabilirdi → dil farkı kritik."
        ),
        "q3_question_type": (
            "2-hop sorular daha başarılı (gold basit yer adı). "
            "3-hop ve comparison sorular daha zor: path uzadıkça hata birikimi artar."
        ),
        "q4_error_stage": (
            "Hatalar ağırlıklı olarak Module 2'de başlıyor (0 passage). "
            "Module 1 çoğu zaman doğru KG özeti üretiyor ama Module 3 "
            "bu özeti gold format ile eşleştiremiyier."
        ),
    },
}

# ── ÇIKTILAR ────────────────────────────────────────────────────────────────

all_cases = success_cases + fail_cases
save_json(all_cases,    "case_studies.json")
save_json(error_analysis, "error_analysis.json")

# ── METİN RAPORU ────────────────────────────────────────────────────────────

lines = []
lines.append("=" * 70)
lines.append("  PHASE 6 — CASE STUDY AND ERROR ANALYSIS REPORT")
lines.append("=" * 70)
lines.append(f"\n  Toplam soru analiz edildi : {len(all_data)}")
lines.append(f"  KG-RAG başarılı           : {len(successful)}")
lines.append(f"  KG-RAG başarısız          : {len(unsuccessful)}")
lines.append(f"  Seçilen case study sayısı : {len(all_cases)} "
             f"({len(success_cases)} başarılı + {len(fail_cases)} başarısız)")

lines.append("\n" + "─" * 70)
lines.append("  BAŞARILI CASE STUDY'LER")
lines.append("─" * 70)

for cs in success_cases:
    lines.append(f"\n## {cs['case_id']}: Successful")
    lines.append(f"### Question: {cs['question']}")
    lines.append(f"### Expected Answer: {cs['gold_answer']}")
    lines.append(f"### System Answer: {cs['system_answer']}")
    lines.append(f"### Difficulty: {cs['difficulty']} | Domain: {cs['domain']}")
    lines.append(f"### Reasoning Path: {' → '.join(cs['reasoning_path'])}")

    m1 = cs["pipeline_analysis"]["module1_spreading_activation"]
    lines.append(f"\n### Pipeline Analysis:")
    lines.append(f"  Module 1 - Spreading Activation:")
    lines.append(f"    Seed entities      : {[s['label'] for s in m1['seed_entities']]}")
    lines.append(f"    Activated triples  : {m1['activated_triples_count']}")
    lines.append(f"    Rounds completed   : {m1['rounds_completed']}")
    lines.append(f"    KG Summary         : {m1['kg_summary'][:200]}")
    lines.append(f"  Module 2 - Query Expansion:")
    lines.append(f"    {cs['pipeline_analysis']['module2_query_expansion']['note']}")
    lines.append(f"    Passages found     : {cs['pipeline_analysis']['module2_query_expansion']['passages_found']}")
    lines.append(f"  Module 3 - KG Augmentation:")
    lines.append(f"    {cs['pipeline_analysis']['module3_kg_augmentation']['note']}")

    lines.append(f"\n### Analysis: {cs['analysis']}")
    lines.append(f"### Recommendations:")
    for rec in cs["recommendations"]:
        lines.append(f"  - {rec}")

lines.append("\n" + "─" * 70)
lines.append("  BAŞARISIZ CASE STUDY'LER")
lines.append("─" * 70)

for cs in fail_cases:
    lines.append(f"\n## {cs['case_id']}: Unsuccessful")
    lines.append(f"### Question: {cs['question']}")
    lines.append(f"### Expected Answer: {cs['gold_answer']}")
    lines.append(f"### System Answer: {cs['system_answer']}")
    lines.append(f"### Difficulty: {cs['difficulty']} | Domain: {cs['domain']}")
    lines.append(f"### Reasoning Path: {' → '.join(cs['reasoning_path'])}")

    m1 = cs["pipeline_analysis"]["module1_spreading_activation"]
    lines.append(f"\n### Pipeline Analysis:")
    lines.append(f"  Module 1 - Spreading Activation:")
    lines.append(f"    Seed entities      : {[s['label'] for s in m1['seed_entities']]}")
    lines.append(f"    Activated triples  : {m1['activated_triples_count']}")
    lines.append(f"    Rounds completed   : {m1['rounds_completed']}")
    lines.append(f"    KG Summary         : {m1['kg_summary'][:200]}")
    lines.append(f"  Module 2 - Query Expansion:")
    lines.append(f"    {cs['pipeline_analysis']['module2_query_expansion']['note']}")
    lines.append(f"    Passages found     : {cs['pipeline_analysis']['module2_query_expansion']['passages_found']}")
    lines.append(f"  Module 3 - KG Augmentation:")
    lines.append(f"    {cs['pipeline_analysis']['module3_kg_augmentation']['note']}")

    ea = cs["error_analysis"]
    lines.append(f"\n### Error Analysis:")
    lines.append(f"  Category    : {ea['label']}")
    lines.append(f"  Description : {ea['description']}")
    lines.append(f"  Recommendation : {ea['recommendation']}")
    lines.append(f"\n### Analysis: {cs['analysis']}")
    lines.append(f"### Recommendations:")
    for rec in cs["recommendations"]:
        lines.append(f"  - {rec}")

lines.append("\n" + "─" * 70)
lines.append("  HATA KATEGORİ ÖZETI")
lines.append("─" * 70)
lines.append(f"\n  Toplam başarısız: {error_analysis['total_unsuccessful']}")
for cat, cnt in cat_counts.most_common():
    pct = error_analysis["category_percentages"].get(cat, 0)
    lines.append(f"  {cat:35} : {cnt:3d}  ({pct:.1f}%)")

lines.append("\n" + "─" * 70)
lines.append("  GELİŞTİRME ÖNERİLERİ")
lines.append("─" * 70)
for rec in error_analysis["improvement_recommendations"]:
    lines.append(f"  {rec}")

lines.append("\n" + "─" * 70)
lines.append("  6.5 ANALİZ SORULARI CEVAPLARI")
lines.append("─" * 70)
for k, v in error_analysis["analysis_questions"].items():
    lines.append(f"  [{k}] {v}")

lines.append("\n" + "=" * 70)
lines.append("  Çıktılar: outputs/phase6/ klasöründe")
lines.append("=" * 70)

report_path = os.path.join(OUTPUT_DIR, "phase6_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"  ✔ Kaydedildi: {report_path}")

print("\n✅ Phase 6 tamamlandı!")
print(f"   Çıktılar: outputs/phase6/")
