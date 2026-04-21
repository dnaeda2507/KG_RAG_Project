import sys, os, json, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "outputs", "phase4_kg_infused_rag", "football", "pipeline_results.json"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "outputs", "phase5_eval", "football"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

METHODS = ["no_retrieval", "vanilla_rag", "vanilla_qe", "kg_rag"]
METHOD_LABELS = {
    "no_retrieval": "No-Retrieval",
    "vanilla_rag":  "Vanilla RAG",
    "vanilla_qe":   "Vanilla QE",
    "kg_rag":       "KG-Infused RAG",
}

def save_json(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✔ Kaydedildi: {path}")

def _normalize(text: str) -> str:
    return str(text).lower().strip()

def _exact_match(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)

def _token_f1(pred: str, gold: str) -> float:
    p_tokens = set(_normalize(pred).split())
    g_tokens = set(_normalize(gold).split())
    if not p_tokens or not g_tokens:
        return 0.0
    common = p_tokens & g_tokens
    if not common:
        return 0.0
    prec = len(common) / len(p_tokens)
    rec  = len(common) / len(g_tokens)
    return 2 * prec * rec / (prec + rec)

def _soft_accuracy(pred: str, gold: str) -> bool:
    p = _normalize(pred)
    g = _normalize(gold)
    return (g in p) or (p in g and len(p) > 3)

def _retrieval_recall(passages: list, gold: str) -> bool:
    if not passages:
        return False
    gold_norm    = _normalize(gold)
    passage_text = " ".join(_normalize(p.get("content", "")) for p in passages)
    return gold_norm in passage_text or any(
        tok in passage_text for tok in gold_norm.split() if len(tok) > 3
    )

def _compute_metrics(results: list, method: str) -> dict:
    acc_c = em_c = f1_sum = rec_c = rec_tot = total = 0
    by_diff = {}
    errors = []
    for r in results:
        gold = str(r.get("gold_answer", "")).strip()
        pred = str(r.get("answers", {}).get(method, "")).strip()
        diff = r.get("difficulty", "unknown")
        if not gold:
            continue
        total += 1
        hit_acc = _soft_accuracy(pred, gold)
        hit_em  = _exact_match(pred, gold)
        f1      = _token_f1(pred, gold)
        f1_sum += f1
        if hit_acc: acc_c += 1
        if hit_em:  em_c  += 1
        passages = r.get("passages", {}).get(method, [])
        if passages:
            rec_tot += 1
            if _retrieval_recall(passages, gold):
                rec_c += 1
        if diff not in by_diff:
            by_diff[diff] = {"total": 0, "acc": 0, "em": 0, "f1": 0.0}
        by_diff[diff]["total"] += 1
        by_diff[diff]["f1"]    += f1
        if hit_acc: by_diff[diff]["acc"] += 1
        if hit_em:  by_diff[diff]["em"]  += 1
        if not hit_acc:
            errors.append({
                "question_id":   r.get("question_id", ""),
                "question_text": r.get("question_text", "")[:80],
                "gold":          gold,
                "predicted":     pred,
                "difficulty":    diff,
                "method":        method,
            })
    acc   = round(acc_c / total, 4) if total > 0 else 0.0
    em    = round(em_c  / total, 4) if total > 0 else 0.0
    f1    = round(f1_sum / total,  4) if total > 0 else 0.0
    recall = round(rec_c / rec_tot, 4) if rec_tot > 0 else None
    diff_summary = {}
    for d, v in by_diff.items():
        diff_summary[d] = {
            "total":    v["total"],
            "accuracy": round(v["acc"] / v["total"], 4) if v["total"] > 0 else 0,
            "em":       round(v["em"]  / v["total"], 4) if v["total"] > 0 else 0,
            "f1":       round(v["f1"]  / v["total"], 4) if v["total"] > 0 else 0,
        }
    return {
        "total":            total,
        "accuracy":         acc,
        "f1":               f1,
        "exact_match":      em,
        "retrieval_recall": recall,
        "by_difficulty":    diff_summary,
        "errors":           errors,
    }

def _categorize_error(err: dict) -> str:
    pred  = _normalize(err["predicted"])
    gold  = _normalize(err["gold"])
    # Boş cevap → entity linking / KG eksikliği
    if not pred or pred in ("", "unknown", "cannot be determined"):
        return "empty_answer"
    # Kısmi eşleşme var ama tam değil → retrieval yüzeysel
    if any(tok in pred for tok in gold.split() if len(tok) > 3):
        return "partial_match"
    # Gold 1-2 kelimeyse → yanlış entity seçimi
    if len(gold.split()) <= 2:
        return "wrong_entity"
    # Uzun gold ama tamamen farklı → LLM hallücination
    return "hallucination"

def run_phase5_eval_football():
    if not os.path.exists(RESULTS_PATH):
        print(f"[HATA] Sonuç dosyası bulunamadı: {RESULTS_PATH}")
        return
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"[Phase5] Football Evaluation Başladı ({len(results)} soru)")

    exp = {}
    for m in METHODS:
        exp[m] = _compute_metrics(results, m)

    # Görsel: Yöntem karşılaştırma bar chart
    COLORS = {
        "no_retrieval": "#e74c3c",
        "vanilla_rag":  "#e67e22",
        "vanilla_qe":   "#3498db",
        "kg_rag":       "#2ecc71",
    }
    labels = [METHOD_LABELS[m] for m in METHODS]
    accs   = [exp[m]["accuracy"] for m in METHODS]
    f1s    = [exp[m]["f1"] for m in METHODS]
    x = np.arange(len(METHODS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color=[COLORS[m] for m in METHODS])
    bars2 = ax.bar(x + width/2, f1s, width, label="F1", color="#888888", alpha=0.5)
    ax.set_ylabel("Score")
    ax.set_title("Method Comparison (Football)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1)
    ax.legend()
    for bar, v in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{v:.1%}", ha="center", fontsize=9)
    for bar, v in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{v:.2f}", ha="center", fontsize=8, color="#444444")
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "football_method_comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Grafik kaydedildi: {chart_path}")

    # Görsel: Zorluk bazlı KG-RAG accuracy
    kg = exp["kg_rag"]["by_difficulty"]
    if kg:
        diff_labels = list(kg.keys())
        diff_accs = [kg[d]["accuracy"] for d in diff_labels]
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        bars = ax2.bar(diff_labels, diff_accs, color="#2ecc71")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("KG-RAG Accuracy by Difficulty (Football)")
        ax2.set_ylim(0, 1)
        for bar, v in zip(bars, diff_accs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{v:.1%}", ha="center", fontsize=9)
        plt.tight_layout()
        diff_chart_path = os.path.join(OUTPUT_DIR, "football_kg_rag_difficulty.png")
        plt.savefig(diff_chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✔ Grafik kaydedildi: {diff_chart_path}")
    if not os.path.exists(RESULTS_PATH):
        print(f"[HATA] Sonuç dosyası bulunamadı: {RESULTS_PATH}")
        return
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"[Phase5] Football Evaluation Başladı ({len(results)} soru)")

    exp = {}
    for m in METHODS:
        exp[m] = _compute_metrics(results, m)

    print(f"\n  {'Yöntem':20} | {'Toplam':>6} | {'Acc':>7} | {'F1':>7} | {'EM':>7} | {'Ret.Recall':>10}")
    print("  " + "─" * 70)
    for m in METHODS:
        s  = exp[m]
        rr = s["retrieval_recall"]
        rr_str = f"{rr:>10.2%}" if rr is not None else f"{'N/A':>10}"
        print(f"  {METHOD_LABELS[m]:20} | {s['total']:>6} | "
              f"{s['accuracy']:>7.2%} | {s['f1']:>7.4f} | "
              f"{s['exact_match']:>7.2%} | {rr_str}")

    print("\n  Zorluk bazlı (KG-Infused RAG):")
    kg = exp["kg_rag"]["by_difficulty"]
    for diff, dv in sorted(kg.items()):
        print(f"    {diff:12} → Acc:{dv['accuracy']:.2%}  "
              f"F1:{dv['f1']:.4f}  EM:{dv['em']:.2%}  ({dv['total']} soru)")

    # Hata analizi
    kg_errors = exp["kg_rag"]["errors"]
    cat_counts = collections.Counter(_categorize_error(e) for e in kg_errors)
    print(f"\n  KG-Infused RAG hataları: {len(kg_errors)} soru")
    print(f"\n  Hata kategorileri:")
    for cat, cnt in cat_counts.most_common():
        pct = cnt / len(kg_errors) * 100 if kg_errors else 0
        bar = "█" * min(cnt, 20)
        print(f"    {cat:20} : {cnt:>4}  ({pct:5.1f}%)  {bar}")

    # Zorluk bazlı hatalar
    diff_errors = collections.Counter(e["difficulty"] for e in kg_errors)
    print(f"\n  Zorluk bazlı hatalar (KG-RAG):")
    for diff, cnt in diff_errors.most_common():
        print(f"    {diff:15} : {cnt}")

    # Raporu kaydet
    report = {
        "method_comparison": exp,
        "error_categories": dict(cat_counts),
        "difficulty_errors": dict(diff_errors),
        "total_errors": len(kg_errors),
    }
    save_json(report, "football_eval_report.json")

    # TXT rapor
    lines = []
    lines.append("=" * 70)
    lines.append("  PHASE 5 — FOOTBALL EVALUATION RAPORU")
    lines.append("=" * 70)
    lines.append(f"\n  Toplam soru        : {len(results)}")
    lines.append("\n" + "─" * 70)
    lines.append("  Yöntem Karşılaştırması")
    lines.append("─" * 70)
    lines.append(f"  {'Yöntem':20} | {'Acc':>7} | {'F1':>7} | {'EM':>7} | {'Ret.Recall':>10}")
    lines.append("  " + "─" * 58)
    for m in METHODS:
        s  = exp[m]
        rr = s["retrieval_recall"]
        rr_str = f"{rr:>10.2%}" if rr is not None else f"{'N/A':>10}"
        lines.append(f"  {METHOD_LABELS[m]:20} | {s['accuracy']:>7.2%} | "
                     f"{s['f1']:>7.4f} | {s['exact_match']:>7.2%} | {rr_str}")
    lines.append("\n" + "─" * 70)
    lines.append("  Zorluk Bazlı Analiz (KG-Infused RAG)")
    lines.append("─" * 70)
    for diff, dv in sorted(kg.items()):
        lines.append(f"    {diff:12} → Acc:{dv['accuracy']:.2%}  "
                     f"F1:{dv['f1']:.4f}  EM:{dv['em']:.2%}  ({dv['total']} soru)")
    lines.append("\n" + "─" * 70)
    lines.append("  Hata Kategorileri (KG-Infused RAG)")
    lines.append("─" * 70)
    for cat, cnt in cat_counts.most_common():
        lines.append(f"    {cat:20} : {cnt} ({cnt/len(kg_errors)*100:.1f}%)" if kg_errors else "")
    lines.append("\n" + "─" * 70)
    lines.append("  Zorluk Bazlı Hatalar (KG-Infused RAG)")
    lines.append("─" * 70)
    for diff, cnt in diff_errors.most_common():
        lines.append(f"    {diff:15} : {cnt}")
    lines.append("\n" + "=" * 70)
    lines.append("  Çıktılar: outputs/phase5_eval/football/ klasöründe")
    lines.append("=" * 70)
    report_text = "\n".join(lines)
    report_path = os.path.join(OUTPUT_DIR, "football_eval_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  ✔ Rapor kaydedildi: {report_path}")
    print(report_text)

if __name__ == "__main__":
    run_phase5_eval_football()
