import os
import sys
import json
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "phase5"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


FOOTBALL_RESULTS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "outputs", "phase4_kg_infused_rag", "football", "pipeline_results.json"
))
CINEMA_RESULTS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "outputs", "phase4_kg_infused_rag", "cinema", "pipeline_results.json"
))


METHODS = ["no_retrieval", "vanilla_rag", "vanilla_qe", "kg_rag"]
METHOD_LABELS = {
    "no_retrieval": "No-Retrieval",
    "vanilla_rag":  "Vanilla RAG",
    "vanilla_qe":   "Vanilla QE",
    "kg_rag":       "KG-Infused RAG",
}


# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────────────────────

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


def _is_comparison(gold: str) -> bool:
    return False  # comparison soruları da değerlendirmeye dahil


def _retrieval_recall(passages: list, gold: str) -> bool:
    if not passages:
        return False
    gold_norm    = _normalize(gold)
    passage_text = " ".join(_normalize(p.get("content", "")) for p in passages)
    return gold_norm in passage_text or any(
        tok in passage_text for tok in gold_norm.split() if len(tok) > 3
    )


def _load_results(path: str, domain_label: str) -> list:
    if not os.path.exists(path):
        print(f"  ⚠ Sonuç dosyası bulunamadı: {path}")
        print(f"    Önce pipeline_{'football' if 'football' in path else 'cinema'}.py çalıştırın!")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # domain etiketini zorla
    for r in data:
        if not r.get("domain"):
            r["domain"] = domain_label
    print(f"  ✔ {len(data)} soru yüklendi: {path}")
    return data


def _compute_metrics(results: list, method: str) -> dict:
    acc_c = em_c = f1_sum = rec_c = rec_tot = total = 0

    by_diff   = {}
    errors    = []

    for r in results:
        gold = str(r.get("gold_answer", "")).strip()
        pred = str(r.get("answers", {}).get(method, "")).strip()
        diff = r.get("difficulty", "unknown")

        if not gold or _is_comparison(gold):
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

        # difficulty breakdown
        if diff not in by_diff:
            by_diff[diff] = {"total": 0, "acc": 0, "em": 0, "f1": 0.0}
        by_diff[diff]["total"] += 1
        by_diff[diff]["f1"]    += f1
        if hit_acc: by_diff[diff]["acc"] += 1
        if hit_em:  by_diff[diff]["em"]  += 1

        # hata kaydı
        if not hit_acc:
            errors.append({
                "question_id":   r.get("question_id", ""),
                "question_text": r.get("question_text", "")[:80],
                "gold":          gold,
                "predicted":     pred,
                "difficulty":    diff,
                "domain":        r.get("domain", ""),
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


# ─────────────────────────────────────────────────────────────────────────────
# VERİ YÜKLEME
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  PHASE 5 — Experiments & Evaluation")
print("=" * 70)


football_data = _load_results(FOOTBALL_RESULTS, "football")
cinema_data   = _load_results(CINEMA_RESULTS,   "cinema")
all_data      = football_data + cinema_data

if not all_data:
    print("\n\u274c Hiç sonuç verisi yok. Önce pipeline çalıştırın.")
    sys.exit(1)


print(f"\n  Toplam soru: {len(all_data)} (football: {len(football_data)}, cinema: {len(cinema_data)})")


# ─────────────────────────────────────────────────────────────────────────────
# DENEY 1: 4 YÖNTEM KARŞILAŞTIRMA (tüm sorular)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  DENEY 1: 4 Yöntem Karşılaştırması")
print("=" * 70)

exp1 = {}
for m in METHODS:
    exp1[m] = _compute_metrics(all_data, m)

print(f"\n  {'Yöntem':20} | {'Toplam':>6} | {'Acc':>7} | {'F1':>7} | "
      f"{'EM':>7} | {'Ret.Recall':>10}")
print("  " + "─" * 70)
for m in METHODS:
    s  = exp1[m]
    rr = s["retrieval_recall"]
    rr_str = f"{rr:>10.2%}" if rr is not None else f"{'N/A':>10}"
    print(f"  {METHOD_LABELS[m]:20} | {s['total']:>6} | "
          f"{s['accuracy']:>7.2%} | {s['f1']:>7.4f} | "
          f"{s['exact_match']:>7.2%} | {rr_str}")

print("\n  Zorluk bazlı (KG-Infused RAG):")
kg = exp1["kg_rag"]["by_difficulty"]
for diff, dv in sorted(kg.items()):
    print(f"    {diff:12} → Acc:{dv['accuracy']:.2%}  "
          f"F1:{dv['f1']:.4f}  EM:{dv['em']:.2%}  ({dv['total']} soru)")

save_json(exp1, "experiment1_method_comparison.json")


# ─────────────────────────────────────────────────────────────────────────────
# DENEY 2: DOMAIN BAZLI ANALİZ
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  DENEY 2: Domain Bazlı Analiz")
print("=" * 70)


exp2 = {}
for domain, data in [("football", football_data), ("cinema", cinema_data)]:
    if not data:
        continue
    exp2[domain] = {}
    for m in METHODS:
        exp2[domain][m] = _compute_metrics(data, m)

    print(f"\n  Domain: {domain.upper()}")
    print(f"  {'Yöntem':20} | {'Acc':>7} | {'F1':>7} | {'EM':>7} | {'Ret.Recall':>10}")
    print("  " + "─" * 58)
    for m in METHODS:
        s  = exp2[domain][m]
        rr = s["retrieval_recall"]
        rr_str = f"{rr:>10.2%}" if rr is not None else f"{'N/A':>10}"
        print(f"  {METHOD_LABELS[m]:20} | {s['accuracy']:>7.2%} | "
              f"{s['f1']:>7.4f} | {s['exact_match']:>7.2%} | {rr_str}")

# En iyi domain tespiti
best_domain = None
best_acc    = -1
for domain in exp2:
    acc = exp2[domain].get("kg_rag", {}).get("accuracy", 0)
    if acc > best_acc:
        best_acc    = acc
        best_domain = domain

if best_domain:
    print(f"\n  → KG-Infused RAG en iyi domain: {best_domain.upper()} "
          f"(Acc={best_acc:.2%})")

save_json(exp2, "experiment2_domain_analysis.json")


# ─────────────────────────────────────────────────────────────────────────────
# DENEY 3: SORU TİPİ ANALİZİ (2-hop / 3-hop / comparison)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  DENEY 3: Soru Tipi Analizi")
print("=" * 70)


exp3 = {}
for diff in ["2-hop", "3-hop", "comparison"]:
    subset = [r for r in all_data if r.get("difficulty") == diff]
    if not subset:
        continue
    exp3[diff] = {}
    for m in METHODS:
        exp3[diff][m] = _compute_metrics(subset, m)

print(f"\n  {'Tip':12} | {'Yöntem':20} | {'Toplam':>6} | "
      f"{'Acc':>7} | {'F1':>7} | {'EM':>7}")
print("  " + "─" * 72)
for diff in ["2-hop", "3-hop", "comparison"]:
    if diff not in exp3:
        continue
    for m in METHODS:
        s = exp3[diff][m]
        if s["total"] == 0:
            continue
        print(f"  {diff:12} | {METHOD_LABELS[m]:20} | {s['total']:>6} | "
              f"{s['accuracy']:>7.2%} | {s['f1']:>7.4f} | {s['exact_match']:>7.2%}")
    print("  " + "─" * 72)

save_json(exp3, "experiment3_questiontype_analysis.json")


# ─────────────────────────────────────────────────────────────────────────────
# HATA ANALİZİ
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  HATA ANALİZİ")
print("=" * 70)

# Kategori tespiti
def _categorize_error(err: dict) -> str:
    pred  = _normalize(err["predicted"])
    gold  = _normalize(err["gold"])
    q     = _normalize(err["question_text"])

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

kg_errors = exp1["kg_rag"]["errors"]
cat_counts = collections.Counter(_categorize_error(e) for e in kg_errors)

print(f"\n  KG-Infused RAG hataları: {len(kg_errors)} soru")
print(f"\n  Hata kategorileri:")
for cat, cnt in cat_counts.most_common():
    pct = cnt / len(kg_errors) * 100 if kg_errors else 0
    bar = "█" * min(cnt, 20)
    print(f"    {cat:20} : {cnt:>4}  ({pct:5.1f}%)  {bar}")


# Domain bazlı hatalar (sadece football ve cinema)
domain_errors = collections.Counter(e["domain"] for e in kg_errors if e["domain"] in ("football", "cinema"))
print(f"\n  Domain bazlı hatalar (KG-RAG):")
for domain, cnt in domain_errors.most_common():
    print(f"    {domain:15} : {cnt}")

# Zorluk bazlı hatalar
diff_errors = collections.Counter(e["difficulty"] for e in kg_errors)
print(f"\n  Zorluk bazlı hatalar (KG-RAG):")
for diff, cnt in diff_errors.most_common():
    print(f"    {diff:15} : {cnt}")

# Tüm metodlardan hata örnekleri
sample_errors = []
for m in METHODS:
    for e in exp1[m]["errors"][:3]:
        e_copy = dict(e)
        sample_errors.append(e_copy)

error_report = {
    "kg_rag_total_errors":    len(kg_errors),
    "kg_rag_error_categories": dict(cat_counts),
    "kg_rag_domain_errors":   dict(domain_errors),
    "kg_rag_difficulty_errors": dict(diff_errors),
    "sample_errors_by_method": sample_errors,
    "analysis": {
        "most_common_error":   cat_counts.most_common(1)[0][0] if cat_counts else "N/A",
        "hardest_difficulty":  diff_errors.most_common(1)[0][0] if diff_errors else "N/A",
        "weakest_domain":      domain_errors.most_common(1)[0][0] if domain_errors else "N/A",
    }
}
save_json(error_report, "error_analysis.json")


# ─────────────────────────────────────────────────────────────────────────────
# GRAFİKLER
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  GRAFİKLER oluşturuluyor...")
print("=" * 70)

COLORS = {
    "no_retrieval": "#e74c3c",
    "vanilla_rag":  "#e67e22",
    "vanilla_qe":   "#3498db",
    "kg_rag":       "#2ecc71",
}

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Phase 5: KG-Infused RAG Deney Sonuçları",
             fontsize=16, fontweight="bold")

# ── Graf 1: Accuracy karşılaştırma (tüm sorular) ──
ax = axes[0, 0]
labels = [METHOD_LABELS[m] for m in METHODS]
accs   = [exp1[m]["accuracy"] for m in METHODS]
bars   = ax.bar(labels, accs, color=[COLORS[m] for m in METHODS], edgecolor="white")
ax.set_title("Accuracy Karşılaştırması (Tüm Sorular)", fontweight="bold")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=15)
for bar, v in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{v:.1%}", ha="center", fontsize=9, fontweight="bold")

# ── Graf 2: F1 karşılaştırma ──
ax = axes[0, 1]
f1s = [exp1[m]["f1"] for m in METHODS]
bars = ax.bar(labels, f1s, color=[COLORS[m] for m in METHODS], edgecolor="white")
ax.set_title("F1 Score Karşılaştırması", fontweight="bold")
ax.set_ylabel("F1")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=15)
for bar, v in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

# ── Graf 3: Retrieval Recall ──
ax = axes[0, 2]
recalls = []
r_labels = []
for m in METHODS:
    rr = exp1[m]["retrieval_recall"]
    if rr is not None:
        recalls.append(rr)
        r_labels.append(METHOD_LABELS[m])
if recalls:
    bars = ax.bar(r_labels, recalls,
                  color=[COLORS[m] for m in METHODS if exp1[m]["retrieval_recall"] is not None],
                  edgecolor="white")
    ax.set_title("Retrieval Recall", fontweight="bold")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    for bar, v in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.1%}", ha="center", fontsize=9, fontweight="bold")
else:
    ax.text(0.5, 0.5, "Pipeline çalıştırıldıktan\nsonra görünür",
            ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_title("Retrieval Recall", fontweight="bold")

# ── Graf 4: Domain bazlı KG-RAG performans ──
ax = axes[1, 0]
domains = list(exp2.keys())
if domains:
    x     = np.arange(len(domains))
    width = 0.2
    metrics_to_plot = ["accuracy", "f1", "exact_match"]
    metric_colors   = ["#2ecc71", "#3498db", "#e74c3c"]
    for i, (metric, color) in enumerate(zip(metrics_to_plot, metric_colors)):
        vals = [exp2[d]["kg_rag"].get(metric, 0) for d in domains]
        ax.bar(x + i * width, vals, width, label=metric, color=color, edgecolor="white")
    ax.set_title("Domain Bazlı KG-Infused RAG", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.capitalize() for d in domains])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "Domain verisi yok", ha="center", va="center",
            transform=ax.transAxes)
    ax.set_title("Domain Bazlı KG-Infused RAG", fontweight="bold")

# ── Graf 5: Soru tipi bazlı KG-RAG ──
ax = axes[1, 1]
diff_types = [d for d in ["2-hop", "3-hop", "comparison"] if d in exp3]
if diff_types:
    x     = np.arange(len(diff_types))
    width = 0.2
    for i, m in enumerate(METHODS):
        vals = [exp3[d][m]["accuracy"] if d in exp3 and exp3[d][m]["total"] > 0 else 0
                for d in diff_types]
        ax.bar(x + i * width, vals, width, label=METHOD_LABELS[m],
               color=COLORS[m], edgecolor="white")
    ax.set_title("Soru Tipi Bazlı Accuracy", fontweight="bold")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(diff_types)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
else:
    ax.text(0.5, 0.5, "Soru tipi verisi yok", ha="center", va="center",
            transform=ax.transAxes)
    ax.set_title("Soru Tipi Bazlı Accuracy", fontweight="bold")

# ── Graf 6: Hata kategorileri (pie) ──
ax = axes[1, 2]
if cat_counts:
    sizes  = list(cat_counts.values())
    clabels = list(cat_counts.keys())
    pie_colors = ["#e74c3c", "#e67e22", "#3498db", "#9b59b6", "#1abc9c"]
    ax.pie(sizes, labels=clabels, autopct="%1.1f%%",
           startangle=140, colors=pie_colors[:len(sizes)])
    ax.set_title("KG-RAG Hata Kategorileri", fontweight="bold")
else:
    ax.text(0.5, 0.5, "Hata verisi yok", ha="center", va="center",
            transform=ax.transAxes)
    ax.set_title("KG-RAG Hata Kategorileri", fontweight="bold")

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "phase5_charts.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✔ Grafik kaydedildi: {chart_path}")


# ─────────────────────────────────────────────────────────────────────────────
# METİN RAPORU
# ─────────────────────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 70)
lines.append("  PHASE 5 — DENEY SONUÇLARI RAPORU")
lines.append("=" * 70)
lines.append(f"\n  Toplam soru        : {len(all_data)}")
lines.append(f"  Football           : {len(football_data)}")
lines.append(f"  Cinema             : {len(cinema_data)}")

lines.append("\n" + "─" * 70)
lines.append("  DENEY 1: 4-Yöntem Karşılaştırması")
lines.append("─" * 70)
lines.append(f"  {'Yöntem':20} | {'Acc':>7} | {'F1':>7} | {'EM':>7} | {'Ret.Recall':>10}")
lines.append("  " + "─" * 58)
for m in METHODS:
    s  = exp1[m]
    rr = s["retrieval_recall"]
    rr_str = f"{rr:>10.2%}" if rr is not None else f"{'N/A':>10}"
    lines.append(f"  {METHOD_LABELS[m]:20} | {s['accuracy']:>7.2%} | "
                 f"{s['f1']:>7.4f} | {s['exact_match']:>7.2%} | {rr_str}")

# KG-RAG vs Vanilla RAG karşılaştırması
kg_acc  = exp1["kg_rag"]["accuracy"]
vr_acc  = exp1["vanilla_rag"]["accuracy"]
nor_acc = exp1["no_retrieval"]["accuracy"]
improvement_vr  = (kg_acc - vr_acc)  * 100
improvement_nor = (kg_acc - nor_acc) * 100
lines.append(f"\n  KG-RAG vs Vanilla RAG  : {improvement_vr:+.1f}% puan")
lines.append(f"  KG-RAG vs No-Retrieval : {improvement_nor:+.1f}% puan")

lines.append("\n" + "─" * 70)
lines.append("  DENEY 2: Domain Analizi")
lines.append("─" * 70)
for domain in exp2:
    kg_d = exp2[domain]["kg_rag"]
    lines.append(f"\n  {domain.upper():}")
    lines.append(f"    KG-RAG → Acc:{kg_d['accuracy']:.2%}  "
                 f"F1:{kg_d['f1']:.4f}  EM:{kg_d['exact_match']:.2%}")

lines.append("\n" + "─" * 70)
lines.append("  DENEY 3: Soru Tipi Analizi (KG-Infused RAG)")
lines.append("─" * 70)
for diff in ["2-hop", "3-hop", "comparison"]:
    if diff in exp3 and exp3[diff]["kg_rag"]["total"] > 0:
        d = exp3[diff]["kg_rag"]
        lines.append(f"  {diff:12} → Acc:{d['accuracy']:.2%}  "
                     f"F1:{d['f1']:.4f}  EM:{d['exact_match']:.2%}  "
                     f"({d['total']} soru)")

lines.append("\n" + "─" * 70)
lines.append("  HATA ANALİZİ (KG-Infused RAG)")
lines.append("─" * 70)
lines.append(f"  Toplam hata: {len(kg_errors)}")
for cat, cnt in cat_counts.most_common():
    lines.append(f"    {cat:20} : {cnt} ({cnt/len(kg_errors)*100:.1f}%)" if kg_errors else "")

lines.append("\n" + "─" * 70)
lines.append("  6.5 ANALİZ SORULARI")
lines.append("─" * 70)
lines.append(f"  1. En başarılı domain  : {best_domain or 'N/A'} (Acc={best_acc:.2%})")
lines.append(f"  2. KG vs Vanilla RAG farkı : {improvement_vr:+.1f}% → "
             f"{'KG daha iyi ✅' if improvement_vr > 0 else 'KG daha kötü ⚠️'}")
lines.append(f"  3. En zor soru tipi   : "
             f"{max(diff_errors, key=diff_errors.get) if diff_errors else 'N/A'} "
             f"(en fazla hata)")
lines.append(f"  4. En yaygın hata tipi: "
             f"{cat_counts.most_common(1)[0][0] if cat_counts else 'N/A'}")

lines.append("\n" + "=" * 70)
lines.append("  Çıktılar: outputs/phase5/ klasöründe")
lines.append("=" * 70)

report_text = "\n".join(lines)
report_path = os.path.join(OUTPUT_DIR, "phase5_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"  ✔ Rapor kaydedildi: {report_path}")

print(report_text)
print("\n✅ Phase 5 tamamlandı!")
