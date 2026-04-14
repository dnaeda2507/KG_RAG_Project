"""
Pipeline - KG-Infused RAG Ana Orchestrator
==========================================
3 modülü sırasıyla çalıştırır ve tam sonuç üretir.
4 yöntemi karşılaştırmalı olarak test eder:
  1. no_retrieval   (NoR)
  2. vanilla_rag
  3. vanilla_qe
  4. kg_rag         (Ana yöntem)

Düzeltme:
  - retrieval_recall hesabında comparison soruları artık doğru atlanıyor.
    gold.startswith("COMPARISON::") kontrolü eklendi — Phase 3'te üretilen
    karşılaştırma sorularının gold_answer formatıyla tutarlı.
  - Diğer düzeltmeler korundu (lazy initialization, F1/EM metrikleri, retry).

Kullanım:
    python pipeline.py --query "Galatasaray'ın teknik direktörünün doğum yeri neresidir?"
    python pipeline.py --dataset outputs/phase3_football/qa_dataset.json --method all
    python pipeline.py --dataset outputs/phase3_football/qa_dataset.json --method kg_rag
"""

import os
import json
import time
import argparse
import functools
import wikipedia
from groq import Groq
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    GROQ_MODEL, GROQ_API_KEY,
    K_P, MAX_PASSAGE_CHARS,
    WIKI_SEARCH_DELAY, WIKI_PAGE_DELAY,
)

wikipedia.set_lang("en")


def retry_on_failure(max_retries: int = 3, delay: float = 2.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    wait = delay * (attempt + 1)
                    print(f"  [Retry] {func.__name__} hata (deneme {attempt+1}/{max_retries}): "
                          f"{e} — {wait:.1f}s bekleniyor.")
                    time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════

@retry_on_failure(max_retries=3, delay=2.0)
def _groq_call(groq_client: Groq, prompt: str, max_tokens: int = 200) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _wiki_search(query: str, n: int = K_P) -> list:
    passages = []
    try:
        results = wikipedia.search(query, results=n + 2)
        time.sleep(WIKI_SEARCH_DELAY)
    except Exception:
        return []

    for title in results:
        if len(passages) >= n:
            break
        try:
            page    = wikipedia.page(title, auto_suggest=False)
            content = page.content[:MAX_PASSAGE_CHARS].strip()
            if len(content) >= 50:
                passages.append({"title": page.title, "content": content})
            time.sleep(WIKI_PAGE_DELAY)
        except wikipedia.exceptions.DisambiguationError as e:
            if e.options:
                try:
                    page    = wikipedia.page(e.options[0], auto_suggest=False)
                    content = page.content[:MAX_PASSAGE_CHARS].strip()
                    if len(content) >= 50:
                        passages.append({"title": page.title, "content": content})
                    time.sleep(WIKI_PAGE_DELAY)
                except Exception:
                    pass
            continue
        except Exception:
            continue

    return passages


# ══════════════════════════════════════════════════════════════════════════════
# DEĞERLENDİRME METRİKLERİ
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    return text.lower().strip()


def _exact_match(prediction: str, gold: str) -> bool:
    return _normalize(prediction) == _normalize(gold)


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = set(_normalize(prediction).split())
    gold_tokens = set(_normalize(gold).split())

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = pred_tokens & gold_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _soft_accuracy(prediction: str, gold: str) -> bool:
    p = _normalize(prediction)
    g = _normalize(gold)
    return (g in p) or (p in g and len(p) > 3)


# ✔ DÜZELTME: comparison sorusu tespit yardımcısı
# Phase 3'te gold_answer "COMPARISON::" prefix'i ile üretiliyor.
def _is_comparison_question(gold: str) -> bool:
    """
    Phase 3'te comparison sorularının gold_answer'ı "COMPARISON::" ile başlar.
    Eski format uyumluluğu için "karşılaştırma" da kontrol edilir.
    """
    g = gold.strip().lower()
    return g.startswith("comparison::") or g.startswith("karşılaştırma")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class Pipeline:

    def __init__(self):
        print("\n" + "=" * 60)
        print("KG-Infused RAG Pipeline başlatılıyor...")
        print("=" * 60)
        self.groq = Groq(api_key=GROQ_API_KEY)

        self._m1 = None
        self._m2 = None
        self._m3 = None
        print("✔ Pipeline hazır (modüller lazy yükleniyor).\n")

    @property
    def m1(self):
        if self._m1 is None:
            from module1_spreading_activation import SpreadingActivation
            self._m1 = SpreadingActivation()
        return self._m1

    @property
    def m2(self):
        if self._m2 is None:
            from module2_query_expansion import QueryExpansion
            self._m2 = QueryExpansion()
        return self._m2

    @property
    def m3(self):
        if self._m3 is None:
            from module3_answer_generation import AnswerGeneration
            self._m3 = AnswerGeneration()
        return self._m3

    def no_retrieval(self, query: str) -> dict:
        print(f"\n[NoR] {query[:60]}...")
        prompt = (
            f"Answer the following question based on your knowledge.\n"
            f"Question: {query}\n"
            f"Give a direct, concise answer (1-2 sentences).\nAnswer:"
        )
        try:
            answer = _groq_call(self.groq, prompt)
            if answer.lower().startswith("answer:"):
                answer = answer[7:].strip()
        except Exception as e:
            print(f"  [NoR] Hata: {e}")
            answer = ""

        return {
            "method":       "no_retrieval",
            "query":        query,
            "final_answer": answer,
            "passages":     [],
            "kg_summary":   "",
        }

    def vanilla_rag(self, query: str) -> dict:
        print(f"\n[VanillaRAG] {query[:60]}...")
        passages = _wiki_search(query, n=K_P)

        passages_text = "\n".join(
            f"[{p['title']}] {p['content']}" for p in passages
        )
        prompt = (
            f"Answer the following question using the provided passages.\n"
            f"Question: {query}\n\nPassages:\n{passages_text}\n\n"
            f"Give a direct, concise answer (1-2 sentences).\nAnswer:"
        )
        try:
            answer = _groq_call(self.groq, prompt)
            if answer.lower().startswith("answer:"):
                answer = answer[7:].strip()
        except Exception as e:
            print(f"  [VanillaRAG] Hata: {e}")
            answer = ""

        return {
            "method":       "vanilla_rag",
            "query":        query,
            "final_answer": answer,
            "passages":     passages,
            "kg_summary":   "",
        }

    def vanilla_qe(self, query: str) -> dict:
        print(f"\n[VanillaQE] {query[:60]}...")

        expand_prompt = (
            f"Generate an expanded search query for Wikipedia to help answer:\n"
            f"Question: {query}\n"
            f"Return ONLY the expanded query, nothing else."
        )
        try:
            expanded = _groq_call(self.groq, expand_prompt, max_tokens=100)
        except Exception:
            expanded = query

        p_orig = _wiki_search(query,    n=K_P // 2)
        p_exp  = _wiki_search(expanded, n=K_P // 2)

        seen, passages = set(), []
        for p in p_orig + p_exp:
            if p["title"] not in seen:
                seen.add(p["title"])
                passages.append(p)

        passages_text = "\n".join(
            f"[{p['title']}] {p['content']}" for p in passages
        )
        prompt = (
            f"Answer the following question using the provided passages.\n"
            f"Question: {query}\n\nPassages:\n{passages_text}\n\n"
            f"Give a direct, concise answer (1-2 sentences).\nAnswer:"
        )
        try:
            answer = _groq_call(self.groq, prompt)
            if answer.lower().startswith("answer:"):
                answer = answer[7:].strip()
        except Exception as e:
            print(f"  [VanillaQE] Hata: {e}")
            answer = ""

        return {
            "method":         "vanilla_qe",
            "query":          query,
            "expanded_query": expanded,
            "final_answer":   answer,
            "passages":       passages,
            "kg_summary":     "",
        }

    def kg_rag(self, query: str) -> dict:
        print(f"\n[KG-RAG] {query[:60]}...")

        m1_result = self.m1.run(query)
        m2_result = self.m2.run(
            query      = query,
            kg_summary = m1_result["kg_summary"],
        )
        m3_result = self.m3.run(
            query      = query,
            passages   = m2_result["passages"],
            kg_summary = m1_result["kg_summary"],
        )

        return {
            "method":             "kg_rag",
            "query":              query,
            "seed_entities":      m1_result["seed_entities"],
            "activated_triples":  m1_result["activated_triples"],
            "kg_summary":         m1_result["kg_summary"],
            "expanded_query":     m2_result["expanded_query"],
            "passages":           m2_result["passages"],
            "passage_note":       m3_result["passage_note"],
            "fact_enhanced_note": m3_result["fact_enhanced_note"],
            "final_answer":       m3_result["final_answer"],
            "reasoning_chain":    m3_result["reasoning_chain"],
        }

    def run_single(self, query: str, method: str = "kg_rag") -> dict:
        methods = {
            "no_retrieval": self.no_retrieval,
            "vanilla_rag":  self.vanilla_rag,
            "vanilla_qe":   self.vanilla_qe,
            "kg_rag":       self.kg_rag,
        }

        if method == "all":
            results = {}
            for name, fn in methods.items():
                results[name] = fn(query)
                time.sleep(1)
            return results

        if method not in methods:
            raise ValueError(f"Geçersiz method: {method}. Seçenekler: {list(methods.keys())}")

        return methods[method](query)

    def run_dataset(
        self,
        dataset_path:  str,
        method:        str = "kg_rag",
        max_questions: int = 50,
        output_dir:    str = "outputs/phase4",
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)

        with open(dataset_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        questions = questions[:max_questions]
        print(f"\n[Pipeline] {len(questions)} soru işlenecek | Yöntem: {method}")
        print("=" * 60)

        methods_to_run = (
            ["no_retrieval", "vanilla_rag", "vanilla_qe", "kg_rag"]
            if method == "all"
            else [method]
        )

        all_results = []

        for i, q in enumerate(questions, 1):
            q_id   = q.get("question_id", f"Q{i}")
            query  = q.get("question_text", "")
            gold   = q.get("gold_answer",   "")
            diff   = q.get("difficulty",    "")
            domain = q.get("domain",        "")

            print(f"\n[{i}/{len(questions)}] {q_id} | {diff} | {query[:55]}...")

            row = {
                "question_id":    q_id,
                "question_text":  query,
                "gold_answer":    gold,
                "difficulty":     diff,
                "domain":         domain,
                "reasoning_path": q.get("reasoning_path", []),
                "answers":        {},
                "passages":       {},
            }

            for m in methods_to_run:
                try:
                    result = getattr(self, m)(query)
                    row["answers"][m]  = result.get("final_answer", "")
                    row["passages"][m] = result.get("passages", [])
                    print(f"  [{m:15}] → {str(row['answers'][m])[:60]}")
                except Exception as e:
                    print(f"  [{m:15}] ⚠ Hata: {e}")
                    row["answers"][m]  = ""
                    row["passages"][m] = []
                time.sleep(0.5)

            all_results.append(row)

            if i % 10 == 0:
                _save_json(all_results,
                           os.path.join(output_dir, "pipeline_results_partial.json"))
                print(f"  [Checkpoint] {i} soru kaydedildi.")

        _save_json(all_results, os.path.join(output_dir, "pipeline_results.json"))

        summary = self._build_summary(all_results, methods_to_run)
        _save_json(summary, os.path.join(output_dir, "pipeline_summary.json"))

        self._print_summary(summary)
        return {"results": all_results, "summary": summary}

    def _build_summary(self, results: list, methods: list) -> dict:
        summary = {
            "total_questions": len(results),
            "methods":         {},
        }

        for m in methods:
            acc_correct      = 0
            em_correct       = 0
            f1_total         = 0.0
            recall_correct   = 0
            total            = 0
            total_retrieving = 0
            by_diff          = {}

            for r in results:
                gold   = str(r.get("gold_answer", "")).strip()
                answer = str(r.get("answers", {}).get(m, "")).strip()
                diff   = r.get("difficulty", "unknown")

                # ✔ DÜZELTME: comparison soruları _is_comparison_question() ile atlanıyor
                # Eski kod sadece "karşılaştırma" prefix'ini kontrol ediyordu;
                # yeni Phase 3 çıktısında "COMPARISON::" prefix'i var.
                # İki format da destekleniyor.
                if not gold or _is_comparison_question(gold):
                    continue

                total += 1

                hit_acc = _soft_accuracy(answer, gold)
                hit_em  = _exact_match(answer, gold)
                f1      = _token_f1(answer, gold)

                if hit_acc:
                    acc_correct += 1
                if hit_em:
                    em_correct += 1
                f1_total += f1

                # Retrieval Recall
                passages = r.get("passages", {}).get(m, [])
                if passages:
                    total_retrieving += 1
                    gold_norm    = gold.lower()
                    passage_text = " ".join(
                        p.get("content", "").lower() for p in passages
                    )
                    if gold_norm in passage_text or any(
                        token in passage_text
                        for token in gold_norm.split()
                        if len(token) > 3
                    ):
                        recall_correct += 1

                if diff not in by_diff:
                    by_diff[diff] = {
                        "acc_correct": 0, "em_correct": 0,
                        "f1_total": 0.0, "recall_correct": 0,
                        "recall_total": 0, "total": 0
                    }
                by_diff[diff]["total"]    += 1
                by_diff[diff]["f1_total"] += f1
                if hit_acc:
                    by_diff[diff]["acc_correct"] += 1
                if hit_em:
                    by_diff[diff]["em_correct"]  += 1
                if passages:
                    by_diff[diff]["recall_total"] += 1
                    gold_norm    = gold.lower()
                    passage_text = " ".join(
                        p.get("content", "").lower() for p in passages
                    )
                    if gold_norm in passage_text or any(
                        token in passage_text
                        for token in gold_norm.split()
                        if len(token) > 3
                    ):
                        by_diff[diff]["recall_correct"] += 1

            acc_score    = round(acc_correct    / total,            4) if total            > 0 else 0.0
            em_score     = round(em_correct     / total,            4) if total            > 0 else 0.0
            f1_score     = round(f1_total       / total,            4) if total            > 0 else 0.0
            recall_score = round(recall_correct / total_retrieving, 4) if total_retrieving > 0 else None

            summary["methods"][m] = {
                "total":            total,
                "accuracy":         acc_score,
                "exact_match":      em_score,
                "f1":               f1_score,
                "retrieval_recall": recall_score,
                "by_difficulty": {
                    d: {
                        "accuracy":         round(v["acc_correct"]    / v["total"],        4) if v["total"]        > 0 else 0,
                        "exact_match":      round(v["em_correct"]     / v["total"],        4) if v["total"]        > 0 else 0,
                        "f1":               round(v["f1_total"]       / v["total"],        4) if v["total"]        > 0 else 0,
                        "retrieval_recall": round(v["recall_correct"] / v["recall_total"], 4) if v["recall_total"] > 0 else None,
                        **{k: val for k, val in v.items() if k not in ("f1_total",)},
                    }
                    for d, v in by_diff.items()
                },
            }

        return summary

    def _print_summary(self, summary: dict):
        print("\n" + "=" * 80)
        print("  PIPELINE SONUÇ ÖZETİ")
        print("=" * 80)
        print(f"  Toplam soru: {summary['total_questions']}")
        print(f"\n  {'Yöntem':20} | {'Toplam':>6} | {'Accuracy':>8} | {'F1':>8} | {'EM':>8} | {'Ret.Recall':>10}")
        print("  " + "─" * 72)
        for m, stats in summary["methods"].items():
            rr = stats.get("retrieval_recall")
            rr_str = f"{rr:>10.2%}" if rr is not None else f"{'N/A':>10}"
            print(f"  {m:20} | {stats['total']:>6} | "
                  f"{stats['accuracy']:>8.2%} | "
                  f"{stats['f1']:>8.4f} | "
                  f"{stats['exact_match']:>8.2%} | "
                  f"{rr_str}")

        print("\n  Zorluk bazlı (KG-RAG):")
        if "kg_rag" in summary["methods"]:
            for diff, dstats in summary["methods"]["kg_rag"]["by_difficulty"].items():
                rr = dstats.get("retrieval_recall")
                rr_str = f"{rr:.2%}" if rr is not None else "N/A"
                print(f"    {diff:12} → Acc:{dstats['accuracy']:.2%}  "
                      f"F1:{dstats['f1']:.4f}  "
                      f"EM:{dstats['exact_match']:.2%}  "
                      f"Recall:{rr_str}  "
                      f"({dstats['acc_correct']}/{dstats['total']})")

    def close(self):
        if self._m1 is not None:
            self._m1.close()


def _save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✔ Kaydedildi: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG-Infused RAG Pipeline")
    parser.add_argument("--query",   type=str, default=None,
                        help="Tek soru çalıştır")
    parser.add_argument("--dataset", type=str, default=None,
                        help="QA dataset JSON dosya yolu")
    parser.add_argument("--method",  type=str, default="kg_rag",
                        choices=["no_retrieval", "vanilla_rag",
                                 "vanilla_qe", "kg_rag", "all"],
                        help="Kullanılacak yöntem")
    parser.add_argument("--max_q",   type=int, default=50,
                        help="Dataset modunda max soru sayısı")
    parser.add_argument("--output",  type=str, default="outputs/phase4",
                        help="Çıktı klasörü")
    args = parser.parse_args()

    pipe = Pipeline()

    try:
        if args.query:
            print(f"\nSorgu: {args.query}")
            result = pipe.run_single(args.query, method=args.method)

            if args.method == "all":
                for m, r in result.items():
                    print(f"\n[{m}] Cevap: {r.get('final_answer', '')}")
            else:
                print(f"\nCevap: {result.get('final_answer', '')}")

            os.makedirs(args.output, exist_ok=True)
            _save_json(result, os.path.join(args.output, "single_query_result.json"))

        elif args.dataset:
            pipe.run_dataset(
                dataset_path  = args.dataset,
                method        = args.method,
                max_questions = args.max_q,
                output_dir    = args.output,
            )

        else:
            demo_query = "Where was the coach of Galatasaray born?"
            print(f"\nDemo sorgu: {demo_query}")
            result = pipe.run_single(demo_query, method="kg_rag")
            print(f"\nFinal Cevap: {result.get('final_answer', '')}")

    finally:
        pipe.close()
