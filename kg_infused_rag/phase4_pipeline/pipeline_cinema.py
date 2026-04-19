"""
Pipeline Cinema - KG-Infused RAG Cinema Domain
===============================================
Cinema (sinema) QA dataseti üzerinde pipeline'ı çalıştırır.
Tüm 4 yöntemi karşılaştırmalı test eder:
  1. no_retrieval   (NoR)
  2. vanilla_rag
  3. vanilla_qe
  4. kg_rag         (Ana yöntem)

Kullanım:
    python pipeline_cinema.py
    python pipeline_cinema.py --method kg_rag
    python pipeline_cinema.py --method all --max_q 100
    python pipeline_cinema.py --query "Christopher Nolan hangi ülkede doğmuştur?"
"""

import os
import sys
import argparse

# pipeline.py ile aynı dizinden import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CINEMA_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "outputs", "phase3", "qa_dataset.json"
)
CINEMA_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "outputs", "phase4_cinema"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG-Infused RAG Pipeline — Cinema Domain")
    parser.add_argument("--query",  type=str, default=None,
                        help="Tek soru çalıştır")
    parser.add_argument("--method", type=str, default="all",
                        choices=["no_retrieval", "vanilla_rag",
                                 "vanilla_qe", "kg_rag", "all"],
                        help="Kullanılacak yöntem (varsayılan: all)")
    parser.add_argument("--max_q",  type=int, default=50,
                        help="Dataset modunda max soru sayısı (varsayılan: 50)")
    parser.add_argument("--output", type=str, default=CINEMA_OUTPUT,
                        help="Çıktı klasörü")
    args = parser.parse_args()

    from pipeline import Pipeline, _save_json
    import time

    pipe = Pipeline()
    output_dir = os.path.normpath(args.output)

    try:
        if args.query:
            print(f"\nSinema Sorgusu: {args.query}")
            result = pipe.run_single(args.query, method=args.method)

            if args.method == "all":
                for m, r in result.items():
                    print(f"\n[{m}] Cevap: {r.get('final_answer', '')}")
            else:
                print(f"\nCevap: {result.get('final_answer', '')}")

            os.makedirs(output_dir, exist_ok=True)
            _save_json(result, os.path.join(output_dir, "single_query_result.json"))

        else:
            dataset_path = os.path.normpath(CINEMA_DATASET)
            if not os.path.exists(dataset_path):
                print(f"[HATA] Cinema dataset bulunamadı: {dataset_path}")
                sys.exit(1)

            print(f"\n[Cinema Pipeline]")
            print(f"  Dataset : {dataset_path}")
            print(f"  Output  : {output_dir}")
            print(f"  Method  : {args.method}")
            print(f"  Max Q   : {args.max_q}")

            pipe.run_dataset(
                dataset_path  = dataset_path,
                method        = args.method,
                max_questions = args.max_q,
                output_dir    = output_dir,
            )

    finally:
        pipe.close()