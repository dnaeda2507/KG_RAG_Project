"""
Module 2 - KG-Based Query Expansion & Wikipedia Retrieval
==========================================================
Module 1'den gelen KG özetini kullanarak sorguyu genişletir,
orijinal + genişletilmiş sorgu ile Wikipedia'dan passage çeker.

Düzeltmeler:
  - Wikipedia arama sorgusuna da rate limit eklendi
  - DisambiguationError sonrası tekrar sayım hatası giderildi
  - LLM çağrısına retry mekanizması eklendi
"""

import os
import time
import functools
import wikipedia
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Parametreler ──────────────────────────────────────────────────────────────
K_P             = 6      # Toplam passage sayısı
MAX_PASSAGE_LEN = 500    # Her passage'ın maksimum karakter uzunluğu
WIKI_LANGUAGE   = "en"
GROQ_MODEL      = "llama-3.1-8b-instant"
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
WIKI_SEARCH_DELAY = 0.4  # Arama sonrası bekleme (saniye)
WIKI_PAGE_DELAY   = 0.2  # Sayfa çekimi sonrası bekleme (saniye)

wikipedia.set_lang(WIKI_LANGUAGE)


# ── Retry decorator ───────────────────────────────────────────────────────────
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


class QueryExpansion:
    """
    KG-Infused RAG — Modül 2
    KG özeti ile sorguyu genişletir ve Wikipedia'dan passage çeker.
    """

    def __init__(self):
        print("[Module2] Başlatılıyor...")
        self.groq = Groq(api_key=GROQ_API_KEY)
        print("[Module2] ✔ Groq hazır.")

    # ──────────────────────────────────────────────────────────────────────────
    # ADIM 1: QUERY EXPANSION
    # ──────────────────────────────────────────────────────────────────────────

    @retry_on_failure(max_retries=3, delay=2.0)
    def expand_query(self, query: str, kg_summary: str) -> str:
        """
        Orijinal sorgu + KG özeti → LLM → genişletilmiş sorgu üretir.
        """
        print("\n[Step 1] Query expansion yapılıyor...")

        prompt = f"""You are a query expansion assistant for a retrieval system.

Original Query: {query}

Knowledge Graph Summary:
{kg_summary}

Based on the knowledge graph information, generate an EXPANDED search query
that complements the original query. The expanded query should:
- Include specific entity names mentioned in the KG summary
- Be suitable for Wikipedia search
- Be concise (1-2 sentences max)
- Help retrieve passages that answer the original query

Return ONLY the expanded query text, nothing else."""

        response = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150,
        )
        expanded = response.choices[0].message.content.strip()
        print(f"  Orijinal     : {query}")
        print(f"  Genişletilmiş: {expanded}")
        return expanded

    # ──────────────────────────────────────────────────────────────────────────
    # ADIM 2: WİKİPEDİA PASSAGE ÇEKME
    # ──────────────────────────────────────────────────────────────────────────

    def _search_wikipedia(self, query: str, n_passages: int) -> list:
        """
        Wikipedia'da arama yapar ve passage listesi döndürür.
        Her passage: {"title": str, "content": str, "query_source": str}
        """
        passages = []

        try:
            search_results = wikipedia.search(query, results=n_passages + 2)
            time.sleep(WIKI_SEARCH_DELAY)   # ✔ arama sonrası rate limit bekleme
        except Exception as e:
            print(f"  [Wiki] Arama hatası: {e}")
            return []

        for title in search_results:
            if len(passages) >= n_passages:
                break

            try:
                page    = wikipedia.page(title, auto_suggest=False)
                content = page.content[:MAX_PASSAGE_LEN].strip()

                if len(content) >= 50:
                    passages.append({
                        "title":        page.title,
                        "content":      content,
                        "url":          page.url,
                        "query_source": query,
                    })
                time.sleep(WIKI_PAGE_DELAY)

            except wikipedia.exceptions.DisambiguationError as e:
                # ✔ Disambiguation: ilk seçeneği dene, sonra bir sonraki başlığa geç
                if not e.options:
                    continue
                try:
                    page    = wikipedia.page(e.options[0], auto_suggest=False)
                    content = page.content[:MAX_PASSAGE_LEN].strip()
                    if len(content) >= 50:
                        passages.append({
                            "title":        page.title,
                            "content":      content,
                            "url":          page.url,
                            "query_source": query,
                        })
                    time.sleep(WIKI_PAGE_DELAY)
                except Exception:
                    pass   # disambiguation başarısız → bir sonraki başlığa geç
                continue   # ✔ eklendi — ana döngü devam eder

            except wikipedia.exceptions.PageError:
                continue
            except Exception as e:
                print(f"  [Wiki] Sayfa hatası ({title}): {e}")
                continue

        return passages

    def retrieve_passages(
        self, original_query: str, expanded_query: str
    ) -> list:
        """
        Dual-query retrieval:
        - Orijinal sorgu ile K_P/2 passage
        - Genişletilmiş sorgu ile K_P/2 passage
        Toplam K_P passage döndürür (tekrarlar temizlenir).
        """
        print(f"\n[Step 2] Wikipedia'dan passage çekiliyor (K_p={K_P})...")
        n_each = K_P // 2

        print(f"  Orijinal sorgu ile {n_each} passage aranıyor...")
        passages_orig = self._search_wikipedia(original_query, n_each)

        print(f"  Genişletilmiş sorgu ile {n_each} passage aranıyor...")
        passages_exp  = self._search_wikipedia(expanded_query, n_each)

        # Tekrar eden başlıkları temizle (orijinal öncelikli)
        seen_titles  = set()
        all_passages = []
        for p in passages_orig + passages_exp:
            if p["title"] not in seen_titles:
                seen_titles.add(p["title"])
                all_passages.append(p)

        print(f"  ✔ Toplam {len(all_passages)} unique passage çekildi.")
        for i, p in enumerate(all_passages, 1):
            print(f"    {i}. [{p['query_source'][:30]:30}] → {p['title']}")

        return all_passages

    # ──────────────────────────────────────────────────────────────────────────
    # ANA FONKSİYON
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, query: str, kg_summary: str) -> dict:
        """
        Query expansion + Wikipedia retrieval pipeline'ını çalıştırır.

        Args:
            query:      Orijinal doğal dil sorusu
            kg_summary: Module 1'den gelen KG subgraph özeti

        Returns:
            {
                "original_query":  str,
                "expanded_query":  str,
                "passages":        list,
                "total_passages":  int
            }
        """
        print("\n" + "=" * 60)
        print("[Module 2] Query Expansion & Retrieval")
        print(f"Query: {query}")
        print("=" * 60)

        try:
            expanded_query = self.expand_query(query, kg_summary)
        except Exception as e:
            print(f"  [Module2] Query expansion başarısız: {e} — orijinal sorgu kullanılıyor.")
            expanded_query = query

        passages = self.retrieve_passages(query, expanded_query)

        result = {
            "original_query": query,
            "expanded_query": expanded_query,
            "passages":       passages,
            "total_passages": len(passages),
        }

        print("\n[Module 2] ✅ Tamamlandı.")
        return result


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_query      = "Galatasaray'ın teknik direktörünün doğum yeri neresidir?"
    test_kg_summary = (
        "Galatasaray F.C. is a Turkish football club based in Istanbul. "
        "The club has had several coaches born in different cities of Turkey. "
        "Okan Buruk, a former player and coach, was born in Istanbul, Turkey."
    )

    qe     = QueryExpansion()
    result = qe.run(query=test_query, kg_summary=test_kg_summary)

    print("\n" + "─" * 60)
    print("SONUÇ:")
    print(f"  Orijinal sorgu    : {result['original_query']}")
    print(f"  Genişletilmiş     : {result['expanded_query']}")
    print(f"  Passage sayısı    : {result['total_passages']}")

    if result["passages"]:
        p = result["passages"][0]
        print(f"\nİlk passage:")
        print(f"  Başlık  : {p['title']}")
        print(f"  İçerik  : {p['content'][:200]}...")

    os.makedirs("outputs/phase4", exist_ok=True)
    with open("outputs/phase4/module2_test_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n  ✔ Test sonucu: outputs/phase4/module2_test_result.json")
