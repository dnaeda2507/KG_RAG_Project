import os
import time
import functools
import wikipedia
import logging
import re
import unicodedata
from groq import Groq
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    K_P, MAX_PASSAGE_LEN, WIKI_LANGUAGE, GROQ_MODEL, GROQ_API_KEYS,
    WIKI_SEARCH_DELAY, WIKI_PAGE_DELAY, GroqKeyRotator,
)

logging.basicConfig(
    filename='pipeline_debug.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
)


def retry_on_failure(max_retries: int = 3, delay: float = 2.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            self_obj = args[0] if args and hasattr(args[0], 'groq') else None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if '429' in str(e) and self_obj is not None and len(GROQ_API_KEYS) > 1:
                        self_obj.groq.rotate()
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

    Düzeltmeler:
      - expand_query(): LLM bazen "Expanded Query: ..." prefix'i ile
        döner. Bu prefix temizlenmeden Wikipedia'ya gönderilince
        "Unfortunately..." başlıklı sayfalar çekiliyordu. Şimdi
        tüm bilinen prefix'ler ve tırnak işaretleri temizleniyor.
    """

    def __init__(self):
        print("[Module2] Başlatılıyor...")
        self.groq = GroqKeyRotator(GROQ_API_KEYS)
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
- Never introduce unrelated entities that do not appear in the original query or KG summary
- If the query asks birth location, include terms like "birthplace" and "born in"
- If the query asks stadium location, include terms like "stadium", "home ground", "located in"

Return ONLY the expanded query text, nothing else."""

        response = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150,
        )
        expanded = response.choices[0].message.content.strip()

        # ── LLM prefix temizleme ──────────────────────────────────────────────
        # LLM bazen "Expanded Query:", "New Query:", "Search Query:" gibi
        # prefix'lerle başlıyor. Bu prefix Wikipedia aramasına giderse
        # alakasız sayfalar çekiliyor ("Unfortunately..." vakası).
        _EXPANSION_PREFIXES = (
            "expanded query:",
            "new query:",
            "search query:",
            "query:",
            "expanded search query:",
        )
        expanded_lower = expanded.lower()
        for prefix in _EXPANSION_PREFIXES:
            if expanded_lower.startswith(prefix):
                expanded = expanded[len(prefix):].strip()
                expanded_lower = expanded.lower()
                break  # tek prefix temizle, döngüden çık

        # Baştaki/sondaki tırnak işaretlerini temizle
        expanded = expanded.strip('"\'')

        # Boş kaldıysa orijinal sorguya dön
        if not expanded:
            print("  [Expand] Genişletilmiş sorgu boş — orijinal kullanılıyor.")
            expanded = query

        print(f"  Orijinal     : {query}")
        print(f"  Genişletilmiş: {expanded}")
        return expanded

    # ──────────────────────────────────────────────────────────────────────────
    # ADIM 2: WİKİPEDİA PASSAGE ÇEKME
    # ──────────────────────────────────────────────────────────────────────────

    def _clean_query(self, query: str) -> str:
        """
        Sorgudan noktalama ve özel karakterleri temizler.
        Tüm kelimeler korunur; Wikipedia arama için tam sorgu kullanılır.
        """
        cleaned = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ]", "", query)
        return cleaned.strip()

    def _normalize_ascii(self, text: str) -> str:
        if not text:
            return ""
        tr_map = str.maketrans({
            "ı": "i", "İ": "i",
            "ğ": "g", "Ğ": "g",
            "ş": "s", "Ş": "s",
            "ç": "c", "Ç": "c",
            "ö": "o", "Ö": "o",
            "ü": "u", "Ü": "u",
        })
        s = text.translate(tr_map)
        s = "".join(ch for ch in unicodedata.normalize("NFKD", s)
                    if not unicodedata.combining(ch))
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _extract_aliases(self, query: str, kg_summary: str) -> list:
        aliases = set()
        q = query.strip()

        # Common template: "Where was X born?"
        m = re.search(r"where\s+was\s+(.+?)\s+born", q, re.IGNORECASE)
        if m:
            aliases.add(m.group(1).strip())

        # Common template: "stadium of X located"
        m = re.search(r"stadium\s+of\s+(.+?)\s+located", q, re.IGNORECASE)
        if m:
            aliases.add(m.group(1).strip())

        # KG özetinden büyük harfle başlayan olası entity/isimleri çek
        kg_entities = re.findall(r'([A-ZÇĞİÖŞÜ][a-zçğıöşüA-ZÇĞİÖŞÜ\- ]{2,})', kg_summary)
        for ent in kg_entities:
            if len(ent.strip()) > 2:
                aliases.add(ent.strip())

        # Sorgunun tamamını ve normalize edilmiş halini ekle
        aliases.add(q)
        aliases.add(self._normalize_ascii(q))

        # Varyantlar: title-case, ascii, lower
        variants = set()
        for a in list(aliases):
            variants.add(a)
            variants.add(self._normalize_ascii(a))
            variants.add(a.title())
            variants.add(a.lower())

        # Kısa ve tekrar edenleri at
        out = []
        seen = set()
        for v in variants:
            vv = v.strip()
            if len(vv) < 3:
                continue
            k = vv.lower()
            if k not in seen:
                seen.add(k)
                out.append(vv)
        return out[:12]

    def _search_wikipedia(self, query: str, n_passages: int, aliases: list = None) -> list:
        """
        Wikipedia'da arama yapar ve passage listesi döndürür.
        """
        import difflib
        passages = []
        tried_queries = [query]
        if aliases:
            tried_queries.extend([a for a in aliases if a not in tried_queries])

        for q in tried_queries:
            cleaned_q = self._clean_query(q)
            logging.info(f"Wikipedia arama sorgusu: {cleaned_q}")
            try:
                search_results = wikipedia.search(cleaned_q, results=n_passages + 4)
                logging.info(f"Wikipedia arama başlıkları: {search_results}")
                time.sleep(WIKI_SEARCH_DELAY)
            except Exception as e:
                logging.error(f"[Wiki] Arama hatası: {e} | Sorgu: {cleaned_q}")
                continue

            if search_results and cleaned_q:
                close_matches = difflib.get_close_matches(cleaned_q, search_results, n=2, cutoff=0.7)
                for cm in close_matches:
                    if cm not in search_results:
                        search_results.append(cm)

            for title in search_results:
                if len(passages) >= n_passages:
                    break
                try:
                    page    = wikipedia.page(title, auto_suggest=True, redirect=True)
                    content = page.content[:MAX_PASSAGE_LEN].strip()
                    if len(content) >= 50:
                        passages.append({
                            "title":        page.title,
                            "content":      content,
                            "url":          page.url,
                            "query_source": cleaned_q,
                        })
                        logging.info(f"Passage eklendi: {page.title}")
                    else:
                        logging.info(f"Passage kısa/elendi: {page.title}")
                    time.sleep(WIKI_PAGE_DELAY)
                except wikipedia.exceptions.DisambiguationError as e:
                    if not e.options:
                        logging.info(f"Disambiguation, seçenek yok: {title}")
                        continue
                    best_option = e.options[0]
                    if len(e.options) > 1:
                        best_option = difflib.get_close_matches(cleaned_q, e.options, n=1, cutoff=0.5)
                        if best_option:
                            best_option = best_option[0]
                        else:
                            best_option = e.options[0]
                    try:
                        page    = wikipedia.page(best_option, auto_suggest=True, redirect=True)
                        content = page.content[:MAX_PASSAGE_LEN].strip()
                        if len(content) >= 50:
                            passages.append({
                                "title":        page.title,
                                "content":      content,
                                "url":          page.url,
                                "query_source": cleaned_q,
                            })
                            logging.info(f"Disambiguation passage eklendi: {page.title}")
                        else:
                            logging.info(f"Disambiguation passage kısa/elendi: {page.title}")
                        time.sleep(WIKI_PAGE_DELAY)
                    except Exception as ex:
                        logging.error(f"Disambiguation hata: {ex} | Başlık: {title}")
                    continue
                except wikipedia.exceptions.PageError:
                    logging.info(f"PageError: {title}")
                    continue
                except Exception as e:
                    logging.error(f"[Wiki] Sayfa hatası ({title}): {e}")
                    continue
            if passages:
                break  # İlk başarılı sorguda çık

        if not passages:
            logging.warning(f"Hiç passage bulunamadı. Sorgular: {tried_queries}")
        return passages

    def retrieve_passages(
        self, original_query: str, expanded_query: str, aliases: list = None
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
        passages_orig = self._search_wikipedia(original_query, n_each, aliases=aliases)

        print(f"  Genişletilmiş sorgu ile {n_each} passage aranıyor...")
        passages_exp  = self._search_wikipedia(expanded_query, n_each, aliases=aliases)

        # Tekrar eden başlıkları temizle (orijinal öncelikli)
        seen_titles  = set()
        all_passages = []
        for p in passages_orig + passages_exp:
            if p["title"] not in seen_titles:
                seen_titles.add(p["title"])
                all_passages.append(p)
            else:
                logging.info(f"Passage tekrar/elendi: {p['title']}")

        print(f"  ✔ Toplam {len(all_passages)} unique passage çekildi.")
        for i, p in enumerate(all_passages, 1):
            print(f"    {i}. [{p['query_source'][:30]:30}] → {p['title']}")

        if not all_passages:
            logging.warning(
                f"Hiç passage bulunamadı. "
                f"Orijinal: {original_query}, "
                f"Genişletilmiş: {expanded_query}, "
                f"Aliases: {aliases}"
            )

        return all_passages

    # ──────────────────────────────────────────────────────────────────────────
    # ANA FONKSİYON
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, query: str, kg_summary: str) -> dict:
        """
        Query expansion + Wikipedia retrieval pipeline'ını çalıştırır.
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

        aliases  = self._extract_aliases(query, kg_summary)
        passages = self.retrieve_passages(query, expanded_query, aliases=aliases)

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