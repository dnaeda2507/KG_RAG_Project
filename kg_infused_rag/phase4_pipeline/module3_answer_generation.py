"""
Module 3 - KG-Augmented Answer Generation
==========================================
Module 2'den gelen passage'ları ve Module 1'den gelen KG özetini
birleştirerek final cevabı üretir.

3 adım:
  1. Passage Note Construction   — passage'ları sorgu odaklı nota dönüştür
  2. KG-Guided Augmentation      — nota + KG özeti → fact-enhanced nota
  3. Answer Generation           — fact-enhanced nota → final cevap

Düzeltmeler:
  - Tüm LLM çağrılarına retry mekanizması eklendi
"""

import os
import functools
import time
from groq import Groq
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import GROQ_MODEL, GROQ_API_KEY, MAX_PASSAGE_CHARS


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


class AnswerGeneration:
    """
    KG-Infused RAG — Modül 3
    Passage Note → Fact-Enhanced Note → Final Answer
    """

    def __init__(self):
        print("[Module3] Başlatılıyor...")
        self.groq = Groq(api_key=GROQ_API_KEY)
        print("[Module3] ✔ Groq hazır.")

    # ──────────────────────────────────────────────────────────────────────────
    # ADIM 1: PASSAGE NOTE CONSTRUCTION
    # ──────────────────────────────────────────────────────────────────────────

    @retry_on_failure(max_retries=3, delay=2.0)
    def build_passage_note(self, query: str, passages: list) -> str:
        """
        Retrieved passage'ları sorgu odaklı kısa bir nota dönüştürür.
        """
        print("\n[Step 1] Passage note oluşturuluyor...")

        if not passages:
            print("  ⚠ Passage bulunamadı, boş nota döndürülüyor.")
            return "No relevant passages were retrieved."

        passages_text = ""
        for i, p in enumerate(passages, 1):
            content = p.get("content", "")[:MAX_PASSAGE_CHARS]
            passages_text += f"\n[Passage {i} - {p.get('title', 'Unknown')}]\n{content}\n"

        prompt = f"""You are a reading comprehension assistant.

Query: {query}

Retrieved Passages:
{passages_text}

Read all passages carefully and write a concise note that:
- Extracts ONLY information relevant to the query
- Focuses on key facts, names, dates, and locations
- Is 3-5 sentences maximum
- Does NOT add information not present in the passages

Write the note directly, no preamble."""

        response = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=250,
        )
        note = response.choices[0].message.content.strip()
        print(f"  ✔ Passage Note ({len(note)} karakter)")
        print(f"  Preview: {note[:120]}...")
        return note

    # ──────────────────────────────────────────────────────────────────────────
    # ADIM 2: KG-GUIDED KNOWLEDGE AUGMENTATION
    # ──────────────────────────────────────────────────────────────────────────

    @retry_on_failure(max_retries=3, delay=2.0)
    def augment_with_kg(
        self, query: str, passage_note: str, kg_summary: str
    ) -> str:
        """
        Passage Note + KG özeti → Fact-Enhanced Note
        """
        print("\n[Step 2] KG augmentation yapılıyor...")

        prompt = f"""You are a knowledge augmentation assistant.

Query: {query}

Passage Note (from Wikipedia):
{passage_note}

Knowledge Graph Summary (structured facts):
{kg_summary}

Combine both sources into a FACT-ENHANCED NOTE that:
- Integrates structured KG facts with passage information
- Resolves any conflicts by trusting KG facts for specific entities/relations
- Highlights the most relevant facts for answering the query
- Is 4-6 sentences maximum
- Clearly states what is known about the query topic

Write the fact-enhanced note directly, no preamble."""

        response = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        enhanced = response.choices[0].message.content.strip()
        print(f"  ✔ Fact-Enhanced Note ({len(enhanced)} karakter)")
        print(f"  Preview: {enhanced[:120]}...")
        return enhanced

    # ──────────────────────────────────────────────────────────────────────────
    # ADIM 3: ANSWER GENERATION
    # ──────────────────────────────────────────────────────────────────────────

    @retry_on_failure(max_retries=3, delay=2.0)
    def generate_answer(
        self, query: str, fact_enhanced_note: str
    ) -> tuple[str, list]:
        """
        Orijinal sorgu + Fact-Enhanced Note → Final cevap üretir.
        Ayrıca reasoning chain listesi döndürür.
        """
        print("\n[Step 3] Final cevap üretiliyor...")

        prompt = f"""You are a question answering assistant.

Question: {query}

Fact-Enhanced Note:
{fact_enhanced_note}

Based on the fact-enhanced note, answer the question.
Requirements:
- Give a DIRECT, CONCISE answer (1-3 sentences)
- If the answer is a specific entity (person, place, award), state it clearly
- If the information is insufficient, say "Cannot be determined from available information"
- Do NOT add information beyond what is in the note

Answer:"""

        response = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        answer = response.choices[0].message.content.strip()

        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()

        print(f"  ✔ Final Cevap: {answer}")

        reasoning_chain = [
            "Step 1: Retrieved relevant passages from Wikipedia",
            "Step 2: Built passage note focusing on query-relevant facts",
            "Step 3: Augmented passage note with KG structured facts",
            f"Step 4: Generated answer: {answer[:80]}..."
        ]

        return answer, reasoning_chain

    # ──────────────────────────────────────────────────────────────────────────
    # ANA FONKSİYON
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        query:      str,
        passages:   list,
        kg_summary: str,
    ) -> dict:
        """
        Answer generation pipeline'ını çalıştırır.

        Returns:
            {
                "passage_note":       str,
                "fact_enhanced_note": str,
                "final_answer":       str,
                "reasoning_chain":    list[str]
            }
        """
        print("\n" + "=" * 60)
        print("[Module 3] KG-Augmented Answer Generation")
        print(f"Query: {query}")
        print(f"Passage sayısı: {len(passages)}")
        print("=" * 60)

        # Adım 1: Passage Note
        try:
            passage_note = self.build_passage_note(query, passages)
        except Exception as e:
            print(f"  [Module3] Passage note başarısız: {e}")
            passage_note = " ".join(
                p.get("content", "")[:100] for p in passages[:3]
            )

        # Adım 2: KG Augmentation
        try:
            fact_enhanced_note = self.augment_with_kg(query, passage_note, kg_summary)
        except Exception as e:
            print(f"  [Module3] Augmentation başarısız: {e}")
            fact_enhanced_note = f"{passage_note}\n\nAdditional KG facts: {kg_summary}"

        # Adım 3: Final Answer
        try:
            final_answer, reasoning_chain = self.generate_answer(
                query, fact_enhanced_note
            )
        except Exception as e:
            print(f"  [Module3] Answer generation başarısız: {e}")
            final_answer    = "Answer generation failed."
            reasoning_chain = []

        result = {
            "passage_note":       passage_note,
            "fact_enhanced_note": fact_enhanced_note,
            "final_answer":       final_answer,
            "reasoning_chain":    reasoning_chain,
        }

        print("\n[Module 3] ✅ Tamamlandı.")
        return result


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_query = "Galatasaray'ın teknik direktörünün doğum yeri neresidir?"

    test_kg_summary = (
        "Galatasaray F.C. is a Turkish football club based in Istanbul. "
        "Okan Buruk served as head coach of Galatasaray. "
        "Okan Buruk was born in Istanbul, Turkey."
    )

    test_passages = [
        {
            "title":   "Okan Buruk",
            "content": (
                "Okan Buruk is a Turkish football manager and former player. "
                "He was born on 19 October 1973 in Istanbul, Turkey. "
                "He managed Galatasaray F.C. and led the team to the Super Lig title."
            ),
            "query_source": "Galatasaray coach birthplace"
        },
        {
            "title":   "Galatasaray F.C.",
            "content": (
                "Galatasaray Spor Kulübü is a Turkish sports club founded in 1905 in Istanbul. "
                "The football section has won numerous Super Lig titles. "
                "The club is based in the Şişli district of Istanbul."
            ),
            "query_source": "Galatasaray football club"
        },
    ]

    ag     = AnswerGeneration()
    result = ag.run(
        query      = test_query,
        passages   = test_passages,
        kg_summary = test_kg_summary,
    )

    print("\n" + "─" * 60)
    print("SONUÇ:")
    print(f"\nPassage Note:\n  {result['passage_note']}")
    print(f"\nFact-Enhanced Note:\n  {result['fact_enhanced_note']}")
    print(f"\nFinal Cevap:\n  {result['final_answer']}")
    print(f"\nReasoning Chain:")
    for step in result["reasoning_chain"]:
        print(f"  → {step}")

    os.makedirs("outputs/phase4", exist_ok=True)
    with open("outputs/phase4/module3_test_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n  ✔ Test sonucu: outputs/phase4/module3_test_result.json")
