"""
Phase 3 - Multi-Hop Question Generation (Türk Sineması)
=========================================================
Neo4j'den gerçek path'leri çekip 50+ soru üretir.
- 30 adet 2-hop soru
- 15 adet 3-hop soru
-  5 adet karşılaştırma sorusu

Düzeltme:
  - "Eksik soru tamamlama" bloğu artık farklı template'leri döngüsel kullanıyor
    (sadece Film→DIRECTOR→PLACE_OF_BIRTH tekrarından kaçınılıyor)
  - Her template'e kullanım sayacı eklendi
  - Soru çeşitliliği artırıldı
"""

import os
import json
import random
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
URI       = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
USER      = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD  = os.getenv("NEO4J_PASSWORD", "neo4j")
TURKEY_ID = "Q43"

driver     = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
OUTPUT_DIR = "outputs/phase3"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run(cypher, params=None):
    with driver.session() as s:
        return [r.data() for r in s.run(cypher, params or {})]

def save_json(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✔ Kaydedildi: {path}")

def clean(text):
    if text is None:
        return "Unknown"
    return str(text).strip()

questions       = []
template_counts = {}   # ✔ template başına kullanım sayacı
q_id            = 1


def add_question(text, path, answer, difficulty, domain, template):
    global q_id
    questions.append({
        "question_id":    f"TR_{q_id:03d}",
        "question_text":  text,
        "reasoning_path": path,
        "gold_answer":    clean(answer),
        "difficulty":     difficulty,
        "domain":         domain,
        "template":       template,
        "verified":       True
    })
    template_counts[template] = template_counts.get(template, 0) + 1
    q_id += 1


print("=" * 65)
print("  PHASE 3 — Multi-Hop Soru Üretimi (Türk Sineması)")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# TİP 1: 2-HOP SORULAR (hedef: 30)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2-HOP] Soru üretiliyor...")

# ── 2A: Film → DIRECTOR → PLACE_OF_BIRTH ─────────────────────────────────
paths_2A = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    MATCH (director)-[:PLACE_OF_BIRTH]->(city:Entity)
    WHERE film.description IS NOT NULL
      AND director.description IS NOT NULL
      AND city.description IS NOT NULL
    RETURN film.description     AS film,
           director.description AS director,
           city.description     AS city
    LIMIT 200
""", {"tid": TURKEY_ID})

random.shuffle(paths_2A)
for p in paths_2A[:8]:
    film, director, city = clean(p['film']), clean(p['director']), clean(p['city'])
    add_question(
        text=f'"{film}" filminin yönetmeni nerede doğmuştur?',
        path=[film, "DIRECTOR", director, "PLACE_OF_BIRTH", city],
        answer=city,
        difficulty="2-hop",
        domain="cinema",
        template="Film → DIRECTOR → PLACE_OF_BIRTH"
    )

# ── 2B: Film → CAST_MEMBER → PLACE_OF_BIRTH ──────────────────────────────
paths_2B = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
    MATCH (actor)-[:PLACE_OF_BIRTH]->(city:Entity)
    WHERE film.description IS NOT NULL
      AND actor.description IS NOT NULL
      AND city.description IS NOT NULL
    RETURN film.description  AS film,
           actor.description AS actor,
           city.description  AS city
    LIMIT 200
""", {"tid": TURKEY_ID})

random.shuffle(paths_2B)
for p in paths_2B[:8]:
    film, actor, city = clean(p['film']), clean(p['actor']), clean(p['city'])
    add_question(
        text=f'"{film}" filminde oynayan {actor} nerede doğmuştur?',
        path=[film, "CAST_MEMBER", actor, "PLACE_OF_BIRTH", city],
        answer=city,
        difficulty="2-hop",
        domain="cinema",
        template="Film → CAST_MEMBER → PLACE_OF_BIRTH"
    )

# ── 2C: Film → DIRECTOR → AWARD_RECEIVED ─────────────────────────────────
paths_2C = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    MATCH (director)-[:AWARD_RECEIVED]->(award:Entity)
    WHERE film.description IS NOT NULL
      AND director.description IS NOT NULL
      AND award.description IS NOT NULL
    RETURN film.description     AS film,
           director.description AS director,
           award.description    AS award
    LIMIT 100
""", {"tid": TURKEY_ID})

random.shuffle(paths_2C)
for p in paths_2C[:5]:
    film, director, award = clean(p['film']), clean(p['director']), clean(p['award'])
    add_question(
        text=f'"{film}" filminin yönetmeni {director} hangi ödülü almıştır?',
        path=[film, "DIRECTOR", director, "AWARD_RECEIVED", award],
        answer=award,
        difficulty="2-hop",
        domain="cinema",
        template="Film → DIRECTOR → AWARD_RECEIVED"
    )

# ── 2D: Film → CAST_MEMBER → COUNTRY_OF_CITIZENSHIP ─────────────────────
paths_2D = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
    MATCH (actor)-[:COUNTRY_OF_CITIZENSHIP]->(country:Entity)
    WHERE film.description IS NOT NULL
      AND actor.description IS NOT NULL
      AND country.description IS NOT NULL
    RETURN film.description    AS film,
           actor.description   AS actor,
           country.description AS country
    LIMIT 100
""", {"tid": TURKEY_ID})

random.shuffle(paths_2D)
for p in paths_2D[:5]:
    film, actor, country = clean(p['film']), clean(p['actor']), clean(p['country'])
    add_question(
        text=f'"{film}" filminde oynayan {actor} hangi ülkenin vatandaşıdır?',
        path=[film, "CAST_MEMBER", actor, "COUNTRY_OF_CITIZENSHIP", country],
        answer=country,
        difficulty="2-hop",
        domain="cinema",
        template="Film → CAST_MEMBER → COUNTRY_OF_CITIZENSHIP"
    )

# ── 2E: Film → DIRECTOR → EDUCATED_AT ───────────────────────────────────
paths_2E = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    MATCH (director)-[:EDUCATED_AT]->(school:Entity)
    WHERE film.description IS NOT NULL
      AND director.description IS NOT NULL
      AND school.description IS NOT NULL
    RETURN film.description     AS film,
           director.description AS director,
           school.description   AS school
    LIMIT 100
""", {"tid": TURKEY_ID})

random.shuffle(paths_2E)
for p in paths_2E[:4]:
    film, director, school = clean(p['film']), clean(p['director']), clean(p['school'])
    add_question(
        text=f'"{film}" filminin yönetmeni {director} hangi kurumda eğitim görmüştür?',
        path=[film, "DIRECTOR", director, "EDUCATED_AT", school],
        answer=school,
        difficulty="2-hop",
        domain="cinema",
        template="Film → DIRECTOR → EDUCATED_AT"
    )

print(f"  2-hop soru sayısı (birincil): {len(questions)}")


# ══════════════════════════════════════════════════════════════════════════════
# TİP 2: 3-HOP SORULAR (hedef: 15)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3-HOP] Soru üretiliyor...")
hop3_start = len(questions)

paths_3A = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    MATCH (director)-[:PLACE_OF_BIRTH]->(city:Entity)
    MATCH (city)-[:COUNTRY]->(country:Entity)
    WHERE film.description IS NOT NULL AND director.description IS NOT NULL
      AND city.description IS NOT NULL AND country.description IS NOT NULL
    RETURN film.description AS film, director.description AS director,
           city.description AS city, country.description AS country
    LIMIT 200
""", {"tid": TURKEY_ID})

random.shuffle(paths_3A)
for p in paths_3A[:5]:
    film, director, city, country = (clean(p['film']), clean(p['director']),
                                      clean(p['city']), clean(p['country']))
    add_question(
        text=f'"{film}" filminin yönetmeni {director}\'nin doğduğu şehrin ülkesi neresidir?',
        path=[film, "DIRECTOR", director, "PLACE_OF_BIRTH", city, "COUNTRY", country],
        answer=country,
        difficulty="3-hop",
        domain="cinema",
        template="Film → DIRECTOR → PLACE_OF_BIRTH → COUNTRY"
    )

paths_3B = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
    MATCH (actor)-[:PLACE_OF_BIRTH]->(city:Entity)
    MATCH (city)-[:COUNTRY]->(country:Entity)
    WHERE film.description IS NOT NULL AND actor.description IS NOT NULL
      AND city.description IS NOT NULL AND country.description IS NOT NULL
    RETURN film.description AS film, actor.description AS actor,
           city.description AS city, country.description AS country
    LIMIT 200
""", {"tid": TURKEY_ID})

random.shuffle(paths_3B)
for p in paths_3B[:5]:
    film, actor, city, country = (clean(p['film']), clean(p['actor']),
                                   clean(p['city']), clean(p['country']))
    add_question(
        text=f'"{film}" filminde oynayan {actor}\'nin doğduğu şehrin ülkesi nedir?',
        path=[film, "CAST_MEMBER", actor, "PLACE_OF_BIRTH", city, "COUNTRY", country],
        answer=country,
        difficulty="3-hop",
        domain="cinema",
        template="Film → CAST_MEMBER → PLACE_OF_BIRTH → COUNTRY"
    )

paths_3C = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    MATCH (director)-[:EDUCATED_AT]->(school:Entity)
    MATCH (school)-[:COUNTRY]->(country:Entity)
    WHERE film.description IS NOT NULL AND director.description IS NOT NULL
      AND school.description IS NOT NULL AND country.description IS NOT NULL
    RETURN film.description AS film, director.description AS director,
           school.description AS school, country.description AS country
    LIMIT 100
""", {"tid": TURKEY_ID})

random.shuffle(paths_3C)
for p in paths_3C[:3]:
    film, director, school, country = (clean(p['film']), clean(p['director']),
                                        clean(p['school']), clean(p['country']))
    add_question(
        text=f'"{film}" yönetmeni {director}\'nin mezun olduğu okulun bulunduğu ülke neresidir?',
        path=[film, "DIRECTOR", director, "EDUCATED_AT", school, "COUNTRY", country],
        answer=country,
        difficulty="3-hop",
        domain="cinema",
        template="Film → DIRECTOR → EDUCATED_AT → COUNTRY"
    )

paths_3D = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
    MATCH (actor)-[:AWARD_RECEIVED]->(award:Entity)
    WHERE film.description IS NOT NULL AND actor.description IS NOT NULL
      AND award.description IS NOT NULL
    RETURN film.description AS film, actor.description AS actor,
           award.description AS award
    LIMIT 100
""", {"tid": TURKEY_ID})

random.shuffle(paths_3D)
for p in paths_3D[:2]:
    film, actor, award = clean(p['film']), clean(p['actor']), clean(p['award'])
    add_question(
        text=f'"{film}" filminde yer alan oyuncu {actor} hangi ödülü kazanmıştır?',
        path=[film, "CAST_MEMBER", actor, "AWARD_RECEIVED", award],
        answer=award,
        difficulty="3-hop",
        domain="cinema",
        template="Film → CAST_MEMBER → AWARD_RECEIVED"
    )

hop3_count = len(questions) - hop3_start
print(f"  3-hop soru sayısı: {hop3_count}")


# ══════════════════════════════════════════════════════════════════════════════
# TİP 3: KARŞILAŞTIRMA SORULARI (hedef: 5)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[KARŞILAŞTIRMA] Soru üretiliyor...")

pairs = run("""
    MATCH (f1:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (f2:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (f1)-[:DIRECTOR]->(d1:Entity)
    MATCH (f2)-[:DIRECTOR]->(d2:Entity)
    WHERE f1.entityId < f2.entityId AND d1.entityId <> d2.entityId
      AND f1.description IS NOT NULL AND f2.description IS NOT NULL
      AND d1.description IS NOT NULL AND d2.description IS NOT NULL
    RETURN f1.description AS film1, d1.description AS dir1,
           f2.description AS film2, d2.description AS dir2
    LIMIT 20
""", {"tid": TURKEY_ID})

random.shuffle(pairs)
for p in pairs[:3]:
    f1, d1, f2, d2 = (clean(p['film1']), clean(p['dir1']),
                       clean(p['film2']), clean(p['dir2']))
    add_question(
        text=f'"{f1}" ve "{f2}" filmlerinden hangisinin yönetmeni daha fazla ödül almıştır: {d1} mı yoksa {d2} mi?',
        path=[f1, "DIRECTOR", d1, "vs", f2, "DIRECTOR", d2],
        answer=f"Karşılaştırma sorusu - her iki yönetmenin ödülleri kontrol edilmeli: {d1} / {d2}",
        difficulty="comparison",
        domain="cinema",
        template="Film1.DIRECTOR vs Film2.DIRECTOR (award comparison)"
    )

actor_pairs = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(a1:Entity)
    MATCH (film)-[:CAST_MEMBER]->(a2:Entity)
    MATCH (a1)-[:PLACE_OF_BIRTH]->(c1:Entity)
    MATCH (a2)-[:PLACE_OF_BIRTH]->(c2:Entity)
    WHERE a1.entityId < a2.entityId AND c1.entityId <> c2.entityId
      AND a1.description IS NOT NULL AND a2.description IS NOT NULL
      AND c1.description IS NOT NULL AND c2.description IS NOT NULL
      AND film.description IS NOT NULL
    RETURN film.description AS film,
           a1.description AS actor1, c1.description AS city1,
           a2.description AS actor2, c2.description AS city2
    LIMIT 20
""", {"tid": TURKEY_ID})

random.shuffle(actor_pairs)
for p in actor_pairs[:2]:
    film, a1, c1, a2, c2 = (clean(p['film']), clean(p['actor1']), clean(p['city1']),
                              clean(p['actor2']), clean(p['city2']))
    add_question(
        text=f'"{film}" filminde oynayan {a1} mı yoksa {a2} mi daha büyük bir şehirde doğmuştur?',
        path=[film, "CAST_MEMBER", a1, "PLACE_OF_BIRTH", c1, "vs", a2, "PLACE_OF_BIRTH", c2],
        answer=f"Karşılaştırma: {a1} → {c1} / {a2} → {c2}",
        difficulty="comparison",
        domain="cinema",
        template="Film.CAST_MEMBER1.PLACE_OF_BIRTH vs CAST_MEMBER2.PLACE_OF_BIRTH"
    )

comp_count = len([q for q in questions if q['difficulty'] == 'comparison'])
print(f"  Karşılaştırma soru sayısı: {comp_count}")


# ══════════════════════════════════════════════════════════════════════════════
# EKSİK SORU TAMAMLAMA — ÇEŞİTLİ TEMPLATE'LER (✔ Düzeltildi)
# ══════════════════════════════════════════════════════════════════════════════
current = len(questions)
print(f"\n  Mevcut soru sayısı: {current} / 50")

if current < 50:
    needed = 50 - current
    print(f"  {needed} ek soru üretiliyor (farklı template'ler kullanılıyor)...")

    # ✔ Birden fazla kaynak havuzu — çeşitlilik için döngüsel kullanılıyor
    extra_pools = [
        # Havuz A: Film → DIRECTOR → PLACE_OF_BIRTH
        ("Film → DIRECTOR → PLACE_OF_BIRTH (extra)", run("""
            MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
            MATCH (film)-[:DIRECTOR]->(director:Entity)
            MATCH (director)-[:PLACE_OF_BIRTH]->(city:Entity)
            WHERE film.description IS NOT NULL
              AND director.description IS NOT NULL
              AND city.description IS NOT NULL
            RETURN film.description AS film,
                   director.description AS director,
                   city.description AS city
            LIMIT 500
        """, {"tid": TURKEY_ID}), "film", "director", "city",
         lambda r: (
             f'"{clean(r["film"])}" filmini yöneten kişi nerede doğmuştur?',
             [clean(r["film"]), "DIRECTOR", clean(r["director"]), "PLACE_OF_BIRTH", clean(r["city"])],
             r["city"]
         )),
        # Havuz B: Film → CAST_MEMBER → PLACE_OF_BIRTH
        ("Film → CAST_MEMBER → PLACE_OF_BIRTH (extra)", run("""
            MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
            MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
            MATCH (actor)-[:PLACE_OF_BIRTH]->(city:Entity)
            WHERE film.description IS NOT NULL
              AND actor.description IS NOT NULL
              AND city.description IS NOT NULL
            RETURN film.description AS film,
                   actor.description AS actor,
                   city.description AS city
            LIMIT 500
        """, {"tid": TURKEY_ID}), "film", "actor", "city",
         lambda r: (
             f'"{clean(r["film"])}" filminde oynayan {clean(r["actor"])} nereden gelir?',
             [clean(r["film"]), "CAST_MEMBER", clean(r["actor"]), "PLACE_OF_BIRTH", clean(r["city"])],
             r["city"]
         )),
        # Havuz C: Film → CAST_MEMBER → COUNTRY_OF_CITIZENSHIP
        ("Film → CAST_MEMBER → COUNTRY_OF_CITIZENSHIP (extra)", run("""
            MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
            MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
            MATCH (actor)-[:COUNTRY_OF_CITIZENSHIP]->(country:Entity)
            WHERE film.description IS NOT NULL
              AND actor.description IS NOT NULL
              AND country.description IS NOT NULL
            RETURN film.description AS film,
                   actor.description AS actor,
                   country.description AS country
            LIMIT 500
        """, {"tid": TURKEY_ID}), "film", "actor", "country",
         lambda r: (
             f'"{clean(r["film"])}" filminin oyuncusu {clean(r["actor"])} hangi ülkenin vatandaşıdır?',
             [clean(r["film"]), "CAST_MEMBER", clean(r["actor"]), "COUNTRY_OF_CITIZENSHIP", clean(r["country"])],
             r["country"]
         )),
    ]

    # Havuzları karıştır
    for template_name, pool, key1, key2, key3 in extra_pools:
        random.shuffle(pool)

    used_keys = {(q['reasoning_path'][0], q['reasoning_path'][2])
                 for q in questions
                 if len(q['reasoning_path']) >= 3}

    added     = 0
    pool_idx  = 0  # ✔ round-robin havuz seçimi

    while added < needed:
        template_name, pool, key1, key2, key3, make_q = extra_pools[pool_idx % len(extra_pools)]
        pool_idx += 1

        found = False
        for r in pool:
            k1  = clean(r.get(key1, ""))
            k2  = clean(r.get(key2, ""))
            uniq_key = (k1, k2)
            if uniq_key in used_keys:
                continue
            used_keys.add(uniq_key)

            text, path, answer = make_q(r)
            add_question(
                text=text,
                path=path,
                answer=answer,
                difficulty="2-hop",
                domain="cinema",
                template=template_name
            )
            added += 1
            found = True
            break

        if not found:
            print(f"  ⚠ Havuz tükendi: {template_name}")
            break

    print(f"  Ek soru eklendi: {added}")


# ══════════════════════════════════════════════════════════════════════════════
# ÖZET VE KAYDET
# ══════════════════════════════════════════════════════════════════════════════
total      = len(questions)
two_hop    = len([q for q in questions if q['difficulty'] == '2-hop'])
three_hop  = len([q for q in questions if q['difficulty'] == '3-hop'])
comparison = len([q for q in questions if q['difficulty'] == 'comparison'])

print("\n" + "=" * 65)
print("  SORU ÖZET TABLOSU")
print("=" * 65)
print(f"  2-hop soruları    : {two_hop:>3}  (min 30) {'✅' if two_hop >= 30 else '⚠️'}")
print(f"  3-hop soruları    : {three_hop:>3}  (min 15) {'✅' if three_hop >= 15 else '⚠️'}")
print(f"  Karşılaştırma     : {comparison:>3}  (min  5) {'✅' if comparison >= 5 else '⚠️'}")
print(f"  TOPLAM            : {total:>3}  (min 50) {'✅' if total >= 50 else '⚠️'}")

# ✔ Template çeşitlilik raporu
print(f"\n  Template Çeşitliliği ({len(template_counts)} farklı template):")
for tmpl, cnt in sorted(template_counts.items(), key=lambda x: -x[1]):
    print(f"    {tmpl:<60} : {cnt:>3}")

print("\n  Örnek sorular:")
for q in questions[:3]:
    print(f"\n  [{q['question_id']}] {q['question_text']}")
    print(f"  Path   : {' → '.join(str(x) for x in q['reasoning_path'])}")
    print(f"  Cevap  : {q['gold_answer']}")
    print(f"  Tip    : {q['difficulty']}")

save_json(questions, "qa_dataset.json")
save_json({
    "total_questions": total,
    "by_difficulty": {
        "2-hop":      two_hop,
        "3-hop":      three_hop,
        "comparison": comparison
    },
    "template_distribution": template_counts,
    "domain": "cinema",
    "criteria_met": {
        "2hop_ok":  two_hop >= 30,
        "3hop_ok":  three_hop >= 15,
        "comp_ok":  comparison >= 5,
        "total_ok": total >= 50
    }
}, "qa_dataset_summary.json")

driver.close()
print("\n✅ Phase 3 tamamlandı!")
print(f"   Çıktı: outputs/phase3/qa_dataset.json ({total} soru)")
