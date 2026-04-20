"""
phase3_cinema_question_generator.py
====================================
Düzeltme:
  - _is_valid_answer(): LOCATED_IN / COUNTRY zincirlerindeki Wikidata
    artifact gold answer'ları filtrelenir.
    "Archbishopric of ...", "List of ...", "Theodosiopolis" gibi
    tarihi/kirli değerler dataset'e girmez.
"""

import os
import sys
import json
import math
import random
import re

from neo4j import GraphDatabase

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, TURKEY_ID, OUTPUT_PHASE3_CINEMA

driver     = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
OUTPUT_DIR = OUTPUT_PHASE3_CINEMA
os.makedirs(OUTPUT_DIR, exist_ok=True)

_HERE       = os.path.dirname(os.path.abspath(__file__))
PHASE2_PATH = os.path.join(_HERE, "..", "..", "outputs",
                            "phase2_domain_selection", "cinema",
                            "cinema_path_density_report.json")

print("=" * 65)
print("  PHASE 3 -- Türk Sineması Soru Üretimi")
print("  Pipeline: Phase 2 -> Phase 3")
print("=" * 65)

if not os.path.exists(PHASE2_PATH):
    raise FileNotFoundError(
        f"Phase 2 raporu bulunamadı: {PHASE2_PATH}\n"
        "Önce phase2_cinema_verification.py'yi çalıştırın."
    )

with open(PHASE2_PATH, encoding="utf-8") as f:
    PHASE2 = json.load(f)

FILM_RELS      = set(PHASE2["summary"]["unique_rel_types"])
PATH_COUNTS_2H = PHASE2["path_type_breakdown"]["2hop"].copy()
PATH_COUNTS_3H = PHASE2["path_type_breakdown"]["3hop"].copy()

_3B_cnt = PATH_COUNTS_3H.pop("3B_film_actor_award", 0)
PATH_COUNTS_2H["3B_actor_award"] = _3B_cnt

ZERO_PATH_TEMPLATES_2H = sorted([k for k, v in PATH_COUNTS_2H.items() if v == 0])
ZERO_PATH_TEMPLATES_3H = sorted([k for k, v in PATH_COUNTS_3H.items() if v == 0])

TOTAL_2H = sum(PATH_COUNTS_2H.values())
TOTAL_3H = sum(PATH_COUNTS_3H.values())

REQUIRED_DIRECT_RELS = {"DIRECTOR", "CAST_MEMBER", "COUNTRY_OF_ORIGIN", "AWARD_RECEIVED"}
missing = REQUIRED_DIRECT_RELS - FILM_RELS
if missing:
    print(f"\n  UYARI: {missing} Phase 2 raporunda eksik!")

print(f"\n  Phase 2 raporu yüklendi:")
print(f"    Film relation'ları : {sorted(FILM_RELS)}")
print(f"    2-hop path toplam  : {TOTAL_2H}")
print(f"    3-hop path toplam  : {TOTAL_3H}")


def proportional_limit(count: int, total: int, target: int, minimum: int = 1) -> int:
    if total == 0 or count == 0:
        return 0
    return max(minimum, math.ceil(target * count / total))


TARGET_2HOP  = 30
TARGET_3HOP  = 15
TARGET_COMP  = 5
TARGET_TOTAL = 50

L2 = {k: proportional_limit(v, TOTAL_2H, TARGET_2HOP) for k, v in PATH_COUNTS_2H.items()}
L3 = {k: proportional_limit(v, TOTAL_3H, TARGET_3HOP) for k, v in PATH_COUNTS_3H.items()}

print(f"\n  Orantılı 2-hop limitleri: {L2}")
print(f"  Orantılı 3-hop limitleri: {L3}")

# -----------------------------------------------------------------------------
# Gold answer filtresi
# -----------------------------------------------------------------------------
_NOISY_PATTERNS = [
    r"^archbishopric\b", r"^bishopric\b", r"^list of\b",
    r"^theodosiopolis\b", r"^nicaea\b", r"^ancient\b",
    r"\(ancient\)", r"\(historical\)", r"\(armenia\)",
    r"^diocese\b", r"^patriarchate\b", r"^metropolitan\b", r"^eparchy\b",
]


def _is_valid_answer(answer: str) -> bool:
    if not answer or answer.strip().lower() in ("unknown", "none", ""):
        return False
    a = answer.strip().lower()
    if any(re.search(pat, a) for pat in _NOISY_PATTERNS):
        return False
    if len(answer.strip()) > 120:
        return False
    return True


# -----------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# -----------------------------------------------------------------------------

def run(cypher: str, params: dict = None):
    with driver.session() as s:
        return [r.data() for r in s.run(cypher, params or {})]


def clean(text) -> str:
    if text is None:
        return "Unknown"
    return str(text).strip()


questions       = []
template_counts = {}
used_keys       = set()
q_id            = 1
filtered_count  = 0


def add_question(text: str, path: list, answer: str,
                 difficulty: str, template: str) -> bool:
    global q_id, filtered_count
    if not _is_valid_answer(answer):
        filtered_count += 1
        return False
    e1  = path[0] if path else ""
    e2  = path[2] if len(path) > 2 else ""
    key = (e1, e2, template)
    if key in used_keys:
        return False
    used_keys.add(key)
    questions.append({
        "question_id":    f"TR_{q_id:03d}",
        "question_text":  text,
        "reasoning_path": path,
        "gold_answer":    clean(answer),
        "difficulty":     difficulty,
        "domain":         "cinema",
        "template":       template,
        "verified":       True,
    })
    template_counts[template] = template_counts.get(template, 0) + 1
    q_id += 1
    return True


def fill_from_pool(pool: list, limit: int, make_fn, difficulty: str, template: str) -> int:
    random.shuffle(pool)
    added = 0
    for r in pool:
        if added >= limit:
            break
        text, path, ans = make_fn(r)
        if add_question(text, path, ans, difficulty, template):
            added += 1
    return added


# -----------------------------------------------------------------------------
# 2-HOP SORGULAR
# -----------------------------------------------------------------------------
print("\n[2-HOP] Sorgular çalıştırılıyor...")

paths_2A = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    MATCH (director)-[:PLACE_OF_BIRTH]->(city:Entity)
    WHERE film.name IS NOT NULL AND director.name IS NOT NULL AND city.name IS NOT NULL
    RETURN film.name AS film, director.name AS director, city.name AS city LIMIT 400
""", {"tid": TURKEY_ID})
fill_from_pool(paths_2A, L2.get("A_film_director_city", 6),
    lambda r: (
        f'Where was the director of "{clean(r["film"])}" born?',
        [clean(r["film"]), "DIRECTOR", clean(r["director"]), "PLACE_OF_BIRTH", clean(r["city"])],
        r["city"]),
    "2-hop", "Film->DIRECTOR->PLACE_OF_BIRTH")

paths_2B = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
    MATCH (actor)-[:PLACE_OF_BIRTH]->(city:Entity)
    WHERE film.name IS NOT NULL AND actor.name IS NOT NULL AND city.name IS NOT NULL
    RETURN film.name AS film, actor.name AS actor, city.name AS city LIMIT 400
""", {"tid": TURKEY_ID})
fill_from_pool(paths_2B, L2.get("B_film_actor_city", 7),
    lambda r: (
        f'Where was {clean(r["actor"])} from "{clean(r["film"])}" born?',
        [clean(r["film"]), "CAST_MEMBER", clean(r["actor"]), "PLACE_OF_BIRTH", clean(r["city"])],
        r["city"]),
    "2-hop", "Film->CAST_MEMBER->PLACE_OF_BIRTH")

paths_2C = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    MATCH (director)-[:AWARD_RECEIVED]->(award:Entity)
    WHERE film.name IS NOT NULL AND director.name IS NOT NULL AND award.name IS NOT NULL
    RETURN film.name AS film, director.name AS director, award.name AS award LIMIT 200
""", {"tid": TURKEY_ID})
fill_from_pool(paths_2C, L2.get("C_film_director_award", 1),
    lambda r: (
        f'Which award was won by {clean(r["director"])}, the director of "{clean(r["film"])}"?',
        [clean(r["film"]), "DIRECTOR", clean(r["director"]), "AWARD_RECEIVED", clean(r["award"])],
        r["award"]),
    "2-hop", "Film->DIRECTOR->AWARD_RECEIVED")

paths_2D = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
    MATCH (actor)-[:COUNTRY_OF_CITIZENSHIP]->(country:Entity)
    WHERE film.name IS NOT NULL AND actor.name IS NOT NULL AND country.name IS NOT NULL
    RETURN film.name AS film, actor.name AS actor, country.name AS country LIMIT 400
""", {"tid": TURKEY_ID})
fill_from_pool(paths_2D, L2.get("D_film_actor_citizenship", 7),
    lambda r: (
        f'Which country is {clean(r["actor"])} from "{clean(r["film"])}" a citizen of?',
        [clean(r["film"]), "CAST_MEMBER", clean(r["actor"]), "COUNTRY_OF_CITIZENSHIP", clean(r["country"])],
        r["country"]),
    "2-hop", "Film->CAST_MEMBER->COUNTRY_OF_CITIZENSHIP")

paths_2E = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    MATCH (director)-[:EDUCATED_AT]->(school:Entity)
    WHERE film.name IS NOT NULL AND director.name IS NOT NULL AND school.name IS NOT NULL
    RETURN film.name AS film, director.name AS director, school.name AS school LIMIT 200
""", {"tid": TURKEY_ID})
fill_from_pool(paths_2E, L2.get("E_film_director_school", 2),
    lambda r: (
        f'Where did {clean(r["director"])}, the director of "{clean(r["film"])}", study?',
        [clean(r["film"]), "DIRECTOR", clean(r["director"]), "EDUCATED_AT", clean(r["school"])],
        r["school"]),
    "2-hop", "Film->DIRECTOR->EDUCATED_AT")

paths_2F = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:AWARD_RECEIVED]->(award:Entity)
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    WHERE film.name IS NOT NULL AND director.name IS NOT NULL AND award.name IS NOT NULL
    RETURN film.name AS film, director.name AS director, award.name AS award LIMIT 100
""", {"tid": TURKEY_ID})
fill_from_pool(paths_2F, L2.get("F_film_director_award_film", 1),
    lambda r: (
        f'Which award was won by the film "{clean(r["film"])}"?',
        [clean(r["film"]), "AWARD_RECEIVED", clean(r["award"]), "<-DIRECTOR-", clean(r["director"])],
        r["award"]),
    "2-hop", "Film->AWARD_RECEIVED (with DIRECTOR)")

paths_2G = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
    MATCH (actor)-[:EDUCATED_AT]->(school:Entity)
    WHERE film.name IS NOT NULL AND actor.name IS NOT NULL AND school.name IS NOT NULL
    RETURN film.name AS film, actor.name AS actor, school.name AS school LIMIT 400
""", {"tid": TURKEY_ID})
fill_from_pool(paths_2G, L2.get("G_film_actor_school", 6),
    lambda r: (
        f'Where did {clean(r["actor"])} from "{clean(r["film"])}" study?',
        [clean(r["film"]), "CAST_MEMBER", clean(r["actor"]), "EDUCATED_AT", clean(r["school"])],
        r["school"]),
    "2-hop", "Film->CAST_MEMBER->EDUCATED_AT")

paths_2H = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
    MATCH (actor)-[:AWARD_RECEIVED]->(award:Entity)
    WHERE film.name IS NOT NULL AND actor.name IS NOT NULL AND award.name IS NOT NULL
    RETURN film.name AS film, actor.name AS actor, award.name AS award LIMIT 400
""", {"tid": TURKEY_ID})
_limit_2H = L2.get("H_film_actor_award", 5) + L2.get("3B_actor_award", 0)
fill_from_pool(paths_2H, _limit_2H,
    lambda r: (
        f'Which award was won by {clean(r["actor"])} from "{clean(r["film"])}"?',
        [clean(r["film"]), "CAST_MEMBER", clean(r["actor"]), "AWARD_RECEIVED", clean(r["award"])],
        r["award"]),
    "2-hop", "Film->CAST_MEMBER->AWARD_RECEIVED")

hop2_count = len(questions)
print(f"  2-hop soru sayısı  : {hop2_count}")
print(f"  Filtrelenen (kirli): {filtered_count}")

# -----------------------------------------------------------------------------
# 3-HOP SORGULAR
# -----------------------------------------------------------------------------
print("\n[3-HOP] Sorgular çalıştırılıyor...")
hop3_start      = len(questions)
filtered_before = filtered_count

paths_3A = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    MATCH (director)-[:PLACE_OF_BIRTH]->(city:Entity)
    MATCH (city)-[:COUNTRY]->(country:Entity)
    WHERE film.name IS NOT NULL AND director.name IS NOT NULL
      AND city.name IS NOT NULL AND country.name IS NOT NULL
    RETURN film.name AS film, director.name AS director,
           city.name AS city, country.name AS country LIMIT 400
""", {"tid": TURKEY_ID})
fill_from_pool(paths_3A, L3.get("3A_film_director_city_country", 4),
    lambda r: (
        f'Which country is the birth city of {clean(r["director"])}, '
        f'the director of "{clean(r["film"])}", in?',
        [clean(r["film"]), "DIRECTOR", clean(r["director"]),
         "PLACE_OF_BIRTH", clean(r["city"]), "COUNTRY", clean(r["country"])],
        r["country"]),
    "3-hop", "Film->DIRECTOR->PLACE_OF_BIRTH->COUNTRY")

paths_3C = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    MATCH (director)-[:EDUCATED_AT]->(school:Entity)
    MATCH (school)-[:COUNTRY]->(country:Entity)
    WHERE film.name IS NOT NULL AND director.name IS NOT NULL
      AND school.name IS NOT NULL AND country.name IS NOT NULL
    RETURN film.name AS film, director.name AS director,
           school.name AS school, country.name AS country LIMIT 200
""", {"tid": TURKEY_ID})
fill_from_pool(paths_3C, L3.get("3C_film_director_school_country", 1),
    lambda r: (
        f'Which country is {clean(r["school"])}, where {clean(r["director"])} studied, '
        f'in for the film "{clean(r["film"])}"?',
        [clean(r["film"]), "DIRECTOR", clean(r["director"]),
         "EDUCATED_AT", clean(r["school"]), "COUNTRY", clean(r["country"])],
        r["country"]),
    "3-hop", "Film->DIRECTOR->EDUCATED_AT->COUNTRY")

paths_3D = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
    MATCH (actor)-[:PLACE_OF_BIRTH]->(city:Entity)
    MATCH (city)-[:COUNTRY]->(country:Entity)
    WHERE film.name IS NOT NULL AND actor.name IS NOT NULL
      AND city.name IS NOT NULL AND country.name IS NOT NULL
    RETURN film.name AS film, actor.name AS actor,
           city.name AS city, country.name AS country LIMIT 400
""", {"tid": TURKEY_ID})
fill_from_pool(paths_3D, L3.get("3D_film_actor_city_country", 4),
    lambda r: (
        f'Which country is the birth city of {clean(r["actor"])} '
        f'from "{clean(r["film"])}" in?',
        [clean(r["film"]), "CAST_MEMBER", clean(r["actor"]),
         "PLACE_OF_BIRTH", clean(r["city"]), "COUNTRY", clean(r["country"])],
        r["country"]),
    "3-hop", "Film->CAST_MEMBER->PLACE_OF_BIRTH->COUNTRY")

paths_3E = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
    MATCH (actor)-[:EDUCATED_AT]->(school:Entity)
    MATCH (school)-[:COUNTRY]->(country:Entity)
    WHERE film.name IS NOT NULL AND actor.name IS NOT NULL
      AND school.name IS NOT NULL AND country.name IS NOT NULL
    RETURN film.name AS film, actor.name AS actor,
           school.name AS school, country.name AS country LIMIT 400
""", {"tid": TURKEY_ID})
fill_from_pool(paths_3E, L3.get("3E_film_actor_school_country", 4),
    lambda r: (
        f'Which country is {clean(r["school"])}, where {clean(r["actor"])} studied '
        f'in "{clean(r["film"])}", in?',
        [clean(r["film"]), "CAST_MEMBER", clean(r["actor"]),
         "EDUCATED_AT", clean(r["school"]), "COUNTRY", clean(r["country"])],
        r["country"]),
    "3-hop", "Film->CAST_MEMBER->EDUCATED_AT->COUNTRY")

paths_3F = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
    MATCH (actor)-[:PLACE_OF_BIRTH]->(city:Entity)
    MATCH (city)-[:LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY]->(province:Entity)
    WHERE film.name IS NOT NULL AND actor.name IS NOT NULL
      AND city.name IS NOT NULL AND province.name IS NOT NULL
    RETURN film.name AS film, actor.name AS actor,
           city.name AS city, province.name AS province LIMIT 400
""", {"tid": TURKEY_ID})
fill_from_pool(paths_3F, L3.get("3F_film_actor_city_province", 4),
    lambda r: (
        f'Which province or administrative area contains {clean(r["city"])}, '
        f'where {clean(r["actor"])} from "{clean(r["film"])}" was born?',
        [clean(r["film"]), "CAST_MEMBER", clean(r["actor"]),
         "PLACE_OF_BIRTH", clean(r["city"]),
         "LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY", clean(r["province"])],
        r["province"]),
    "3-hop", "Film->CAST_MEMBER->PLACE_OF_BIRTH->LOCATED_IN")

hop3_count = len(questions) - hop3_start
print(f"  3-hop soru sayısı  : {hop3_count}")
print(f"  Filtrelenen (kirli): {filtered_count - filtered_before}")

# -----------------------------------------------------------------------------
# KARŞILAŞTIRMA SORULARI
# -----------------------------------------------------------------------------
print("\n[COMPARISON] Sorgular çalıştırılıyor...")
comp_start = len(questions)

actor_cit = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(a1:Entity)-[:COUNTRY_OF_CITIZENSHIP]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(a2:Entity)-[:COUNTRY_OF_CITIZENSHIP]->(other:Entity)
    WHERE a1.entityId < a2.entityId AND other.entityId <> $tid
      AND film.name IS NOT NULL AND a1.name IS NOT NULL AND a2.name IS NOT NULL
    RETURN film.name AS film, a1.name AS turkish_actor, a2.name AS foreign_actor LIMIT 50
""", {"tid": TURKEY_ID})
random.shuffle(actor_cit)
for p in actor_cit[:5]:
    film, a1, a2 = clean(p["film"]), clean(p["turkish_actor"]), clean(p["foreign_actor"])
    add_question(
        text=f'Which actor in "{film}" is Turkish, {a1} or {a2}?',
        path=[film, "CAST_MEMBER", a1, "COUNTRY_OF_CITIZENSHIP", "Turkiye",
              "vs", film, "CAST_MEMBER", a2],
        answer=a1, difficulty="comparison",
        template="Film.CastMember(Turkish) vs Film.CastMember(Foreign)")

dir_cit = run("""
    MATCH (f1:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (f1)-[:DIRECTOR]->(d1:Entity)-[:COUNTRY_OF_CITIZENSHIP]->(:Entity {entityId: $tid})
    MATCH (f2:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (f2)-[:DIRECTOR]->(d2:Entity)-[:COUNTRY_OF_CITIZENSHIP]->(other:Entity)
    WHERE f1.entityId < f2.entityId AND d1.entityId <> d2.entityId
      AND other.entityId <> $tid
      AND f1.name IS NOT NULL AND f2.name IS NOT NULL
      AND d1.name IS NOT NULL AND d2.name IS NOT NULL
    RETURN f1.name AS film1, d1.name AS dir1, f2.name AS film2, d2.name AS dir2 LIMIT 30
""", {"tid": TURKEY_ID})
random.shuffle(dir_cit)
for p in dir_cit[:3]:
    f1, d1, f2, d2 = (clean(p["film1"]), clean(p["dir1"]),
                      clean(p["film2"]), clean(p["dir2"]))
    add_question(
        text=f'Which director is Turkish, {d1} from "{f1}" or {d2} from "{f2}"?',
        path=[f1, "DIRECTOR", d1, "COUNTRY_OF_CITIZENSHIP", "Turkiye",
              "vs", f2, "DIRECTOR", d2],
        answer=d1, difficulty="comparison",
        template="Film1.Director(Turkish) vs Film2.Director(Foreign)")

comp_count = len(questions) - comp_start
print(f"  Karşılaştırma soru sayısı: {comp_count}")

# -----------------------------------------------------------------------------
# EK HAVUZ
# -----------------------------------------------------------------------------
if len(questions) < TARGET_TOTAL:
    print(f"\n[EK HAVUZ] {TARGET_TOTAL - len(questions)} soru eksik...")
    extra_pools = [
        (paths_2A, lambda r: (
            f'Where was the person who directed "{clean(r["film"])}" born?',
            [clean(r["film"]), "DIRECTOR", clean(r["director"]), "PLACE_OF_BIRTH", clean(r["city"])],
            r["city"]), "2-hop", "Film->DIRECTOR->PLACE_OF_BIRTH (extra)"),
        (paths_2B, lambda r: (
            f'What is the birthplace of {clean(r["actor"])} in "{clean(r["film"])}"?',
            [clean(r["film"]), "CAST_MEMBER", clean(r["actor"]), "PLACE_OF_BIRTH", clean(r["city"])],
            r["city"]), "2-hop", "Film->CAST_MEMBER->PLACE_OF_BIRTH (extra)"),
        (paths_2D, lambda r: (
            f'Which country is {clean(r["actor"])} from "{clean(r["film"])}" a citizen of?',
            [clean(r["film"]), "CAST_MEMBER", clean(r["actor"]), "COUNTRY_OF_CITIZENSHIP", clean(r["country"])],
            r["country"]), "2-hop", "Film->CAST_MEMBER->COUNTRY_OF_CITIZENSHIP (extra)"),
    ]
    for pool, make_fn, diff, tmpl in extra_pools:
        if len(questions) >= TARGET_TOTAL:
            break
        fill_from_pool(pool, TARGET_TOTAL - len(questions), make_fn, diff, tmpl)

# -----------------------------------------------------------------------------
# ÖZET VE KAYIT
# -----------------------------------------------------------------------------
total      = len(questions)
two_hop    = len([q for q in questions if q["difficulty"] == "2-hop"])
three_hop  = len([q for q in questions if q["difficulty"] == "3-hop"])
comparison = len([q for q in questions if q["difficulty"] == "comparison"])

criteria = {
    "2hop_ok":  two_hop    >= TARGET_2HOP,
    "3hop_ok":  three_hop  >= TARGET_3HOP,
    "comp_ok":  comparison >= TARGET_COMP,
    "total_ok": total      >= TARGET_TOTAL,
}

print("\n" + "=" * 65)
print("  CINEMA SORU ÜRETİMİ TAMAMLANDI")
print(f"  Toplam soru        : {total}")
print(f"  Filtrelenen kirli  : {filtered_count}")
print(f"  2-hop              : {two_hop}  ({'OK' if criteria['2hop_ok'] else 'EKSİK'}, min {TARGET_2HOP})")
print(f"  3-hop              : {three_hop}  ({'OK' if criteria['3hop_ok'] else 'EKSİK'}, min {TARGET_3HOP})")
print(f"  Karşılaştırma      : {comparison}  ({'OK' if criteria['comp_ok'] else 'EKSİK'}, min {TARGET_COMP})")
print(f"  Toplam             : {total}  ({'OK' if criteria['total_ok'] else 'EKSİK'}, min {TARGET_TOTAL})")
print("=" * 65)


def save_json(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Kaydedildi: {path}")


save_json(questions, "qa_dataset.json")
save_json({
    "domain":          "cinema",
    "total_questions": total,
    "filtered_noisy":  filtered_count,
    "by_difficulty":   {"2-hop": two_hop, "3-hop": three_hop, "comparison": comparison},
    "criteria_met":    criteria,
    "template_distribution": template_counts,
    "phase2_pipeline": {
        "source_report":      os.path.relpath(PHASE2_PATH),
        "verified_film_rels": sorted(FILM_RELS),
        "zero_path_templates": {
            "2hop": ZERO_PATH_TEMPLATES_2H,
            "3hop": ZERO_PATH_TEMPLATES_3H,
        },
        "phase2_path_counts": {"2hop": PATH_COUNTS_2H, "3hop": PATH_COUNTS_3H},
        "derived_limits":     {"2hop": L2, "3hop": L3},
    },
}, "qa_dataset_summary.json")

driver.close()
print("Phase 3 cinema tamamlandı!")