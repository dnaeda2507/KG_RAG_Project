"""
phase3_football_question_generator.py
======================================
Düzeltme:
  - _is_valid_answer(): LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY
    hedefi bazen Wikidata artifact entity'lerine işaret eder.
    "Archbishopric of Adrianopolis", "List of people from Malatya",
    "Theodosiopolis (Armenia)" gibi gold answer'lar filtrelenir.
    Bu sayede dataset'e kirli soru girmez, metrik hesabı doğru kalır.
"""

import os
import sys
import json
import math
import random
import re

from neo4j import GraphDatabase

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, TURKEY_ID, OUTPUT_PHASE3_FOOTBALL

driver     = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
OUTPUT_DIR = OUTPUT_PHASE3_FOOTBALL
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# ADIM 0 -- Phase 2 çıktısını yükle
# -----------------------------------------------------------------------------
_HERE          = os.path.dirname(os.path.abspath(__file__))
PHASE2_REPORT  = os.path.join(_HERE, "..", "..", "outputs",
                               "phase2_domain_selection", "football",
                               "football_path_density_report.json")
PHASE2_ENTITIES = os.path.join(_HERE, "..", "..", "outputs",
                                "phase2_domain_selection", "football",
                                "football_main_entities.json")

print("=" * 65)
print("  PHASE 3 -- Türk Futbolu Soru Üretimi")
print("  Pipeline: Phase 2 -> Phase 3")
print("=" * 65)

for path in [PHASE2_REPORT, PHASE2_ENTITIES]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Phase 2 dosyası bulunamadı: {path}\n"
            "Önce phase2_football_verification.py'yi çalıştırın."
        )

with open(PHASE2_REPORT,   encoding="utf-8") as f:
    PHASE2 = json.load(f)
with open(PHASE2_ENTITIES, encoding="utf-8") as f:
    ENTITIES = json.load(f)

SEED_IDS       = [e["entity_id"] for e in ENTITIES]
CLUB_RELS      = set(PHASE2["summary"]["unique_rel_types"])
PATH_COUNTS_2H = PHASE2["path_type_breakdown"]["2hop"].copy()
PATH_COUNTS_3H = PHASE2["path_type_breakdown"]["3hop"].copy()

ZERO_PATH_TEMPLATES_2H = sorted([k for k, v in PATH_COUNTS_2H.items() if v == 0])
ZERO_PATH_TEMPLATES_3H = sorted([k for k, v in PATH_COUNTS_3H.items() if v == 0])

TOTAL_2H = sum(PATH_COUNTS_2H.values())
TOTAL_3H = sum(PATH_COUNTS_3H.values())

missing = {r for r in ["HOME_VENUE", "LEAGUE", "COUNTRY"] if r not in CLUB_RELS}
if missing:
    print(f"\n  UYARI: {missing} Phase 2 raporunda eksik!")

print(f"\n  Phase 2 raporu yüklendi:")
print(f"    Seed entity sayısı   : {len(SEED_IDS)}")
print(f"    Kulüp relation'ları  : {sorted(CLUB_RELS)}")
print(f"    2-hop path toplam    : {TOTAL_2H}")
print(f"    3-hop path toplam    : {TOTAL_3H}")


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
# LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY bazen Wikidata'da
# tarihi/arşiv entity'lerine işaret eder. Bu gold answer'lar dataset'e
# girerse metrik hesabı bozulur — filtreliyoruz.
_NOISY_PATTERNS = [
    r"^archbishopric\b",
    r"^bishopric\b",
    r"^list of\b",
    r"^theodosiopolis\b",
    r"^nicaea\b",
    r"^ancient\b",
    r"\(ancient\)",
    r"\(historical\)",
    r"\(armenia\)",
    r"^diocese\b",
    r"^patriarchate\b",
    r"^metropolitan\b",
    r"^eparchy\b",
]


def _is_valid_answer(answer: str) -> bool:
    """
    Gold answer'ın anlamlı ve temiz olup olmadığını kontrol eder.
    Wikidata artifact / tarihi entity adları filtrelenir.
    """
    if not answer or answer.strip().lower() in ("unknown", "none", ""):
        return False
    a = answer.strip().lower()
    # Kirli pattern kontrolü
    if any(re.search(pat, a) for pat in _NOISY_PATTERNS):
        return False
    # Çok uzun cevaplar genellikle Wikidata açıklaması sızmış demektir
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
filtered_count  = 0  # Filtrelenen kirli gold answer sayısı


def add_question(text: str, path: list, answer: str,
                 difficulty: str, template: str) -> bool:
    global q_id, filtered_count
    # Gold answer kalite kontrolü
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
        "domain":         "football",
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

# 2A: Club <-MEMBER_OF_SPORTS_TEAM- Player -> PLACE_OF_BIRTH -> City
paths_2A = run(
    "MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity) "
    "WHERE club.entityId IN $ids "
    "MATCH (player)-[:PLACE_OF_BIRTH]->(city:Entity) "
    "WHERE player.name IS NOT NULL AND club.name IS NOT NULL AND city.name IS NOT NULL "
    "RETURN player.name AS player, club.name AS club, city.name AS city LIMIT 400",
    {"ids": SEED_IDS})
fill_from_pool(paths_2A, L2.get("A_player_club_city", 6),
    lambda r: (
        f"Where was {clean(r['player'])} born?",
        [clean(r["club"]), "<-MEMBER_OF_SPORTS_TEAM-", clean(r["player"]),
         "PLACE_OF_BIRTH", clean(r["city"])],
        r["city"]),
    "2-hop", "Club<-MEMBER_OF_SPORTS_TEAM-Player->PLACE_OF_BIRTH->City")

# 2B: Club -> HOME_VENUE -> Stadium -> LOCATED_IN -> Place
paths_2B = run(
    "MATCH (club:Entity)-[:HOME_VENUE]->(stadium:Entity) "
    "WHERE club.entityId IN $ids "
    "MATCH (stadium)-[:LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY]->(place:Entity) "
    "WHERE club.name IS NOT NULL AND stadium.name IS NOT NULL AND place.name IS NOT NULL "
    "RETURN club.name AS club, stadium.name AS stadium, place.name AS place LIMIT 300",
    {"ids": SEED_IDS})
fill_from_pool(paths_2B, L2.get("B_club_stadium_place", 6),
    lambda r: (
        f"In which city or region is the stadium of {clean(r['club'])} located?",
        [clean(r["club"]), "HOME_VENUE", clean(r["stadium"]),
         "LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY", clean(r["place"])],
        r["place"]),
    "2-hop", "Club->HOME_VENUE->Stadium->LOCATED_IN->Place")

# 2D: Player -> MEMBER_OF_SPORTS_TEAM -> Club -> COUNTRY -> Country
paths_2D = run(
    "MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity) "
    "WHERE club.entityId IN $ids "
    "MATCH (club)-[:COUNTRY]->(country:Entity) "
    "WHERE player.name IS NOT NULL AND club.name IS NOT NULL AND country.name IS NOT NULL "
    "RETURN player.name AS player, club.name AS club, country.name AS country LIMIT 400",
    {"ids": SEED_IDS})
fill_from_pool(paths_2D, L2.get("D_player_club_country", 6),
    lambda r: (
        f"Which country is the club of {clean(r['player'])} in?",
        [clean(r["player"]), "MEMBER_OF_SPORTS_TEAM", clean(r["club"]),
         "COUNTRY", clean(r["country"])],
        r["country"]),
    "2-hop", "Player->MEMBER_OF_SPORTS_TEAM->Club->COUNTRY->Country")

# 2E: Player -> MEMBER_OF_SPORTS_TEAM -> Club -> LEAGUE -> League
paths_2E = run(
    "MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity) "
    "WHERE club.entityId IN $ids "
    "MATCH (club)-[:LEAGUE]->(league:Entity) "
    "WHERE player.name IS NOT NULL AND club.name IS NOT NULL AND league.name IS NOT NULL "
    "RETURN player.name AS player, club.name AS club, league.name AS league LIMIT 400",
    {"ids": SEED_IDS})
fill_from_pool(paths_2E, L2.get("E_player_club_league", 6),
    lambda r: (
        f"Which league does the club {clean(r['club'])}, where {clean(r['player'])} played, belong to?",
        [clean(r["player"]), "MEMBER_OF_SPORTS_TEAM", clean(r["club"]),
         "LEAGUE", clean(r["league"])],
        r["league"]),
    "2-hop", "Player->MEMBER_OF_SPORTS_TEAM->Club->LEAGUE->League")

# 2F: Player -> PLACE_OF_BIRTH -> City -> COUNTRY -> Country
paths_2F = run(
    "MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity) "
    "WHERE club.entityId IN $ids "
    "MATCH (player)-[:PLACE_OF_BIRTH]->(city:Entity) "
    "MATCH (city)-[:COUNTRY]->(country:Entity) "
    "WHERE player.name IS NOT NULL AND city.name IS NOT NULL AND country.name IS NOT NULL "
    "RETURN player.name AS player, club.name AS club, "
    "city.name AS city, country.name AS country LIMIT 400",
    {"ids": SEED_IDS})
fill_from_pool(paths_2F, L2.get("F_player_city_country", 6),
    lambda r: (
        f"Which country is {clean(r['city'])}, the birthplace of {clean(r['player'])}, in?",
        [clean(r["player"]), "PLACE_OF_BIRTH", clean(r["city"]),
         "COUNTRY", clean(r["country"])],
        r["country"]),
    "2-hop", "Player->PLACE_OF_BIRTH->City->COUNTRY->Country")

# 2G: Club -> LEAGUE -> League -> COUNTRY -> Country
paths_2G = run(
    "MATCH (club:Entity)-[:LEAGUE]->(league:Entity) "
    "WHERE club.entityId IN $ids "
    "MATCH (league)-[:COUNTRY]->(country:Entity) "
    "WHERE club.name IS NOT NULL AND league.name IS NOT NULL AND country.name IS NOT NULL "
    "RETURN club.name AS club, league.name AS league, country.name AS country LIMIT 300",
    {"ids": SEED_IDS})
fill_from_pool(paths_2G, L2.get("G_club_league_country", 3),
    lambda r: (
        f"Which country is the league of {clean(r['club'])} in?",
        [clean(r["club"]), "LEAGUE", clean(r["league"]),
         "COUNTRY", clean(r["country"])],
        r["country"]),
    "2-hop", "Club->LEAGUE->League->COUNTRY->Country")

hop2_count = len(questions)
print(f"  2-hop soru sayısı  : {hop2_count}")
print(f"  Filtrelenen (kirli): {filtered_count}")

# -----------------------------------------------------------------------------
# 3-HOP SORGULAR
# -----------------------------------------------------------------------------
print("\n[3-HOP] Sorgular çalıştırılıyor...")
hop3_start     = len(questions)
filtered_before = filtered_count

# 3A: Player -> Club -> HOME_VENUE -> Stadium -> LOCATED_IN -> Place
paths_3A = run(
    "MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity) "
    "WHERE club.entityId IN $ids "
    "MATCH (club)-[:HOME_VENUE]->(stadium:Entity) "
    "MATCH (stadium)-[:LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY]->(place:Entity) "
    "WHERE player.name IS NOT NULL AND club.name IS NOT NULL "
    "AND stadium.name IS NOT NULL AND place.name IS NOT NULL "
    "RETURN player.name AS player, club.name AS club, "
    "stadium.name AS stadium, place.name AS place LIMIT 400",
    {"ids": SEED_IDS})
fill_from_pool(paths_3A, L3.get("3A_player_club_stadium_place", 4),
    lambda r: (
        f"In which city or region is {clean(r['stadium'])}, the home stadium of "
        f"{clean(r['club'])} where {clean(r['player'])} played, located?",
        [clean(r["player"]), "MEMBER_OF_SPORTS_TEAM", clean(r["club"]),
         "HOME_VENUE", clean(r["stadium"]),
         "LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY", clean(r["place"])],
        r["place"]),
    "3-hop", "Player->Club->HOME_VENUE->Stadium->LOCATED_IN->Place")

# 3B: Club <- Player -> PLACE_OF_BIRTH -> City -> COUNTRY -> Country
paths_3B = run(
    "MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity) "
    "WHERE club.entityId IN $ids "
    "MATCH (player)-[:PLACE_OF_BIRTH]->(city:Entity) "
    "MATCH (city)-[:COUNTRY]->(country:Entity) "
    "WHERE player.name IS NOT NULL AND club.name IS NOT NULL "
    "AND city.name IS NOT NULL AND country.name IS NOT NULL "
    "RETURN player.name AS player, club.name AS club, "
    "city.name AS city, country.name AS country LIMIT 400",
    {"ids": SEED_IDS})
fill_from_pool(paths_3B, L3.get("3B_player_club_city_country", 4),
    lambda r: (
        f"Which country is {clean(r['city'])}, the birthplace of "
        f"{clean(r['player'])} from {clean(r['club'])}, in?",
        [clean(r["club"]), "<-MEMBER_OF_SPORTS_TEAM-", clean(r["player"]),
         "PLACE_OF_BIRTH", clean(r["city"]),
         "COUNTRY", clean(r["country"])],
        r["country"]),
    "3-hop", "Player->Club (Player->PLACE_OF_BIRTH->City->COUNTRY->Country)")

# 3D: Player -> Club -> LEAGUE -> League -> COUNTRY -> Country
paths_3D = run(
    "MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity) "
    "WHERE club.entityId IN $ids "
    "MATCH (club)-[:LEAGUE]->(league:Entity) "
    "MATCH (league)-[:COUNTRY]->(country:Entity) "
    "WHERE player.name IS NOT NULL AND club.name IS NOT NULL "
    "AND league.name IS NOT NULL AND country.name IS NOT NULL "
    "RETURN player.name AS player, club.name AS club, "
    "league.name AS league, country.name AS country LIMIT 400",
    {"ids": SEED_IDS})
fill_from_pool(paths_3D, L3.get("3D_player_club_league_country", 4),
    lambda r: (
        f"Which country is the league of {clean(r['club'])}, "
        f"where {clean(r['player'])} played, in?",
        [clean(r["player"]), "MEMBER_OF_SPORTS_TEAM", clean(r["club"]),
         "LEAGUE", clean(r["league"]), "COUNTRY", clean(r["country"])],
        r["country"]),
    "3-hop", "Player->Club->LEAGUE->League->COUNTRY->Country")

# 3E: Player -> Club -> HOME_VENUE -> Stadium -> COUNTRY -> Country
paths_3E = run(
    "MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity) "
    "WHERE club.entityId IN $ids "
    "MATCH (club)-[:HOME_VENUE]->(stadium:Entity) "
    "MATCH (stadium)-[:COUNTRY]->(country:Entity) "
    "WHERE player.name IS NOT NULL AND club.name IS NOT NULL "
    "AND stadium.name IS NOT NULL AND country.name IS NOT NULL "
    "RETURN player.name AS player, club.name AS club, "
    "stadium.name AS stadium, country.name AS country LIMIT 400",
    {"ids": SEED_IDS})
fill_from_pool(paths_3E, L3.get("3E_player_club_stadium_country", 4),
    lambda r: (
        f"Which country is {clean(r['stadium'])}, the stadium of "
        f"{clean(r['club'])} where {clean(r['player'])} played, in?",
        [clean(r["player"]), "MEMBER_OF_SPORTS_TEAM", clean(r["club"]),
         "HOME_VENUE", clean(r["stadium"]), "COUNTRY", clean(r["country"])],
        r["country"]),
    "3-hop", "Player->Club->HOME_VENUE->Stadium->COUNTRY->Country")

hop3_count = len(questions) - hop3_start
print(f"  3-hop soru sayısı  : {hop3_count}")
print(f"  Filtrelenen (kirli): {filtered_count - filtered_before}")

# -----------------------------------------------------------------------------
# KARŞILAŞTIRMA SORULARI
# -----------------------------------------------------------------------------
print("\n[COMPARISON] Sorgular çalıştırılıyor...")
comp_start = len(questions)

player_comp = run(
    "MATCH (club:Entity) WHERE club.entityId IN $ids "
    "MATCH (p1:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club) "
    "MATCH (p1)-[:PLACE_OF_BIRTH]->(c1:Entity)-[:COUNTRY]->(:Entity {entityId: $tid}) "
    "MATCH (p2:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club) "
    "MATCH (p2)-[:PLACE_OF_BIRTH]->(c2:Entity)-[:COUNTRY]->(other:Entity) "
    "WHERE p1.entityId <> p2.entityId AND other.entityId <> $tid "
    "AND club.name IS NOT NULL AND p1.name IS NOT NULL AND p2.name IS NOT NULL "
    "RETURN club.name AS club, p1.name AS tr_player, p2.name AS foreign_player LIMIT 200",
    {"ids": SEED_IDS, "tid": TURKEY_ID})
random.shuffle(player_comp)
for p in player_comp[:5]:
    club, p1, p2 = clean(p["club"]), clean(p["tr_player"]), clean(p["foreign_player"])
    add_question(
        text=f"Which player from {club} was born in Turkey, {p1} or {p2}?",
        path=[p1, "MEMBER_OF_SPORTS_TEAM", club, "PLACE_OF_BIRTH", "Turkey",
              "vs", p2, "MEMBER_OF_SPORTS_TEAM", club],
        answer=p1, difficulty="comparison",
        template="Club.Player(BornTurkey) vs Club.Player(BornForeign)")

tr_players = run(
    "MATCH (p:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity) "
    "WHERE club.entityId IN $ids AND p.name IS NOT NULL AND club.name IS NOT NULL "
    "RETURN p.name AS player, club.name AS club LIMIT 50",
    {"ids": SEED_IDS})
foreign_players = run(
    "MATCH (p:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity) "
    "WHERE NOT club.entityId IN $ids AND p.name IS NOT NULL AND club.name IS NOT NULL "
    "MATCH (p)-[:PLACE_OF_BIRTH]->(c:Entity)-[:COUNTRY]->(other:Entity) "
    "WHERE other.entityId <> $tid "
    "RETURN p.name AS player, club.name AS club LIMIT 50",
    {"ids": SEED_IDS, "tid": TURKEY_ID})
random.shuffle(tr_players)
random.shuffle(foreign_players)
for i in range(min(4, len(tr_players), len(foreign_players))):
    p1 = tr_players[i]
    p2 = foreign_players[i]
    add_question(
        text=f"Which player plays for a Turkish football club, "
             f"{clean(p1['player'])} or {clean(p2['player'])}?",
        path=[clean(p1["player"]), "MEMBER_OF_SPORTS_TEAM", clean(p1["club"]),
              "vs", clean(p2["player"]), "MEMBER_OF_SPORTS_TEAM", clean(p2["club"])],
        answer=p1["player"], difficulty="comparison",
        template="Player(TurkishClub) vs Player(ForeignClub)")

comp_count = len(questions) - comp_start
print(f"  Karşılaştırma soru sayısı: {comp_count}")

# -----------------------------------------------------------------------------
# EK HAVUZ -- Toplam 50'ye tamamla
# -----------------------------------------------------------------------------
if len(questions) < TARGET_TOTAL:
    deficit_total = TARGET_TOTAL - len(questions)
    print(f"\n[EK HAVUZ] Toplam eksik={deficit_total}")
    extra = [
        (paths_2A, lambda r: (
            f"Where was football player {clean(r['player'])} born?",
            [clean(r["club"]), "<-MEMBER_OF_SPORTS_TEAM-", clean(r["player"]),
             "PLACE_OF_BIRTH", clean(r["city"])], r["city"]),
         "2-hop", "Club<-MEMBER_OF_SPORTS_TEAM-Player->PLACE_OF_BIRTH->City (extra)"),
        (paths_2D, lambda r: (
            f"Which country is the club of {clean(r['player'])} in?",
            [clean(r["player"]), "MEMBER_OF_SPORTS_TEAM", clean(r["club"]),
             "COUNTRY", clean(r["country"])], r["country"]),
         "2-hop", "Player->MEMBER_OF_SPORTS_TEAM->Club->COUNTRY->Country (extra)"),
        (paths_3A, lambda r: (
            f"Where is {clean(r['stadium'])}, the stadium of {clean(r['club'])}, located?",
            [clean(r["player"]), "MEMBER_OF_SPORTS_TEAM", clean(r["club"]),
             "HOME_VENUE", clean(r["stadium"]),
             "LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY", clean(r["place"])],
            r["place"]),
         "3-hop", "Player->Club->HOME_VENUE->Stadium->LOCATED_IN->Place (extra)"),
    ]
    for pool, make_fn, diff, tmpl in extra:
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
print("  FOOTBALL SORU ÜRETİMİ TAMAMLANDI")
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
    "domain":          "football",
    "total_questions": total,
    "filtered_noisy":  filtered_count,
    "by_difficulty": {
        "2-hop":      two_hop,
        "3-hop":      three_hop,
        "comparison": comparison,
    },
    "criteria_met": criteria,
    "template_distribution": template_counts,
    "phase2_pipeline": {
        "source_report":      os.path.relpath(PHASE2_REPORT),
        "seed_entity_count":  len(SEED_IDS),
        "verified_club_rels": sorted(CLUB_RELS),
        "zero_path_templates": {
            "2hop": ZERO_PATH_TEMPLATES_2H,
            "3hop": ZERO_PATH_TEMPLATES_3H,
        },
        "phase2_path_counts": {
            "2hop": PATH_COUNTS_2H,
            "3hop": PATH_COUNTS_3H,
        },
        "derived_limits": {
            "2hop": L2,
            "3hop": L3,
        },
    },
}, "qa_dataset_summary.json")

driver.close()
print("Phase 3 football tamamlandı!")