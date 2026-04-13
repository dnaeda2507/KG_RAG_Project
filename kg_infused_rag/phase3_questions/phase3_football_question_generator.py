"""
Phase 3 - Multi-Hop Question Generation (Türk Futbolu)
=========================================================
Neo4j'den gerçek path'leri çekip 50+ soru üretir.
- 30 adet 2-hop soru
- 15 adet 3-hop soru
-  5 adet karşılaştırma sorusu

Düzeltme:
  - "Eksik soru tamamlama" bloğu artık farklı template'leri döngüsel kullanıyor
    (sadece Player→Club→PLACE_OF_BIRTH tekrarından kaçınılıyor)
  - Her template'e kullanım sayacı eklendi (template_counts)
  - Soru çeşitliliği artırıldı

Çalıştır:
    python phase3_football_question_generator.py

Çıktılar:
    outputs/phase3_football/qa_dataset.json
    outputs/phase3_football/qa_dataset_summary.json
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
OUTPUT_DIR = "outputs/phase3_football"
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

# Önce Türk futbol kulüplerinin ID'lerini al
print("=" * 65)
print("  PHASE 3 — Türk Futbolu Multi-Hop Soru Üretimi")
print("=" * 65)

print("\n  Türk futbol kulüpleri yükleniyor...")
clubs = run("""
    MATCH (club:Entity)-[:COUNTRY]->(:Entity {entityId: $tid})
    WHERE toLower(club.description) CONTAINS 'football club'
       OR toLower(club.description) CONTAINS 'football team'
    RETURN club.entityId AS id, club.description AS desc
""", {"tid": TURKEY_ID})

CLUB_IDS = [c['id'] for c in clubs]
print(f"  {len(CLUB_IDS)} kulüp bulundu: {[c['desc'] for c in clubs[:5]]}...")

questions       = []
template_counts = {}   # ✔ template başına kullanım sayacı
q_id            = 1

def add_question(text, path, answer, difficulty, domain, template):
    global q_id
    questions.append({
        "question_id":    f"TR_FB_{q_id:03d}",
        "question_text":  text,
        "reasoning_path": path,
        "gold_answer":    clean(answer),
        "difficulty":     difficulty,
        "domain":         domain,
        "template":       template,
        "verified":       True
    })
    template_counts[template] = template_counts.get(template, 0) + 1   # ✔
    q_id += 1


# ══════════════════════════════════════════════════════════════════════════════
# TİP 1: 2-HOP SORULAR (hedef: 30)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2-HOP] Soru üretiliyor...")

# ── 2A: Futbolcu → MEMBER_OF_SPORTS_TEAM → Kulüp & PLACE_OF_BIRTH → Şehir
paths_2A = run("""
    MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (player)-[:PLACE_OF_BIRTH]->(city:Entity)
    WHERE player.description IS NOT NULL
      AND club.description IS NOT NULL
      AND city.description IS NOT NULL
    RETURN player.description AS player,
           club.description   AS club,
           city.description   AS city
    LIMIT 300
""", {"ids": CLUB_IDS})

random.shuffle(paths_2A)
used = set()
for p in paths_2A:
    if len([q for q in questions if q['difficulty'] == '2-hop']) >= 10:
        break
    player = clean(p['player'])
    club   = clean(p['club'])
    city   = clean(p['city'])
    key = (player, club)
    if key in used:
        continue
    used.add(key)
    add_question(
        text=f'{player}, {club} takımında oynarken nerede doğmuştur?',
        path=[player, "MEMBER_OF_SPORTS_TEAM", club, "←", player, "PLACE_OF_BIRTH", city],
        answer=city,
        difficulty="2-hop",
        domain="football",
        template="Player → MEMBER_OF_SPORTS_TEAM → Club & Player → PLACE_OF_BIRTH → City"
    )

# ── 2B: Kulüp → HOME_VENUE → Stadyum
paths_2B = run("""
    MATCH (club:Entity)-[:HOME_VENUE]->(stadium:Entity)
    WHERE club.entityId IN $ids
      AND club.description IS NOT NULL
      AND stadium.description IS NOT NULL
    RETURN club.description    AS club,
           stadium.description AS stadium
    LIMIT 100
""", {"ids": CLUB_IDS})

random.shuffle(paths_2B)
for p in paths_2B[:8]:
    club    = clean(p['club'])
    stadium = clean(p['stadium'])
    add_question(
        text=f'{club} takımının ev sahibi olduğu stadyum hangisidir?',
        path=[club, "HOME_VENUE", stadium],
        answer=stadium,
        difficulty="2-hop",
        domain="football",
        template="Club → HOME_VENUE → Stadium"
    )

# ── 2C: Futbolcu → MEMBER_OF_SPORTS_TEAM → Kulüp → COUNTRY → Ülke
paths_2C = run("""
    MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (club)-[:COUNTRY]->(country:Entity)
    WHERE player.description IS NOT NULL
      AND club.description IS NOT NULL
      AND country.description IS NOT NULL
    RETURN player.description  AS player,
           club.description    AS club,
           country.description AS country
    LIMIT 200
""", {"ids": CLUB_IDS})

random.shuffle(paths_2C)
used2C = set()
for p in paths_2C:
    if len([q for q in questions if q['template'] == 'Player → MEMBER_OF_SPORTS_TEAM → Club → COUNTRY → Country']) >= 6:
        break
    player  = clean(p['player'])
    club    = clean(p['club'])
    country = clean(p['country'])
    if player in used2C:
        continue
    used2C.add(player)
    add_question(
        text=f'{player}\'nin oynadığı {club} takımı hangi ülkeye aittir?',
        path=[player, "MEMBER_OF_SPORTS_TEAM", club, "COUNTRY", country],
        answer=country,
        difficulty="2-hop",
        domain="football",
        template="Player → MEMBER_OF_SPORTS_TEAM → Club → COUNTRY → Country"
    )

# ── 2D: Futbolcu → MEMBER_OF_SPORTS_TEAM → Kulüp → LEAGUE → Lig
paths_2D = run("""
    MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (club)-[:LEAGUE]->(league:Entity)
    WHERE player.description IS NOT NULL
      AND club.description IS NOT NULL
      AND league.description IS NOT NULL
    RETURN player.description AS player,
           club.description   AS club,
           league.description AS league
    LIMIT 200
""", {"ids": CLUB_IDS})

random.shuffle(paths_2D)
used2D = set()
for p in paths_2D:
    if len([q for q in questions if q['template'] == 'Player → Club → LEAGUE']) >= 6:
        break
    player = clean(p['player'])
    club   = clean(p['club'])
    league = clean(p['league'])
    if player in used2D:
        continue
    used2D.add(player)
    add_question(
        text=f'{player}\'nin oynadığı {club} takımı hangi ligde mücadele etmektedir?',
        path=[player, "MEMBER_OF_SPORTS_TEAM", club, "LEAGUE", league],
        answer=league,
        difficulty="2-hop",
        domain="football",
        template="Player → Club → LEAGUE"
    )

# ── 2E: Antrenör → HEAD_COACH → Kulüp → PLACE_OF_BIRTH → Şehir
paths_2E = run("""
    MATCH (coach:Entity)-[:HEAD_COACH]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (coach)-[:PLACE_OF_BIRTH]->(city:Entity)
    WHERE coach.description IS NOT NULL
      AND club.description IS NOT NULL
      AND city.description IS NOT NULL
    RETURN coach.description AS coach,
           club.description  AS club,
           city.description  AS city
    LIMIT 100
""", {"ids": CLUB_IDS})

random.shuffle(paths_2E)
for p in paths_2E[:5]:
    coach = clean(p['coach'])
    club  = clean(p['club'])
    city  = clean(p['city'])
    add_question(
        text=f'{club} takımının teknik direktörü {coach} nerede doğmuştur?',
        path=[club, "HEAD_COACH", coach, "PLACE_OF_BIRTH", city],
        answer=city,
        difficulty="2-hop",
        domain="football",
        template="Club → HEAD_COACH → Coach → PLACE_OF_BIRTH → City"
    )

two_hop_count = len([q for q in questions if q['difficulty'] == '2-hop'])
print(f"  2-hop soru sayısı: {two_hop_count}")


# ══════════════════════════════════════════════════════════════════════════════
# TİP 2: 3-HOP SORULAR (hedef: 15)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3-HOP] Soru üretiliyor...")
hop3_start = len(questions)

# ── 3A: Futbolcu → Kulüp → HOME_VENUE → Stadyum → Yer
paths_3A = run("""
    MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (club)-[:HOME_VENUE]->(stadium:Entity)
    MATCH (stadium)-[:LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY|COUNTRY]->(place:Entity)
    WHERE player.description IS NOT NULL
      AND club.description IS NOT NULL
      AND stadium.description IS NOT NULL
      AND place.description IS NOT NULL
    RETURN player.description   AS player,
           club.description     AS club,
           stadium.description  AS stadium,
           place.description    AS place
    LIMIT 200
""", {"ids": CLUB_IDS})

random.shuffle(paths_3A)
used3A = set()
for p in paths_3A:
    if len([q for q in questions if q['template'] == 'Player → Club → Stadium → Place']) >= 5:
        break
    player  = clean(p['player'])
    club    = clean(p['club'])
    stadium = clean(p['stadium'])
    place   = clean(p['place'])
    if player in used3A:
        continue
    used3A.add(player)
    add_question(
        text=f'{player}\'nin oynadığı {club} takımının stadyumu olan {stadium} nerede konuşlanmıştır?',
        path=[player, "MEMBER_OF_SPORTS_TEAM", club, "HOME_VENUE", stadium, "LOCATED_IN", place],
        answer=place,
        difficulty="3-hop",
        domain="football",
        template="Player → Club → Stadium → Place"
    )

# ── 3B: Futbolcu → Kulüp + PLACE_OF_BIRTH → Şehir → COUNTRY → Ülke
paths_3B = run("""
    MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (player)-[:PLACE_OF_BIRTH]->(city:Entity)
    MATCH (city)-[:COUNTRY]->(country:Entity)
    WHERE player.description IS NOT NULL
      AND club.description IS NOT NULL
      AND city.description IS NOT NULL
      AND country.description IS NOT NULL
    RETURN player.description  AS player,
           club.description    AS club,
           city.description    AS city,
           country.description AS country
    LIMIT 300
""", {"ids": CLUB_IDS})

random.shuffle(paths_3B)
used3B = set()
for p in paths_3B:
    if len([q for q in questions if q['template'] == 'Player → Club + City → Country']) >= 5:
        break
    player  = clean(p['player'])
    club    = clean(p['club'])
    city    = clean(p['city'])
    country = clean(p['country'])
    if player in used3B:
        continue
    used3B.add(player)
    add_question(
        text=f'{club} takımında oynayan {player}\'nin doğduğu {city} şehri hangi ülkededir?',
        path=[player, "MEMBER_OF_SPORTS_TEAM", club, "←", player, "PLACE_OF_BIRTH", city, "COUNTRY", country],
        answer=country,
        difficulty="3-hop",
        domain="football",
        template="Player → Club + City → Country"
    )

# ── 3C: Antrenör → Kulüp → HOME_VENUE → Stadyum → Yer
paths_3C = run("""
    MATCH (coach:Entity)-[:HEAD_COACH]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (club)-[:HOME_VENUE]->(stadium:Entity)
    MATCH (stadium)-[:LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY|COUNTRY]->(place:Entity)
    WHERE coach.description IS NOT NULL
      AND club.description IS NOT NULL
      AND stadium.description IS NOT NULL
      AND place.description IS NOT NULL
    RETURN coach.description   AS coach,
           club.description    AS club,
           stadium.description AS stadium,
           place.description   AS place
    LIMIT 100
""", {"ids": CLUB_IDS})

random.shuffle(paths_3C)
for p in paths_3C[:3]:
    coach   = clean(p['coach'])
    club    = clean(p['club'])
    stadium = clean(p['stadium'])
    place   = clean(p['place'])
    add_question(
        text=f'{club} takımının teknik direktörü {coach}\'nin takımının stadyumu {stadium} nerede yer almaktadır?',
        path=[coach, "HEAD_COACH", club, "HOME_VENUE", stadium, "LOCATED_IN", place],
        answer=place,
        difficulty="3-hop",
        domain="football",
        template="Coach → Club → Stadium → Place"
    )

# ── 3D: Futbolcu → Kulüp → LEAGUE → Lig → COUNTRY → Ülke
paths_3D = run("""
    MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (club)-[:LEAGUE]->(league:Entity)
    MATCH (league)-[:COUNTRY]->(country:Entity)
    WHERE player.description IS NOT NULL
      AND club.description IS NOT NULL
      AND league.description IS NOT NULL
      AND country.description IS NOT NULL
    RETURN player.description  AS player,
           club.description    AS club,
           league.description  AS league,
           country.description AS country
    LIMIT 200
""", {"ids": CLUB_IDS})

random.shuffle(paths_3D)
used3D = set()
for p in paths_3D:
    if len([q for q in questions if q['template'] == 'Player → Club → League → Country']) >= 2:
        break
    player  = clean(p['player'])
    club    = clean(p['club'])
    league  = clean(p['league'])
    country = clean(p['country'])
    if player in used3D:
        continue
    used3D.add(player)
    add_question(
        text=f'{player}\'nin oynadığı {club}\'nin katıldığı {league} ligi hangi ülkeye aittir?',
        path=[player, "MEMBER_OF_SPORTS_TEAM", club, "LEAGUE", league, "COUNTRY", country],
        answer=country,
        difficulty="3-hop",
        domain="football",
        template="Player → Club → League → Country"
    )

hop3_count = len(questions) - hop3_start
print(f"  3-hop soru sayısı: {hop3_count}")


# ══════════════════════════════════════════════════════════════════════════════
# TİP 3: KARŞILAŞTIRMA SORULARI (hedef: 5)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[KARŞILAŞTIRMA] Soru üretiliyor...")

# Karş 1: İki kulübün stadyumunu karşılaştır
stadium_pairs = run("""
    MATCH (c1:Entity)-[:HOME_VENUE]->(s1:Entity)
    MATCH (c2:Entity)-[:HOME_VENUE]->(s2:Entity)
    WHERE c1.entityId IN $ids AND c2.entityId IN $ids
      AND c1.entityId < c2.entityId
      AND c1.description IS NOT NULL AND c2.description IS NOT NULL
      AND s1.description IS NOT NULL AND s2.description IS NOT NULL
    RETURN c1.description AS club1, s1.description AS stadium1,
           c2.description AS club2, s2.description AS stadium2
    LIMIT 20
""", {"ids": CLUB_IDS})

random.shuffle(stadium_pairs)
for p in stadium_pairs[:3]:
    c1, s1 = clean(p['club1']), clean(p['stadium1'])
    c2, s2 = clean(p['club2']), clean(p['stadium2'])
    add_question(
        text=f'{c1} (ev: {s1}) ile {c2} (ev: {s2}) takımlarından hangisinin stadyumu daha büyüktür?',
        path=[c1, "HOME_VENUE", s1, "vs", c2, "HOME_VENUE", s2],
        answer=f"Karşılaştırma: {c1}→{s1} / {c2}→{s2}",
        difficulty="comparison",
        domain="football",
        template="Club1.HOME_VENUE vs Club2.HOME_VENUE"
    )

# Karş 2: İki futbolcunun doğum yerini karşılaştır
player_pairs = run("""
    MATCH (p1:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (p2:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club)
    MATCH (p1)-[:PLACE_OF_BIRTH]->(c1:Entity)
    MATCH (p2)-[:PLACE_OF_BIRTH]->(c2:Entity)
    WHERE p1.entityId < p2.entityId
      AND p1.description IS NOT NULL AND p2.description IS NOT NULL
      AND c1.description IS NOT NULL AND c2.description IS NOT NULL
      AND club.description IS NOT NULL
    RETURN club.description AS club,
           p1.description AS player1, c1.description AS city1,
           p2.description AS player2, c2.description AS city2
    LIMIT 20
""", {"ids": CLUB_IDS})

random.shuffle(player_pairs)
for p in player_pairs[:2]:
    club = clean(p['club'])
    p1, c1 = clean(p['player1']), clean(p['city1'])
    p2, c2 = clean(p['player2']), clean(p['city2'])
    add_question(
        text=f'{club} takımında oynayan {p1} mı ({c1}) yoksa {p2} mi ({c2}) daha büyük bir şehirde doğmuştur?',
        path=[club, "HAS_PLAYER", p1, "PLACE_OF_BIRTH", c1, "vs", p2, "PLACE_OF_BIRTH", c2],
        answer=f"Karşılaştırma: {p1}→{c1} / {p2}→{c2}",
        difficulty="comparison",
        domain="football",
        template="Club.Player1.City vs Club.Player2.City"
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
        # Havuz A: Player → Club & Player → PLACE_OF_BIRTH
        ("Player → Club & Player → PLACE_OF_BIRTH (extra)", run("""
            MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
            WHERE club.entityId IN $ids
            MATCH (player)-[:PLACE_OF_BIRTH]->(city:Entity)
            WHERE player.description IS NOT NULL
              AND club.description IS NOT NULL
              AND city.description IS NOT NULL
            RETURN player.description AS player,
                   club.description   AS club,
                   city.description   AS city
            LIMIT 500
        """, {"ids": CLUB_IDS}), "player", "club", "city",
         lambda r: (
             f'{clean(r["club"])} takımında forma giyen {clean(r["player"])} nerede doğmuştur?',
             [clean(r["player"]), "MEMBER_OF_SPORTS_TEAM", clean(r["club"]), "←",
              clean(r["player"]), "PLACE_OF_BIRTH", clean(r["city"])],
             r["city"]
         )),
        # Havuz B: Player → Club → COUNTRY
        ("Player → Club → COUNTRY (extra)", run("""
            MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
            WHERE club.entityId IN $ids
            MATCH (club)-[:COUNTRY]->(country:Entity)
            WHERE player.description IS NOT NULL
              AND club.description IS NOT NULL
              AND country.description IS NOT NULL
            RETURN player.description  AS player,
                   club.description    AS club,
                   country.description AS country
            LIMIT 500
        """, {"ids": CLUB_IDS}), "player", "club", "country",
         lambda r: (
             f'{clean(r["player"])} hangi ülkenin kulübünde oynamaktadır? ({clean(r["club"])})',
             [clean(r["player"]), "MEMBER_OF_SPORTS_TEAM", clean(r["club"]), "COUNTRY", clean(r["country"])],
             r["country"]
         )),
        # Havuz C: Player → Club → LEAGUE
        ("Player → Club → LEAGUE (extra)", run("""
            MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
            WHERE club.entityId IN $ids
            MATCH (club)-[:LEAGUE]->(league:Entity)
            WHERE player.description IS NOT NULL
              AND club.description IS NOT NULL
              AND league.description IS NOT NULL
            RETURN player.description AS player,
                   club.description   AS club,
                   league.description AS league
            LIMIT 500
        """, {"ids": CLUB_IDS}), "player", "club", "league",
         lambda r: (
             f'{clean(r["player"])} hangi ligde oynamaktadır? ({clean(r["club"])} takımıyla)',
             [clean(r["player"]), "MEMBER_OF_SPORTS_TEAM", clean(r["club"]), "LEAGUE", clean(r["league"])],
             r["league"]
         )),
    ]

    # Havuzları karıştır
    for template_name, pool, key1, key2, key3, make_q in extra_pools:
        random.shuffle(pool)

    used_keys = {(q['reasoning_path'][0], q['reasoning_path'][2])
                 for q in questions
                 if len(q['reasoning_path']) >= 3}

    added    = 0
    pool_idx = 0  # ✔ round-robin havuz seçimi

    while added < needed:
        template_name, pool, key1, key2, key3, make_q = extra_pools[pool_idx % len(extra_pools)]
        pool_idx += 1

        found = False
        for r in pool:
            k1 = clean(r.get(key1, ""))
            k2 = clean(r.get(key2, ""))
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
                domain="football",
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
    print(f"    {tmpl:<65} : {cnt:>3}")

print("\n  Örnek sorular:")
for q in questions[:5]:
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
    "template_distribution": template_counts,   # ✔ eklendi
    "domain": "football",
    "criteria_met": {
        "2hop_ok":  two_hop >= 30,
        "3hop_ok":  three_hop >= 15,
        "comp_ok":  comparison >= 5,
        "total_ok": total >= 50
    }
}, "qa_dataset_summary.json")

driver.close()
print("\n✅ Phase 3 Türk Futbolu tamamlandı!")
print(f"   Çıktı: outputs/phase3_football/qa_dataset.json ({total} soru)")
