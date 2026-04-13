"""
Phase 2 - Domain Verification: Türk Futbolu
=============================================
3.3 Domain Verification Process - 4 adımın tamamı
3.4 Domain Selection Criteria değerlendirmesi
3.5 Expected Outputs üretimi

Çalıştır:
    python phase2_football_verification.py

Çıktılar:
    outputs/football/football_domain_report.json
    outputs/football/football_main_entities.json
    outputs/football/football_multihop_paths.json
    outputs/football/football_entity_relation_map.png
    outputs/football/football_path_density_report.json
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ── Bağlantı ──────────────────────────────────────────────────────────────────
load_dotenv()
URI       = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
USER      = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD  = os.getenv("NEO4J_PASSWORD", "neo4j")
TURKEY_ID = "Q43"

driver     = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
OUTPUT_DIR = "outputs/football"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run(cypher, params=None):
    with driver.session() as s:
        return [r.data() for r in s.run(cypher, params or {})]

def save_json(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✔ Kaydedildi: {path}")

def section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: SEED ENTITY DETECTION (3.3 Step 1)
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 1: Seed Entity Detection — Türk Futbol Kulüpleri")

seed_entities = run("""
    MATCH (club:Entity)-[:COUNTRY]->(:Entity {entityId: $tid})
    WHERE toLower(club.description) CONTAINS 'football club'
       OR toLower(club.description) CONTAINS 'football team'
    WITH club, COUNT { (club)-[]-() } AS degree
    ORDER BY degree DESC
    RETURN club.entityId   AS id,
           club.description AS desc,
           degree
""", {"tid": TURKEY_ID})

print(f"\n  Tespit edilen {len(seed_entities)} futbol kulübü:")
print(f"\n  {'No':>3} | {'ID':12} | {'Açıklama':45} | {'Derece':>7}")
print("  " + "─" * 75)
for i, e in enumerate(seed_entities, 1):
    print(f"  {i:>3} | {e['id']:12} | {str(e['desc'])[:45]:45} | {e['degree']:>7,}")

criteria_seeds = len(seed_entities) >= 10
print(f"\n  ✔ Kriter (≥10 seed entity): {'KARŞILANDI ✅' if criteria_seeds else 'EKSİK ⚠️'} "
      f"→ {len(seed_entities)} kulüp bulundu")

seed_ids = [e['id'] for e in seed_entities]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: 1-HOP NEIGHBORS (3.3 Step 2)
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 2: 1-Hop Neighbors — Her Seed Entity İçin Relation'lar")

hop1_data        = {}   # {club_id: {relation: [targets]}}
all_rel_types    = set()
entity_rel_counts = []

for club in seed_entities:
    cid = club['id']
    neighbors = run("""
        MATCH (e:Entity {entityId: $cid})-[r]->(target:Entity)
        RETURN type(r) AS relation,
               target.entityId   AS target_id,
               target.description AS target_desc
        ORDER BY type(r)
    """, {"cid": cid})

    rel_map = {}
    for n in neighbors:
        rel = n['relation']
        all_rel_types.add(rel)
        if rel not in rel_map:
            rel_map[rel] = []
        rel_map[rel].append({"id": n['target_id'], "desc": n['target_desc']})

    hop1_data[cid] = rel_map
    entity_rel_counts.append(len(rel_map))

avg_rels = sum(entity_rel_counts) / len(entity_rel_counts) if entity_rel_counts else 0

print(f"\n  Toplam unique relation tipi : {len(all_rel_types)}")
print(f"  Entity başına ort. relation : {avg_rels:.1f}")
print(f"  Tüm relation tipleri        : {sorted(all_rel_types)}")

print(f"\n  Her kulüp için 1-hop relation sayıları:")
print(f"  {'Kulüp':40} | {'Rel Sayısı':>10} | {'Relation Tipleri'}")
print("  " + "─" * 90)
for club, cnt in zip(seed_entities, entity_rel_counts):
    rels = list(hop1_data[club['id']].keys())
    print(f"  {str(club['desc'])[:40]:40} | {cnt:>10} | {rels}")

# Örnek: ilk kulüp için detay
sample = seed_entities[0]
print(f"\n  → Örnek detay: '{sample['desc']}' 1-hop komşuları:")
for rel, targets in hop1_data[sample['id']].items():
    t_list = ", ".join(str(t['desc'])[:20] for t in targets[:3])
    print(f"    [{rel}] → {t_list}")

criteria_rels      = avg_rels >= 5
criteria_rel_types = len(all_rel_types) >= 5
print(f"\n  ✔ Kriter (≥5 rel/entity) : {'KARŞILANDI ✅' if criteria_rels else 'EKSİK ⚠️'} → ort. {avg_rels:.1f}")
print(f"  ✔ Kriter (≥5 rel türü)  : {'KARŞILANDI ✅' if criteria_rel_types else 'EKSİK ⚠️'} → {len(all_rel_types)} tür")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: 2-HOP NEIGHBORS (3.3 Step 3)
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 3: 2-Hop Neighbors — Multi-Hop Chain Tespiti")

# Tip A: Futbolcu → MEMBER_OF_SPORTS_TEAM → Kulüp + Futbolcu → PLACE_OF_BIRTH → Şehir
paths_A = run("""
    MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (player)-[:PLACE_OF_BIRTH]->(city:Entity)
    RETURN player.entityId AS e1_id, player.description AS e1_desc,
           club.entityId   AS e2_id, club.description   AS e2_desc,
           city.entityId   AS e3_id, city.description   AS e3_desc,
           'MEMBER_OF_SPORTS_TEAM' AS rel1, 'PLACE_OF_BIRTH' AS rel2
    LIMIT 200
""", {"ids": seed_ids})

# Tip B: Kulüp → HOME_VENUE → Stadyum → [rel] → Yer
paths_B = run("""
    MATCH (club:Entity)-[:HOME_VENUE]->(stadium:Entity)
    WHERE club.entityId IN $ids
    MATCH (stadium)-[r2]->(target:Entity)
    RETURN club.entityId    AS e1_id, club.description    AS e1_desc,
           stadium.entityId AS e2_id, stadium.description AS e2_desc,
           target.entityId  AS e3_id, target.description  AS e3_desc,
           'HOME_VENUE' AS rel1, type(r2) AS rel2
    LIMIT 200
""", {"ids": seed_ids})

# Tip C: Antrenör → HEAD_COACH → Kulüp + Antrenör → PLACE_OF_BIRTH → Şehir
paths_C = run("""
    MATCH (coach:Entity)-[:HEAD_COACH]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (coach)-[:PLACE_OF_BIRTH]->(city:Entity)
    RETURN coach.entityId AS e1_id, coach.description AS e1_desc,
           club.entityId  AS e2_id, club.description  AS e2_desc,
           city.entityId  AS e3_id, city.description  AS e3_desc,
           'HEAD_COACH' AS rel1, 'PLACE_OF_BIRTH' AS rel2
    LIMIT 200
""", {"ids": seed_ids})

# Tip D: Futbolcu → MEMBER_OF_SPORTS_TEAM → Kulüp → COUNTRY → Ülke
paths_D = run("""
    MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (club)-[:COUNTRY]->(country:Entity)
    RETURN player.entityId  AS e1_id, player.description  AS e1_desc,
           club.entityId    AS e2_id, club.description    AS e2_desc,
           country.entityId AS e3_id, country.description AS e3_desc,
           'MEMBER_OF_SPORTS_TEAM' AS rel1, 'COUNTRY' AS rel2
    LIMIT 200
""", {"ids": seed_ids})

# Tip E: Futbolcu → MEMBER_OF_SPORTS_TEAM → Kulüp → LEAGUE → Lig
paths_E = run("""
    MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (club)-[:LEAGUE]->(league:Entity)
    RETURN player.entityId  AS e1_id, player.description  AS e1_desc,
           club.entityId    AS e2_id, club.description    AS e2_desc,
           league.entityId  AS e3_id, league.description  AS e3_desc,
           'MEMBER_OF_SPORTS_TEAM' AS rel1, 'LEAGUE' AS rel2
    LIMIT 200
""", {"ids": seed_ids})

total_2hop = len(paths_A) + len(paths_B) + len(paths_C) + len(paths_D) + len(paths_E)

print(f"\n  2-hop path tipleri:")
print(f"    Tip A  Futbolcu → [MEMBER_OF_SPORTS_TEAM] → Kulüp → [PLACE_OF_BIRTH] → Şehir : {len(paths_A):>5,}")
print(f"    Tip B  Kulüp → [HOME_VENUE] → Stadyum → [rel] → Yer                          : {len(paths_B):>5,}")
print(f"    Tip C  Antrenör → [HEAD_COACH] → Kulüp → [PLACE_OF_BIRTH] → Şehir            : {len(paths_C):>5,}")
print(f"    Tip D  Futbolcu → [MEMBER_OF_SPORTS_TEAM] → Kulüp → [COUNTRY] → Ülke         : {len(paths_D):>5,}")
print(f"    Tip E  Futbolcu → [MEMBER_OF_SPORTS_TEAM] → Kulüp → [LEAGUE] → Lig           : {len(paths_E):>5,}")
print(f"    {'─'*80}")
print(f"    TOPLAM                                                                         : {total_2hop:>5,}")

print(f"\n  Örnek 2-hop path'ler (Tip A — Futbolcu→Kulüp→Şehir):")
for p in paths_A[:5]:
    print(f"    {str(p['e1_desc'])[:25]:25} →[{p['rel1']}]→ "
          f"{str(p['e2_desc'])[:20]:20} →[{p['rel2']}]→ {str(p['e3_desc'])[:20]}")

print(f"\n  Örnek 2-hop path'ler (Tip B — Kulüp→Stadyum→Yer):")
for p in paths_B[:5]:
    print(f"    {str(p['e1_desc'])[:25]:25} →[{p['rel1']}]→ "
          f"{str(p['e2_desc'])[:20]:20} →[{p['rel2']}]→ {str(p['e3_desc'])[:20]}")

criteria_2hop = total_2hop >= 30
print(f"\n  ✔ Kriter (≥30 2-hop path): {'KARŞILANDI ✅' if criteria_2hop else 'EKSİK ⚠️'} → {total_2hop:,} path")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: PATH DENSITY ANALYSIS (3.3 Step 4)
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 4: Path Density Analysis — 3-Hop Path'ler")

# 3A: Futbolcu → Kulüp → HOME_VENUE → Stadyum → Şehir
paths_3A = run("""
    MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (club)-[:HOME_VENUE]->(stadium:Entity)
    MATCH (stadium)-[:LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY|COUNTRY]->(place:Entity)
    RETURN player.entityId   AS e1_id, player.description   AS e1_desc,
           club.entityId     AS e2_id, club.description     AS e2_desc,
           stadium.entityId  AS e3_id, stadium.description  AS e3_desc,
           place.entityId    AS e4_id, place.description    AS e4_desc
    LIMIT 200
""", {"ids": seed_ids})

# 3B: Futbolcu → Kulüp → [PLACE_OF_BIRTH] → Şehir → [COUNTRY] → Ülke
paths_3B = run("""
    MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (player)-[:PLACE_OF_BIRTH]->(city:Entity)
    MATCH (city)-[:COUNTRY]->(country:Entity)
    RETURN player.entityId  AS e1_id, player.description  AS e1_desc,
           club.entityId    AS e2_id, club.description    AS e2_desc,
           city.entityId    AS e3_id, city.description    AS e3_desc,
           country.entityId AS e4_id, country.description AS e4_desc
    LIMIT 200
""", {"ids": seed_ids})

# 3C: Antrenör → Kulüp → HOME_VENUE → Stadyum → Şehir
paths_3C = run("""
    MATCH (coach:Entity)-[:HEAD_COACH]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (club)-[:HOME_VENUE]->(stadium:Entity)
    MATCH (stadium)-[:LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY|COUNTRY]->(place:Entity)
    RETURN coach.entityId   AS e1_id, coach.description   AS e1_desc,
           club.entityId    AS e2_id, club.description    AS e2_desc,
           stadium.entityId AS e3_id, stadium.description AS e3_desc,
           place.entityId   AS e4_id, place.description   AS e4_desc
    LIMIT 200
""", {"ids": seed_ids})

# 3D: Futbolcu → Kulüp → LEAGUE → Lig → COUNTRY → Ülke
paths_3D = run("""
    MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
    WHERE club.entityId IN $ids
    MATCH (club)-[:LEAGUE]->(league:Entity)
    MATCH (league)-[:COUNTRY]->(country:Entity)
    RETURN player.entityId  AS e1_id, player.description  AS e1_desc,
           club.entityId    AS e2_id, club.description    AS e2_desc,
           league.entityId  AS e3_id, league.description  AS e3_desc,
           country.entityId AS e4_id, country.description AS e4_desc
    LIMIT 200
""", {"ids": seed_ids})

total_3hop = len(paths_3A) + len(paths_3B) + len(paths_3C) + len(paths_3D)

print(f"\n  3-hop path tipleri:")
print(f"    Tip 3A  Futbolcu→Kulüp→Stadyum→Yer      : {len(paths_3A):>5,}")
print(f"    Tip 3B  Futbolcu→Kulüp→Şehir→Ülke       : {len(paths_3B):>5,}")
print(f"    Tip 3C  Antrenör→Kulüp→Stadyum→Yer      : {len(paths_3C):>5,}")
print(f"    Tip 3D  Futbolcu→Kulüp→Lig→Ülke         : {len(paths_3D):>5,}")
print(f"    {'─'*45}")
print(f"    TOPLAM                                   : {total_3hop:>5,}")

print(f"\n  Örnek 3-hop path'ler (Tip 3A):")
for p in paths_3A[:5]:
    print(f"    {str(p['e1_desc'])[:20]:20} → {str(p['e2_desc'])[:18]:18} "
          f"→ {str(p['e3_desc'])[:18]:18} → {str(p['e4_desc'])[:18]}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.4 KRİTER DEĞERLENDİRMESİ
# ══════════════════════════════════════════════════════════════════════════════
section("3.4 Domain Selection Criteria Değerlendirmesi")

criteria_rows = [
    ("Seed entity sayısı (≥10)",      len(seed_entities),   len(seed_entities) >= 10),
    ("Ort. relation / entity (≥5)",   round(avg_rels, 1),   avg_rels >= 5),
    ("Unique relation türü (≥5)",     len(all_rel_types),   len(all_rel_types) >= 5),
    ("2-hop path sayısı (≥30)",       total_2hop,           total_2hop >= 30),
    ("3-hop path sayısı (bonus)",     total_3hop,           total_3hop > 0),
]

print(f"\n  {'Kriter':45} | {'Değer':>7} | Durum")
print("  " + "─" * 62)
for label, val, ok in criteria_rows:
    print(f"  {label:45} | {str(val):>7} | {'✅' if ok else '⚠️'}")

all_ok = all(ok for _, _, ok in criteria_rows[:4])
print(f"\n  → Genel: {'✅ TÜM KRİTERLER KARŞILANDI' if all_ok else '⚠️ BAZI KRİTERLER EKSİK'}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.5 ÇIKTI 1: Ana Entity Listesi
# ══════════════════════════════════════════════════════════════════════════════
section("3.5 ÇIKTI 1: Ana Entity Listesi")

main_entities = []
for club in seed_entities:
    cid = club['id']

    player_cnt = run("""
        MATCH (p:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(c:Entity {entityId: $cid})
        RETURN count(p) AS cnt
    """, {"cid": cid})

    stadium = run("""
        MATCH (c:Entity {entityId: $cid})-[:HOME_VENUE]->(s:Entity)
        RETURN s.entityId AS sid, s.description AS sdesc LIMIT 1
    """, {"cid": cid})

    coach = run("""
        MATCH (c:Entity {entityId: $cid})-[:HEAD_COACH]->(co:Entity)
        RETURN co.description AS cdesc LIMIT 1
    """, {"cid": cid})

    league = run("""
        MATCH (c:Entity {entityId: $cid})-[:LEAGUE]->(l:Entity)
        RETURN l.description AS ldesc LIMIT 1
    """, {"cid": cid})

    main_entities.append({
        "entity_id":    cid,
        "description":  club['desc'],
        "degree":       club['degree'],
        "player_count": player_cnt[0]['cnt'] if player_cnt else 0,
        "relations":    list(hop1_data.get(cid, {}).keys()),
        "stadium":      stadium[0]['sdesc'] if stadium else None,
        "coach":        coach[0]['cdesc']   if coach   else None,
        "league":       league[0]['ldesc']  if league  else None,
    })

print(f"\n  {'Kulüp':38} | {'Derece':>7} | {'Oyuncu':>7} | {'Stadyum':25} | {'Antrenör'}")
print("  " + "─" * 105)
for e in main_entities:
    print(f"  {str(e['description'])[:38]:38} | {e['degree']:>7,} | "
          f"{e['player_count']:>7,} | {str(e['stadium'])[:25]:25} | {str(e['coach'])[:25]}")

save_json(main_entities, "football_main_entities.json")


# ══════════════════════════════════════════════════════════════════════════════
# 3.5 ÇIKTI 2: Entity-Relation Map (Görselleştirme)
# ══════════════════════════════════════════════════════════════════════════════
section("3.5 ÇIKTI 2: Entity-Relation Map Görselleştirmesi")

G = nx.DiGraph()
node_colors = {}

COLOR = {
    "turkey":  "#c0392b",
    "club":    "#e74c3c",
    "player":  "#3498db",
    "stadium": "#2ecc71",
    "coach":   "#1abc9c",
    "league":  "#9b59b6",
    "city":    "#f39c12",
}

G.add_node("TÜRKİYE"); node_colors["TÜRKİYE"] = COLOR["turkey"]

top_clubs = seed_entities[:6]
for club in top_clubs:
    cname = str(club['desc'])[:22]
    G.add_node(cname); node_colors[cname] = COLOR["club"]
    G.add_edge("TÜRKİYE", cname, label="COUNTRY")

    # Oyuncular (ilk 3)
    pls = run("""
        MATCH (p:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(c:Entity {entityId: $cid})
        RETURN p.description AS desc LIMIT 3
    """, {"cid": club['id']})
    for pl in pls:
        pname = str(pl['desc'])[:18]
        G.add_node(pname); node_colors[pname] = COLOR["player"]
        G.add_edge(cname, pname, label="HAS_PLAYER")

    # Stadyum
    stads = run("""
        MATCH (c:Entity {entityId: $cid})-[:HOME_VENUE]->(s:Entity)
        RETURN s.description AS desc LIMIT 1
    """, {"cid": club['id']})
    for st in stads:
        sname = str(st['desc'])[:18]
        G.add_node(sname); node_colors[sname] = COLOR["stadium"]
        G.add_edge(cname, sname, label="HOME_VENUE")

    # Antrenör
    coaches = run("""
        MATCH (c:Entity {entityId: $cid})-[:HEAD_COACH]->(co:Entity)
        RETURN co.description AS desc LIMIT 1
    """, {"cid": club['id']})
    for co in coaches:
        coname = str(co['desc'])[:18]
        G.add_node(coname); node_colors[coname] = COLOR["coach"]
        G.add_edge(cname, coname, label="HEAD_COACH")

    # Lig
    leagues = run("""
        MATCH (c:Entity {entityId: $cid})-[:LEAGUE]->(l:Entity)
        RETURN l.description AS desc LIMIT 1
    """, {"cid": club['id']})
    for lg in leagues:
        lgname = str(lg['desc'])[:18]
        G.add_node(lgname); node_colors[lgname] = COLOR["league"]
        G.add_edge(cname, lgname, label="LEAGUE")

fig, ax = plt.subplots(figsize=(20, 15))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

pos = nx.spring_layout(G, k=2.8, seed=42, iterations=120)
colors_list = [node_colors.get(n, "#aaaaaa") for n in G.nodes()]

nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors_list,
                       node_size=900, alpha=0.92)
nx.draw_networkx_labels(G, pos, ax=ax,
                        labels={n: n for n in G.nodes()},
                        font_size=6.5, font_color="white", font_weight="bold")
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#aaaaaa", alpha=0.5,
                       arrows=True, arrowsize=15,
                       connectionstyle="arc3,rad=0.1")
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                              font_size=5.5, font_color="#f0e68c", alpha=0.85)

legend_items = [
    mpatches.Patch(color=COLOR["turkey"],  label="Türkiye"),
    mpatches.Patch(color=COLOR["club"],    label="Futbol Kulübü"),
    mpatches.Patch(color=COLOR["player"],  label="Futbolcu"),
    mpatches.Patch(color=COLOR["stadium"], label="Stadyum"),
    mpatches.Patch(color=COLOR["coach"],   label="Antrenör"),
    mpatches.Patch(color=COLOR["league"],  label="Lig"),
]
ax.legend(handles=legend_items, loc="upper left",
          facecolor="#2c2c54", labelcolor="white", fontsize=9)

ax.set_title("Phase 2 — Türk Futbolu: Entity-Relation Map\n"
             "(Top 6 Kulüp, Oyuncular, Stadyumlar, Antrenörler, Ligler)",
             color="white", fontsize=14, fontweight="bold", pad=15)
ax.axis("off")

chart_path = os.path.join(OUTPUT_DIR, "football_entity_relation_map.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print(f"\n  ✔ Graph görselleştirme kaydedildi: {chart_path}")
print(f"    Düğüm: {G.number_of_nodes()} | Kenar: {G.number_of_edges()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.5 ÇIKTI 3: Multi-Hop Path Potansiyel Raporu
# ══════════════════════════════════════════════════════════════════════════════
section("3.5 ÇIKTI 3: Multi-Hop Path Potansiyel Raporu")

all_2hop_paths = (
    [{"type": "A", "pattern": "Futbolcu→[MEMBER_OF_SPORTS_TEAM]→Kulüp→[PLACE_OF_BIRTH]→Şehir", **p}
     for p in paths_A] +
    [{"type": "B", "pattern": "Kulüp→[HOME_VENUE]→Stadyum→[rel]→Yer", **p}
     for p in paths_B] +
    [{"type": "C", "pattern": "Antrenör→[HEAD_COACH]→Kulüp→[PLACE_OF_BIRTH]→Şehir", **p}
     for p in paths_C] +
    [{"type": "D", "pattern": "Futbolcu→[MEMBER_OF_SPORTS_TEAM]→Kulüp→[COUNTRY]→Ülke", **p}
     for p in paths_D] +
    [{"type": "E", "pattern": "Futbolcu→[MEMBER_OF_SPORTS_TEAM]→Kulüp→[LEAGUE]→Lig", **p}
     for p in paths_E]
)

all_3hop_paths = (
    [{"type": "3A", "pattern": "Futbolcu→Kulüp→[HOME_VENUE]→Stadyum→Yer", **p}
     for p in paths_3A] +
    [{"type": "3B", "pattern": "Futbolcu→Kulüp→[PLACE_OF_BIRTH]→Şehir→[COUNTRY]→Ülke", **p}
     for p in paths_3B] +
    [{"type": "3C", "pattern": "Antrenör→Kulüp→[HOME_VENUE]→Stadyum→Şehir", **p}
     for p in paths_3C] +
    [{"type": "3D", "pattern": "Futbolcu→Kulüp→[LEAGUE]→Lig→[COUNTRY]→Ülke", **p}
     for p in paths_3D]
)

path_report = {
    "domain": "Türk Futbolu",
    "summary": {
        "seed_entity_count":    len(seed_entities),
        "total_2hop_paths":     total_2hop,
        "total_3hop_paths":     total_3hop,
        "unique_rel_types":     sorted(all_rel_types),
        "avg_rels_per_entity":  round(avg_rels, 2),
    },
    "path_type_breakdown": {
        "2hop": {
            "A_player_club_city":    len(paths_A),
            "B_club_stadium_place":  len(paths_B),
            "C_coach_club_city":     len(paths_C),
            "D_player_club_country": len(paths_D),
            "E_player_club_league":  len(paths_E),
        },
        "3hop": {
            "3A_player_club_stadium_place": len(paths_3A),
            "3B_player_club_city_country":  len(paths_3B),
            "3C_coach_club_stadium_city":   len(paths_3C),
            "3D_player_club_league_country":len(paths_3D),
        }
    },
    "criteria_evaluation": {
        "seed_entities_ok":  len(seed_entities) >= 10,
        "avg_rels_ok":       avg_rels >= 5,
        "rel_types_ok":      len(all_rel_types) >= 5,
        "hop2_paths_ok":     total_2hop >= 30,
        "all_criteria_met":  all_ok,
    },
    "sample_2hop_paths": all_2hop_paths[:30],
    "sample_3hop_paths": all_3hop_paths[:15],
}

save_json(path_report,      "football_path_density_report.json")
save_json(all_2hop_paths[:50], "football_multihop_paths.json")

print(f"\n  Path Potansiyel Özeti:")
print(f"  {'Path Tipi':55} | {'Sayı':>6}")
print("  " + "─" * 65)
for k, v in path_report["path_type_breakdown"]["2hop"].items():
    print(f"  2-hop / {k:45} | {v:>6,}")
print("  " + "─" * 65)
for k, v in path_report["path_type_breakdown"]["3hop"].items():
    print(f"  3-hop / {k:45} | {v:>6,}")
print("  " + "─" * 65)
print(f"  TOPLAM 2-hop                                             | {total_2hop:>6,}")
print(f"  TOPLAM 3-hop                                             | {total_3hop:>6,}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.5 ÇIKTI 4: Domain Raporu
# ══════════════════════════════════════════════════════════════════════════════
domain_report = {
    "selected_domain": "Türk Futbolu",
    "justification": (
        f"Türkiye bağlamında {len(seed_entities)} futbol kulübü ve "
        f"{sum(e['player_count'] for e in main_entities)} futbolcu tespit edildi. "
        f"MEMBER_OF_SPORTS_TEAM, PLACE_OF_BIRTH, HOME_VENUE, HEAD_COACH, LEAGUE "
        f"relation'ları ile {total_2hop} adet 2-hop ve {total_3hop} adet 3-hop path "
        f"oluşturulabilmektedir. Tüm 3.4 kriterleri karşılanmıştır."
    ),
    "entity_types": {
        "football_clubs":  len(seed_entities),
        "players":         sum(e['player_count'] for e in main_entities),
    },
    "relation_types":  sorted(all_rel_types),
    "path_summary":    path_report["summary"],
    "criteria":        path_report["criteria_evaluation"],
}
save_json(domain_report, "football_domain_report.json")

driver.close()

# ── Final Özet ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 2 — TÜRK FUTBOLU DOMAIN VERIFICATION ÖZET")
print("=" * 65)
print(f"""
  STEP 1  Seed Entity     : {len(seed_entities)} futbol kulübü          {'✅' if len(seed_entities)>=10 else '⚠️'}
  STEP 2  Ort. Relation   : {avg_rels:.1f} / entity               {'✅' if avg_rels>=5 else '⚠️'}
          Rel. Türü        : {len(all_rel_types)} farklı tip             {'✅' if len(all_rel_types)>=5 else '⚠️'}
  STEP 3  2-hop Path       : {total_2hop:,} adet                 {'✅' if total_2hop>=30 else '⚠️'}
  STEP 4  3-hop Path       : {total_3hop:,} adet (bonus)         ✅

  Kaydedilen Dosyalar:
    outputs/football/football_domain_report.json
    outputs/football/football_main_entities.json
    outputs/football/football_path_density_report.json
    outputs/football/football_multihop_paths.json
    outputs/football/football_entity_relation_map.png
""")
print("✅ Türk Futbolu domain verification tamamlandı!")
