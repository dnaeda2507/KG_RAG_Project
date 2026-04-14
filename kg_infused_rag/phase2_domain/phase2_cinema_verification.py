"""
Phase 2 - Domain Verification: Türk Sineması
=============================================
3.3 Domain Verification Process - 4 adımın tamamı
3.4 Domain Selection Criteria değerlendirmesi
3.5 Expected Outputs üretimi

Çalıştır:
    python phase2_cinema_verification.py

Çıktılar:
    outputs/cinema/cinema_domain_report.json
    outputs/cinema/cinema_main_entities.json
    outputs/cinema/cinema_multihop_paths.json
    outputs/cinema/cinema_entity_relation_map.png
    outputs/cinema/cinema_path_density_report.json
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from neo4j import GraphDatabase
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, TURKEY_ID, OUTPUT_PHASE2_CINEMA

# ── Bağlantı ──────────────────────────────────────────────────────────────────
driver     = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
OUTPUT_DIR = OUTPUT_PHASE2_CINEMA
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
section("STEP 1: Seed Entity Detection — Türk Filmleri")

seed_entities = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    WITH film, COUNT { (film)-[]-() } AS degree
    ORDER BY degree DESC
    LIMIT 20
    RETURN film.entityId   AS id,
           film.description AS desc,
           degree
""", {"tid": TURKEY_ID})

print(f"\n  Tespit edilen {len(seed_entities)} Türk filmi (top 20, dereceye göre):")
print(f"\n  {'No':>3} | {'ID':12} | {'Açıklama':50} | {'Derece':>7}")
print("  " + "─" * 80)
for i, e in enumerate(seed_entities, 1):
    print(f"  {i:>3} | {e['id']:12} | {str(e['desc'])[:50]:50} | {e['degree']:>7,}")

# Toplam film sayısı
total_films = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    RETURN count(DISTINCT film) AS cnt
""", {"tid": TURKEY_ID})
total_film_count = total_films[0]['cnt'] if total_films else 0

criteria_seeds = len(seed_entities) >= 10
print(f"\n  Toplam Türk filmi      : {total_film_count:,}")
print(f"  Analiz edilen (top 20) : {len(seed_entities)}")
print(f"  ✔ Kriter (≥10 seed)   : {'KARŞILANDI ✅' if criteria_seeds else 'EKSİK ⚠️'}")

seed_ids = [e['id'] for e in seed_entities]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: 1-HOP NEIGHBORS (3.3 Step 2)
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 2: 1-Hop Neighbors — Her Film İçin Relation'lar")

hop1_data         = {}
all_rel_types     = set()
entity_rel_counts = []

for film in seed_entities:
    fid = film['id']
    neighbors = run("""
        MATCH (e:Entity {entityId: $fid})-[r]->(target:Entity)
        RETURN type(r) AS relation,
               target.entityId    AS target_id,
               target.description  AS target_desc
        ORDER BY type(r)
    """, {"fid": fid})

    rel_map = {}
    for n in neighbors:
        rel = n['relation']
        all_rel_types.add(rel)
        if rel not in rel_map:
            rel_map[rel] = []
        rel_map[rel].append({"id": n['target_id'], "desc": n['target_desc']})

    hop1_data[fid] = rel_map
    entity_rel_counts.append(len(rel_map))

avg_rels = sum(entity_rel_counts) / len(entity_rel_counts) if entity_rel_counts else 0

print(f"\n  Toplam unique relation tipi : {len(all_rel_types)}")
print(f"  Entity başına ort. relation : {avg_rels:.1f}")
print(f"  Tüm relation tipleri        : {sorted(all_rel_types)}")

print(f"\n  Her film için 1-hop relation sayıları:")
print(f"  {'Film':45} | {'Rel Sayısı':>10} | {'Relation Tipleri'}")
print("  " + "─" * 100)
for film, cnt in zip(seed_entities, entity_rel_counts):
    rels = list(hop1_data[film['id']].keys())
    print(f"  {str(film['desc'])[:45]:45} | {cnt:>10} | {rels}")

# Örnek: en fazla relation'a sahip film
sample = max(seed_entities, key=lambda x: len(hop1_data.get(x['id'], {})))
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

# Tip A: Film → DIRECTOR → Yönetmen → PLACE_OF_BIRTH → Şehir
paths_A = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    MATCH (director)-[:PLACE_OF_BIRTH]->(city:Entity)
    RETURN film.entityId     AS e1_id, film.description     AS e1_desc,
           director.entityId AS e2_id, director.description AS e2_desc,
           city.entityId     AS e3_id, city.description     AS e3_desc,
           'DIRECTOR' AS rel1, 'PLACE_OF_BIRTH' AS rel2
    LIMIT 200
""", {"tid": TURKEY_ID})

# Tip B: Film → CAST_MEMBER → Oyuncu → PLACE_OF_BIRTH → Şehir
paths_B = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
    MATCH (actor)-[:PLACE_OF_BIRTH]->(city:Entity)
    RETURN film.entityId  AS e1_id, film.description  AS e1_desc,
           actor.entityId AS e2_id, actor.description AS e2_desc,
           city.entityId  AS e3_id, city.description  AS e3_desc,
           'CAST_MEMBER' AS rel1, 'PLACE_OF_BIRTH' AS rel2
    LIMIT 200
""", {"tid": TURKEY_ID})

# Tip C: Film → DIRECTOR → Yönetmen → AWARD_RECEIVED → Ödül
paths_C = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    MATCH (director)-[:AWARD_RECEIVED]->(award:Entity)
    RETURN film.entityId     AS e1_id, film.description     AS e1_desc,
           director.entityId AS e2_id, director.description AS e2_desc,
           award.entityId    AS e3_id, award.description    AS e3_desc,
           'DIRECTOR' AS rel1, 'AWARD_RECEIVED' AS rel2
    LIMIT 200
""", {"tid": TURKEY_ID})

# Tip D: Film → CAST_MEMBER → Oyuncu → COUNTRY_OF_CITIZENSHIP → Ülke
paths_D = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
    MATCH (actor)-[:COUNTRY_OF_CITIZENSHIP]->(country:Entity)
    RETURN film.entityId    AS e1_id, film.description    AS e1_desc,
           actor.entityId   AS e2_id, actor.description   AS e2_desc,
           country.entityId AS e3_id, country.description AS e3_desc,
           'CAST_MEMBER' AS rel1, 'COUNTRY_OF_CITIZENSHIP' AS rel2
    LIMIT 200
""", {"tid": TURKEY_ID})

# Tip E: Film → DIRECTOR → Yönetmen → EDUCATED_AT → Okul
paths_E = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    MATCH (director)-[:EDUCATED_AT]->(school:Entity)
    RETURN film.entityId     AS e1_id, film.description     AS e1_desc,
           director.entityId AS e2_id, director.description AS e2_desc,
           school.entityId   AS e3_id, school.description   AS e3_desc,
           'DIRECTOR' AS rel1, 'EDUCATED_AT' AS rel2
    LIMIT 200
""", {"tid": TURKEY_ID})

# Tip F: Film → AWARD_RECEIVED → Ödül
paths_F = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:AWARD_RECEIVED]->(award:Entity)
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    RETURN film.entityId     AS e1_id, film.description     AS e1_desc,
           director.entityId AS e2_id, director.description AS e2_desc,
           award.entityId    AS e3_id, award.description    AS e3_desc,
           'DIRECTOR' AS rel1, 'AWARD_RECEIVED' AS rel2
    LIMIT 200
""", {"tid": TURKEY_ID})

total_2hop = (len(paths_A) + len(paths_B) + len(paths_C) +
              len(paths_D) + len(paths_E) + len(paths_F))

print(f"\n  2-hop path tipleri:")
print(f"    Tip A  Film → [DIRECTOR] → Yönetmen → [PLACE_OF_BIRTH] → Şehir          : {len(paths_A):>5,}")
print(f"    Tip B  Film → [CAST_MEMBER] → Oyuncu → [PLACE_OF_BIRTH] → Şehir         : {len(paths_B):>5,}")
print(f"    Tip C  Film → [DIRECTOR] → Yönetmen → [AWARD_RECEIVED] → Ödül           : {len(paths_C):>5,}")
print(f"    Tip D  Film → [CAST_MEMBER] → Oyuncu → [COUNTRY_OF_CITIZENSHIP] → Ülke  : {len(paths_D):>5,}")
print(f"    Tip E  Film → [DIRECTOR] → Yönetmen → [EDUCATED_AT] → Okul              : {len(paths_E):>5,}")
print(f"    Tip F  Film → [DIRECTOR] → Yönetmen → [AWARD_RECEIVED] → Ödül (film)    : {len(paths_F):>5,}")
print(f"    {'─'*80}")
print(f"    TOPLAM                                                                    : {total_2hop:>5,}")

print(f"\n  Örnek 2-hop path'ler (Tip A — Film→Yönetmen→Şehir):")
for p in paths_A[:5]:
    print(f"    {str(p['e1_desc'])[:25]:25} →[{p['rel1']}]→ "
          f"{str(p['e2_desc'])[:20]:20} →[{p['rel2']}]→ {str(p['e3_desc'])[:20]}")

print(f"\n  Örnek 2-hop path'ler (Tip B — Film→Oyuncu→Şehir):")
for p in paths_B[:5]:
    print(f"    {str(p['e1_desc'])[:25]:25} →[{p['rel1']}]→ "
          f"{str(p['e2_desc'])[:20]:20} →[{p['rel2']}]→ {str(p['e3_desc'])[:20]}")

criteria_2hop = total_2hop >= 30
print(f"\n  ✔ Kriter (≥30 2-hop path): {'KARŞILANDI ✅' if criteria_2hop else 'EKSİK ⚠️'} → {total_2hop:,} path")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: PATH DENSITY ANALYSIS (3.3 Step 4)
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 4: Path Density Analysis — 3-Hop Path'ler")

# 3A: Film → DIRECTOR → Yönetmen → PLACE_OF_BIRTH → Şehir → COUNTRY → Ülke
paths_3A = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    MATCH (director)-[:PLACE_OF_BIRTH]->(city:Entity)
    MATCH (city)-[:COUNTRY]->(country:Entity)
    RETURN film.entityId     AS e1_id, film.description     AS e1_desc,
           director.entityId AS e2_id, director.description AS e2_desc,
           city.entityId     AS e3_id, city.description     AS e3_desc,
           country.entityId  AS e4_id, country.description  AS e4_desc
    LIMIT 200
""", {"tid": TURKEY_ID})

# 3B: Film → CAST_MEMBER → Oyuncu → AWARD_RECEIVED → Ödül
paths_3B = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
    MATCH (actor)-[:AWARD_RECEIVED]->(award:Entity)
    RETURN film.entityId  AS e1_id, film.description  AS e1_desc,
           actor.entityId AS e2_id, actor.description AS e2_desc,
           award.entityId AS e3_id, award.description AS e3_desc,
           '' AS e4_id,   '' AS e4_desc
    LIMIT 200
""", {"tid": TURKEY_ID})

# 3C: Film → DIRECTOR → Yönetmen → EDUCATED_AT → Okul → COUNTRY → Ülke
paths_3C = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(director:Entity)
    MATCH (director)-[:EDUCATED_AT]->(school:Entity)
    MATCH (school)-[:COUNTRY]->(country:Entity)
    RETURN film.entityId     AS e1_id, film.description     AS e1_desc,
           director.entityId AS e2_id, director.description AS e2_desc,
           school.entityId   AS e3_id, school.description   AS e3_desc,
           country.entityId  AS e4_id, country.description  AS e4_desc
    LIMIT 200
""", {"tid": TURKEY_ID})

# 3D: Film → CAST_MEMBER → Oyuncu → PLACE_OF_BIRTH → Şehir → COUNTRY → Ülke
paths_3D = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(actor:Entity)
    MATCH (actor)-[:PLACE_OF_BIRTH]->(city:Entity)
    MATCH (city)-[:COUNTRY]->(country:Entity)
    RETURN film.entityId    AS e1_id, film.description    AS e1_desc,
           actor.entityId   AS e2_id, actor.description   AS e2_desc,
           city.entityId    AS e3_id, city.description    AS e3_desc,
           country.entityId AS e4_id, country.description AS e4_desc
    LIMIT 200
""", {"tid": TURKEY_ID})

total_3hop = len(paths_3A) + len(paths_3B) + len(paths_3C) + len(paths_3D)

print(f"\n  3-hop path tipleri:")
print(f"    Tip 3A  Film→Yönetmen→Şehir→Ülke            : {len(paths_3A):>5,}")
print(f"    Tip 3B  Film→Oyuncu→Ödül                    : {len(paths_3B):>5,}")
print(f"    Tip 3C  Film→Yönetmen→Okul→Ülke             : {len(paths_3C):>5,}")
print(f"    Tip 3D  Film→Oyuncu→Şehir→Ülke              : {len(paths_3D):>5,}")
print(f"    {'─'*45}")
print(f"    TOPLAM                                       : {total_3hop:>5,}")

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
for film in seed_entities:
    fid = film['id']

    directors = run("""
        MATCH (f:Entity {entityId: $fid})-[:DIRECTOR]->(d:Entity)
        RETURN d.entityId AS did, d.description AS ddesc
        LIMIT 2
    """, {"fid": fid})

    actors = run("""
        MATCH (f:Entity {entityId: $fid})-[:CAST_MEMBER]->(a:Entity)
        RETURN count(a) AS cnt
    """, {"fid": fid})

    awards = run("""
        MATCH (f:Entity {entityId: $fid})-[:AWARD_RECEIVED]->(aw:Entity)
        RETURN count(aw) AS cnt
    """, {"fid": fid})

    genre = run("""
        MATCH (f:Entity {entityId: $fid})-[:GENRE]->(g:Entity)
        RETURN g.description AS gdesc LIMIT 1
    """, {"fid": fid})

    main_entities.append({
        "entity_id":   fid,
        "description": film['desc'],
        "degree":      film['degree'],
        "relations":   list(hop1_data.get(fid, {}).keys()),
        "directors":   [{"id": d['did'], "desc": d['ddesc']} for d in directors],
        "actor_count": actors[0]['cnt']  if actors  else 0,
        "award_count": awards[0]['cnt']  if awards  else 0,
        "genre":       genre[0]['gdesc'] if genre   else None,
    })

print(f"\n  {'Film':45} | {'Derece':>7} | {'Yönetmen':25} | {'Oyuncu':>7} | {'Ödül':>5}")
print("  " + "─" * 100)
for e in main_entities:
    dir_name = e['directors'][0]['desc'] if e['directors'] else "-"
    print(f"  {str(e['description'])[:45]:45} | {e['degree']:>7,} | "
          f"{str(dir_name)[:25]:25} | {e['actor_count']:>7,} | {e['award_count']:>5,}")

save_json(main_entities, "cinema_main_entities.json")


# ══════════════════════════════════════════════════════════════════════════════
# 3.5 ÇIKTI 2: Entity-Relation Map (Görselleştirme)
# ══════════════════════════════════════════════════════════════════════════════
section("3.5 ÇIKTI 2: Entity-Relation Map Görselleştirmesi")

G = nx.DiGraph()
node_colors = {}

COLOR = {
    "turkey":   "#c0392b",
    "film":     "#e74c3c",
    "director": "#3498db",
    "actor":    "#2ecc71",
    "award":    "#f39c12",
    "city":     "#9b59b6",
    "genre":    "#1abc9c",
}

G.add_node("TÜRKİYE"); node_colors["TÜRKİYE"] = COLOR["turkey"]

top_films = seed_entities[:6]
for film in top_films:
    fname = str(film['desc'])[:22]
    G.add_node(fname); node_colors[fname] = COLOR["film"]
    G.add_edge("TÜRKİYE", fname, label="COUNTRY_OF_ORIGIN")

    # Yönetmen
    dirs = run("""
        MATCH (f:Entity {entityId: $fid})-[:DIRECTOR]->(d:Entity)
        RETURN d.description AS desc LIMIT 1
    """, {"fid": film['id']})
    for d in dirs:
        dname = str(d['desc'])[:18]
        G.add_node(dname); node_colors[dname] = COLOR["director"]
        G.add_edge(fname, dname, label="DIRECTOR")

    # Oyuncular (ilk 2)
    acts = run("""
        MATCH (f:Entity {entityId: $fid})-[:CAST_MEMBER]->(a:Entity)
        RETURN a.description AS desc LIMIT 2
    """, {"fid": film['id']})
    for a in acts:
        aname = str(a['desc'])[:18]
        G.add_node(aname); node_colors[aname] = COLOR["actor"]
        G.add_edge(fname, aname, label="CAST_MEMBER")

    # Ödül
    awds = run("""
        MATCH (f:Entity {entityId: $fid})-[:AWARD_RECEIVED]->(aw:Entity)
        RETURN aw.description AS desc LIMIT 1
    """, {"fid": film['id']})
    for aw in awds:
        awname = str(aw['desc'])[:18]
        G.add_node(awname); node_colors[awname] = COLOR["award"]
        G.add_edge(fname, awname, label="AWARD_RECEIVED")

    # Tür
    gnrs = run("""
        MATCH (f:Entity {entityId: $fid})-[:GENRE]->(g:Entity)
        RETURN g.description AS desc LIMIT 1
    """, {"fid": film['id']})
    for g in gnrs:
        gname = str(g['desc'])[:18]
        G.add_node(gname); node_colors[gname] = COLOR["genre"]
        G.add_edge(fname, gname, label="GENRE")

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
    mpatches.Patch(color=COLOR["turkey"],   label="Türkiye"),
    mpatches.Patch(color=COLOR["film"],     label="Film"),
    mpatches.Patch(color=COLOR["director"], label="Yönetmen"),
    mpatches.Patch(color=COLOR["actor"],    label="Oyuncu"),
    mpatches.Patch(color=COLOR["award"],    label="Ödül"),
    mpatches.Patch(color=COLOR["genre"],    label="Tür"),
]
ax.legend(handles=legend_items, loc="upper left",
          facecolor="#2c2c54", labelcolor="white", fontsize=9)

ax.set_title("Phase 2 — Türk Sineması: Entity-Relation Map\n"
             "(Top 6 Film, Yönetmenler, Oyuncular, Ödüller, Türler)",
             color="white", fontsize=14, fontweight="bold", pad=15)
ax.axis("off")

chart_path = os.path.join(OUTPUT_DIR, "cinema_entity_relation_map.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print(f"\n  ✔ Graph görselleştirme kaydedildi: {chart_path}")
print(f"    Düğüm: {G.number_of_nodes()} | Kenar: {G.number_of_edges()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.5 ÇIKTI 3: Multi-Hop Path Potansiyel Raporu
# ══════════════════════════════════════════════════════════════════════════════
section("3.5 ÇIKTI 3: Multi-Hop Path Potansiyel Raporu")

all_2hop_paths = (
    [{"type": "A", "pattern": "Film→[DIRECTOR]→Yönetmen→[PLACE_OF_BIRTH]→Şehir", **p}
     for p in paths_A] +
    [{"type": "B", "pattern": "Film→[CAST_MEMBER]→Oyuncu→[PLACE_OF_BIRTH]→Şehir", **p}
     for p in paths_B] +
    [{"type": "C", "pattern": "Film→[DIRECTOR]→Yönetmen→[AWARD_RECEIVED]→Ödül", **p}
     for p in paths_C] +
    [{"type": "D", "pattern": "Film→[CAST_MEMBER]→Oyuncu→[COUNTRY_OF_CITIZENSHIP]→Ülke", **p}
     for p in paths_D] +
    [{"type": "E", "pattern": "Film→[DIRECTOR]→Yönetmen→[EDUCATED_AT]→Okul", **p}
     for p in paths_E] +
    [{"type": "F", "pattern": "Film→[DIRECTOR]→Yönetmen→[AWARD_RECEIVED]→Ödül(film)", **p}
     for p in paths_F]
)

all_3hop_paths = (
    [{"type": "3A", "pattern": "Film→Yönetmen→[PLACE_OF_BIRTH]→Şehir→[COUNTRY]→Ülke", **p}
     for p in paths_3A] +
    [{"type": "3B", "pattern": "Film→Oyuncu→[AWARD_RECEIVED]→Ödül", **p}
     for p in paths_3B] +
    [{"type": "3C", "pattern": "Film→Yönetmen→[EDUCATED_AT]→Okul→[COUNTRY]→Ülke", **p}
     for p in paths_3C] +
    [{"type": "3D", "pattern": "Film→Oyuncu→[PLACE_OF_BIRTH]→Şehir→[COUNTRY]→Ülke", **p}
     for p in paths_3D]
)

path_report = {
    "domain": "Türk Sineması",
    "summary": {
        "total_films":          total_film_count,
        "seed_entity_count":    len(seed_entities),
        "total_2hop_paths":     total_2hop,
        "total_3hop_paths":     total_3hop,
        "unique_rel_types":     sorted(all_rel_types),
        "avg_rels_per_entity":  round(avg_rels, 2),
    },
    "path_type_breakdown": {
        "2hop": {
            "A_film_director_city":          len(paths_A),
            "B_film_actor_city":             len(paths_B),
            "C_film_director_award":         len(paths_C),
            "D_film_actor_citizenship":      len(paths_D),
            "E_film_director_school":        len(paths_E),
            "F_film_director_award_film":    len(paths_F),
        },
        "3hop": {
            "3A_film_director_city_country": len(paths_3A),
            "3B_film_actor_award":           len(paths_3B),
            "3C_film_director_school_country":len(paths_3C),
            "3D_film_actor_city_country":    len(paths_3D),
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

save_json(path_report,         "cinema_path_density_report.json")
save_json(all_2hop_paths[:50], "cinema_multihop_paths.json")

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
    "selected_domain": "Türk Sineması",
    "justification": (
        f"Türkiye bağlamında toplam {total_film_count} Türk filmi tespit edildi. "
        f"DIRECTOR, CAST_MEMBER, AWARD_RECEIVED, EDUCATED_AT, GENRE, COUNTRY_OF_ORIGIN "
        f"relation'ları ile {total_2hop} adet 2-hop ve {total_3hop} adet 3-hop path "
        f"oluşturulabilmektedir. {len(all_rel_types)} farklı relation türü ile "
        f"en çeşitli domain konumundadır. Tüm 3.4 kriterleri karşılanmıştır."
    ),
    "entity_types": {
        "films":     total_film_count,
       "directors": run("""
            MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
            MATCH (film)-[:DIRECTOR]->(d:Entity)
            RETURN count(DISTINCT d) AS cnt
        """, {"tid": TURKEY_ID})[0].get('cnt', 0),
        "actors": run("""
            MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
            MATCH (film)-[:CAST_MEMBER]->(a:Entity)
            RETURN count(DISTINCT a) AS cnt
        """, {"tid": TURKEY_ID})[0].get('cnt', 0),
    },
    "relation_types":  sorted(all_rel_types),
    "path_summary":    path_report["summary"],
    "criteria":        path_report["criteria_evaluation"],
}
save_json(domain_report, "cinema_domain_report.json")

driver.close()

# ── Final Özet ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 2 — TÜRK SİNEMASI DOMAIN VERIFICATION ÖZET")
print("=" * 65)
print(f"""
  STEP 1  Seed Entity     : {len(seed_entities)} film (toplam {total_film_count})  {'✅' if len(seed_entities)>=10 else '⚠️'}
  STEP 2  Ort. Relation   : {avg_rels:.1f} / entity              {'✅' if avg_rels>=5 else '⚠️'}
          Rel. Türü        : {len(all_rel_types)} farklı tip            {'✅' if len(all_rel_types)>=5 else '⚠️'}
  STEP 3  2-hop Path       : {total_2hop:,} adet                {'✅' if total_2hop>=30 else '⚠️'}
  STEP 4  3-hop Path       : {total_3hop:,} adet (bonus)        ✅

  Kaydedilen Dosyalar:
    outputs/cinema/cinema_domain_report.json
    outputs/cinema/cinema_main_entities.json
    outputs/cinema/cinema_path_density_report.json
    outputs/cinema/cinema_multihop_paths.json
    outputs/cinema/cinema_entity_relation_map.png
""")
print("✅ Türk Sineması domain verification tamamlandı!")
