"""
Phase 1 - Türkiye Entity Analizi
=================================
Çalıştır:
    python phase1_turkey_analysis.py

Çıktılar:
    outputs/phase1_turkey_stats.json
    outputs/phase1_city_report.json
    outputs/phase1_relation_freq.json
    outputs/phase1_domain_counts.json
    outputs/phase1_domain_chart.png

Düzeltme:
    driver.close() en sona taşındı — grafik ve JSON kayıt işlemleri
    tamamlandıktan SONRA çağrılıyor.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ── Bağlantı ──────────────────────────────────────────────────────────────────
load_dotenv()
URI      = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run(cypher, params=None):
    with driver.session() as s:
        return [r.data() for r in s.run(cypher, params or {})]


def save_json(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Kaydedildi: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.3.1 — Türkiye Ana Entity Tespiti
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2.3.1 Türkiye Ana Entity Tespiti")
print("=" * 60)

turkey_candidates = run("""
    MATCH (e:Entity)
    WHERE toLower(e.description) IN ['turkey', 'türkiye', 'turkiye']
       OR toLower(e.description) CONTAINS 'republic of turkey'
       OR toLower(e.description) CONTAINS 'republic of türkiye'
    RETURN e.entityId AS id, e.description AS desc
    ORDER BY size(e.entityId) ASC
    LIMIT 10
""")

print("\n  Bulunan Türkiye adayları:")
for c in turkey_candidates:
    print(f"    {c['id']:10} | {c['desc']}")

TURKEY_ID = turkey_candidates[0]['id']
print(f"\n  → Seçilen Ana Türkiye Entity ID: {TURKEY_ID}")

sample_triples = run("""
    MATCH (e:Entity)-[r]->(t:Entity {entityId: $tid})
    RETURN e.entityId AS subj, type(r) AS rel, t.entityId AS obj,
           e.description AS subj_desc
    LIMIT 10
    UNION
    MATCH (t:Entity {entityId: $tid})-[r]->(e:Entity)
    RETURN t.entityId AS subj, type(r) AS rel, e.entityId AS obj,
           e.description AS subj_desc
    LIMIT 10
""", {"tid": TURKEY_ID})

print(f"\n  Türkiye'ye bağlı örnek triple'lar:")
for t in sample_triples[:15]:
    print(f"    {t['subj']:10} --[{t['rel']}]--> {t['obj']:10}  ({str(t['subj_desc'])[:40]})")

total_incoming = run("""
    MATCH (e:Entity)-[r]->(t:Entity {entityId: $tid})
    RETURN count(DISTINCT e) AS count
""", {"tid": TURKEY_ID})

total_outgoing = run("""
    MATCH (t:Entity {entityId: $tid})-[r]->(e:Entity)
    RETURN count(DISTINCT e) AS count
""", {"tid": TURKEY_ID})

incoming_rels = run("""
    MATCH ()-[r]->(t:Entity {entityId: $tid})
    RETURN type(r) AS rel, count(*) AS cnt
    ORDER BY cnt DESC LIMIT 20
""", {"tid": TURKEY_ID})

outgoing_rels = run("""
    MATCH (t:Entity {entityId: $tid})-[r]->()
    RETURN type(r) AS rel, count(*) AS cnt
    ORDER BY cnt DESC
""", {"tid": TURKEY_ID})

print(f"\n  Türkiye'ye gelen unique entity : {total_incoming[0]['count']:,}")
print(f"  Türkiye'den çıkan unique entity: {total_outgoing[0]['count']:,}")

print("\n  Türkiye'ye GELEN relation tipleri (top 20):")
for r in incoming_rels:
    print(f"    {r['rel']:45} : {r['cnt']:,}")

print("\n  Türkiye'den ÇIKAN relation tipleri:")
for r in outgoing_rels:
    print(f"    {r['rel']:45} : {r['cnt']:,}")

stats = {
    "turkey_entity_id":          TURKEY_ID,
    "detection_method":          "Neo4j description eşleşmesi ile otomatik tespit",
    "turkey_candidates":         turkey_candidates,
    "sample_triples":            sample_triples,
    "total_incoming_entities":   total_incoming[0]['count'],
    "total_outgoing_entities":   total_outgoing[0]['count'],
    "incoming_relations_top20":  incoming_rels,
    "outgoing_relations":        outgoing_rels,
}
save_json(stats, "phase1_turkey_stats.json")


# ══════════════════════════════════════════════════════════════════════════════
# 2.3.2 — Türk Şehirleri
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2.3.2 Türk Şehirleri ve Entity Yoğunluğu")
print("=" * 60)

cities = run("""
    MATCH (city:Entity)-[:COUNTRY]->(t:Entity {entityId: $tid})
    WHERE toLower(city.description) CONTAINS 'city'
       OR toLower(city.description) CONTAINS 'province'
       OR toLower(city.description) CONTAINS 'district'
       OR toLower(city.description) CONTAINS 'turkey'
       OR toLower(city.description) CONTAINS 'türkiye'
    WITH city, COUNT { (city)-[]-() } AS total_connections
    ORDER BY total_connections DESC
    LIMIT 20
    RETURN city.entityId  AS id,
           city.description AS desc,
           total_connections
""", {"tid": TURKEY_ID})

print(f"\n  {'Şehir Açıklaması':35} {'ID':12} {'Bağlantı Sayısı'}")
print("  " + "-" * 65)
for r in cities:
    print(f"  {str(r['desc']):35} {r['id']:12} {r['total_connections']:,}")

city_data = []
for r in cities:
    born = run("""
        MATCH (e:Entity)-[:PLACE_OF_BIRTH]->(city:Entity {entityId: $cid})
        RETURN count(e) AS count
    """, {"cid": r['id']})

    city_data.append({
        "city_id":           r['id'],
        "description":       r['desc'],
        "total_connections": r['total_connections'],
        "people_born_here":  born[0]['count'] if born else 0,
    })

print(f"\n  {'Şehir':35} {'Doğan Kişi':12}")
print("  " + "-" * 50)
for d in city_data:
    print(f"  {str(d['description']):35} {d['people_born_here']:,}")

save_json(city_data, "phase1_city_report.json")


# ══════════════════════════════════════════════════════════════════════════════
# 2.3.3 — Relation Frekansları
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2.3.3 Relation Tip Frekansları")
print("=" * 60)

rel_freq = run("""
    MATCH (e:Entity)-[r]->(t:Entity {entityId: $tid})
    RETURN type(r) AS relation, count(*) AS frequency
    ORDER BY frequency DESC LIMIT 30
""", {"tid": TURKEY_ID})

print(f"\n  {'Relation':45} | Frekans")
print("  " + "-" * 58)
for r in rel_freq:
    print(f"  {r['relation']:45} | {r['frequency']:,}")

target_relations = [
    "COUNTRY", "PLACE_OF_BIRTH", "HEADQUARTERS_LOCATION",
    "MEMBER_OF_SPORTS_TEAM", "DIRECTOR", "EDUCATED_AT",
    "COUNTRY_OF_CITIZENSHIP", "COUNTRY_OF_ORIGIN",
]

print("\n  Ödevde istenen relation'ların Türkiye bağlamındaki kullanımı:")
specific_rel = {}
for rel in target_relations:
    result = run(f"""
        MATCH (e:Entity)-[:{rel}]->(t:Entity {{entityId: $tid}})
        RETURN count(e) AS cnt
    """, {"tid": TURKEY_ID})
    cnt = result[0]['cnt'] if result else 0
    specific_rel[rel] = cnt
    bar = "█" * min(cnt // 500, 30)
    print(f"    {rel:45} : {cnt:>6,}  {bar}")

save_json({
    "top_relations":    rel_freq,
    "target_relations": specific_rel
}, "phase1_relation_freq.json")


# ══════════════════════════════════════════════════════════════════════════════
# 2.4 — Domain Bazlı Veri Yoğunluğu
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2.4 Domain Bazlı Veri Yoğunluğu")
print("=" * 60)

domain_queries = {
    "Spor - Futbolcular": """
        MATCH (e:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
        WHERE (club)-[:COUNTRY]->(:Entity {entityId: $tid})
        RETURN count(DISTINCT e) AS cnt
    """,
    "Spor - Kulüpler": """
        MATCH (club:Entity)-[:COUNTRY]->(:Entity {entityId: $tid})
        WHERE toLower(club.description) CONTAINS 'football'
           OR toLower(club.description) CONTAINS 'sport'
        RETURN count(DISTINCT club) AS cnt
    """,
    "Sinema - Filmler": """
        MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
        RETURN count(DISTINCT film) AS cnt
    """,
    "Sinema - Yönetmenler": """
        MATCH (e:Entity)-[:COUNTRY_OF_CITIZENSHIP]->(:Entity {entityId: $tid})
        WHERE toLower(e.description) CONTAINS 'director'
           OR toLower(e.description) CONTAINS 'filmmaker'
        RETURN count(DISTINCT e) AS cnt
    """,
    "Şirketler": """
        MATCH (co:Entity)-[:HEADQUARTERS_LOCATION]->(city:Entity)
        WHERE (city)-[:COUNTRY]->(:Entity {entityId: $tid})
        RETURN count(DISTINCT co) AS cnt
    """,
    "Eğitim - Üniversiteler": """
        MATCH (u:Entity)-[:COUNTRY]->(:Entity {entityId: $tid})
        WHERE toLower(u.description) CONTAINS 'university'
           OR toLower(u.description) CONTAINS 'college'
        RETURN count(DISTINCT u) AS cnt
    """,
    "Müzik - Sanatçılar": """
        MATCH (e:Entity)-[:COUNTRY_OF_CITIZENSHIP]->(:Entity {entityId: $tid})
        WHERE toLower(e.description) CONTAINS 'singer'
           OR toLower(e.description) CONTAINS 'musician'
           OR toLower(e.description) CONTAINS 'composer'
        RETURN count(DISTINCT e) AS cnt
    """,
    "Doğum Yeri Türkiye": """
        MATCH (e:Entity)-[:PLACE_OF_BIRTH]->(city:Entity)
        WHERE (city)-[:COUNTRY]->(:Entity {entityId: $tid})
        RETURN count(DISTINCT e) AS cnt
    """,
}

domain_counts = {}
print(f"\n  {'Domain':30} | Sayı")
print("  " + "-" * 42)
for domain, query in domain_queries.items():
    result = run(query, {"tid": TURKEY_ID})
    cnt = result[0]['cnt'] if result else 0
    domain_counts[domain] = cnt
    bar = "█" * min(cnt // 100, 25)
    print(f"  {domain:30} | {cnt:>6,}  {bar}")

save_json(domain_counts, "phase1_domain_counts.json")


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİKLER
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("GRAFİKLER oluşturuluyor...")
print("=" * 60)

colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71',
          '#3498db','#9b59b6','#1abc9c','#e91e63']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Phase 1: Wikidata5M Türkiye Domain Analizi",
             fontsize=16, fontweight='bold')

ax1 = axes[0, 0]
d_labels = list(domain_counts.keys())
d_values = list(domain_counts.values())
bars = ax1.barh(d_labels, d_values, color=colors[:len(d_labels)])
ax1.set_title("Domain Bazlı Entity Sayısı", fontweight='bold')
ax1.set_xlabel("Entity Sayısı")
mx = max(d_values) if d_values else 1
for bar, val in zip(bars, d_values):
    ax1.text(bar.get_width() + mx * 0.01,
             bar.get_y() + bar.get_height() / 2,
             f'{val:,}', va='center', fontsize=9)
ax1.tick_params(axis='y', labelsize=8)

ax2 = axes[0, 1]
valid = [(d['description'][:20], d['people_born_here'])
         for d in city_data if d['people_born_here'] > 0]
if valid:
    valid.sort(key=lambda x: x[1], reverse=True)
    cn, cv = zip(*valid)
    ax2.bar(cn, cv, color='#3498db', edgecolor='white')
    ax2.set_title("Şehir Bazlı Doğan Kişi Sayısı", fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.set_ylabel("Kişi Sayısı")
else:
    ax2.text(0.5, 0.5, "Veri bulunamadı",
             ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title("Şehir Bazlı Doğan Kişi Sayısı", fontweight='bold')

ax3 = axes[1, 0]
if rel_freq:
    rn = [r['relation'] for r in rel_freq[:12]]
    rv = [r['frequency'] for r in rel_freq[:12]]
    ax3.barh(rn[::-1], rv[::-1], color='#e74c3c')
    ax3.set_title("Türkiye'ye Gelen Top 12 Relation", fontweight='bold')
    ax3.set_xlabel("Frekans")
    ax3.tick_params(axis='y', labelsize=7)

ax4 = axes[1, 1]
nz = {k: v for k, v in domain_counts.items() if v > 0}
if nz:
    ax4.pie(nz.values(), labels=nz.keys(), autopct='%1.1f%%',
            startangle=140, colors=colors[:len(nz)])
    ax4.set_title("Domain Dağılımı (%)", fontweight='bold')
else:
    ax4.text(0.5, 0.5, "Veri bulunamadı",
             ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title("Domain Dağılımı (%)", fontweight='bold')

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "phase1_domain_chart.png")
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Grafik kaydedildi: {chart_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ÖZET
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 1 ÖZET")
print("=" * 60)
print(f"  Türkiye Entity ID              : {TURKEY_ID}")
print(f"  Toplam gelen unique entity     : {total_incoming[0]['count']:,}")
print(f"  Toplam giden unique entity     : {total_outgoing[0]['count']:,}")
print(f"  Bulunan şehir sayısı           : {len(city_data)}")
if domain_counts:
    best = max(domain_counts, key=domain_counts.get)
    print(f"  En yoğun domain               : {best} ({domain_counts[best]:,})")
if rel_freq:
    print(f"  En popüler relation            : {rel_freq[0]['relation']} ({rel_freq[0]['frequency']:,})")

# ✔ driver.close() artık en sonda — tüm işlemler tamamlandıktan sonra
driver.close()
print("\n✅ Phase 1 tamamlandı!")
print(f"   Çıktılar: {OUTPUT_DIR}/ klasöründe")
