"""
Phase 2 - 5 Domain Hızlı Karşılaştırma
========================================
Tüm 5 domain'e bakış yapar, karşılaştırma tablosu üretir.
Çıktıya göre domain seçimi yapılacak.

Çalıştır:
    python phase2_domain_overview.py

Çıktılar:
    outputs/phase2_domain_overview.json
    outputs/phase2_domain_overview_chart.png

Düzeltme:
    phase1_summary artık Phase 1 çıktı JSON'undan okunuyor (hardcode değil).
    Dosya yoksa uyarı verilip devam ediliyor.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from neo4j import GraphDatabase
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, TURKEY_ID, OUTPUT_PHASE1

# ── Bağlantı ──────────────────────────────────────────────────────────────────
driver    = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
OUTPUT_DIR = OUTPUT_PHASE1
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


# ✔ DÜZELTME: phase1_summary Phase 1 çıktı dosyasından okunuyor
def _load_phase1_summary() -> dict:
    """
    Phase 1 çıktı JSON dosyalarından özet değerleri okur.
    Dosyalar yoksa boş dict döner ve uyarı verir.
    """
    summary = {}
    mapping = {
        "Spor_Futbolcular":        ("outputs/phase1_domain_counts.json", "Spor - Futbolcular"),
        "Spor_Kulupler":           ("outputs/phase1_domain_counts.json", "Spor - Kulüpler"),
        "Sinema_Filmler":          ("outputs/phase1_domain_counts.json", "Sinema - Filmler"),
        "Sinema_Yonetmenler":      ("outputs/phase1_domain_counts.json", "Sinema - Yönetmenler"),
        "Sirketler":               ("outputs/phase1_domain_counts.json", "Şirketler"),
        "Egitim_Universiteler":    ("outputs/phase1_domain_counts.json", "Eğitim - Üniversiteler"),
        "Muzik_Sanatcilar":        ("outputs/phase1_domain_counts.json", "Müzik - Sanatçılar"),
        "Dogum_Yeri_Turkiye":      ("outputs/phase1_domain_counts.json", "Doğum Yeri Türkiye"),
    }

    domain_counts_path = "outputs/phase1_domain_counts.json"
    if not os.path.exists(domain_counts_path):
        print(f"  ⚠ {domain_counts_path} bulunamadı — phase1_summary boş bırakılıyor.")
        print("    Phase 1'i önce çalıştırın: python phase1_turkey_analysis.py")
        return {}

    with open(domain_counts_path, "r", encoding="utf-8") as f:
        domain_counts = json.load(f)

    for key, (_, json_key) in mapping.items():
        summary[key] = domain_counts.get(json_key, 0)

    print(f"  ✔ phase1_summary yüklendi: {domain_counts_path}")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 1: TÜRK FUTBOLU
# ══════════════════════════════════════════════════════════════════════════════
section("DOMAIN 1: Türk Futbolu")

football_clubs = run("""
    MATCH (club:Entity)-[:COUNTRY]->(:Entity {entityId: $tid})
    WHERE toLower(club.description) CONTAINS 'football club'
       OR toLower(club.description) CONTAINS 'football team'
    RETURN count(DISTINCT club) AS cnt
""", {"tid": TURKEY_ID})
football_club_count = football_clubs[0]['cnt'] if football_clubs else 0

football_players = run("""
    MATCH (p:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
          -[:COUNTRY]->(:Entity {entityId: $tid})
    WHERE toLower(club.description) CONTAINS 'football club'
       OR toLower(club.description) CONTAINS 'football team'
    RETURN count(DISTINCT p) AS cnt
""", {"tid": TURKEY_ID})
football_player_count = football_players[0]['cnt'] if football_players else 0

football_rel_types = run("""
    MATCH (club:Entity)-[:COUNTRY]->(:Entity {entityId: $tid})
    WHERE toLower(club.description) CONTAINS 'football club'
       OR toLower(club.description) CONTAINS 'football team'
    MATCH (club)-[r]->(target)
    RETURN DISTINCT type(r) AS rel
    LIMIT 50
""", {"tid": TURKEY_ID})
football_rels = [r['rel'] for r in football_rel_types]

football_2hop = run("""
    MATCH (p:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
          -[:COUNTRY]->(:Entity {entityId: $tid})
    WHERE toLower(club.description) CONTAINS 'football club'
       OR toLower(club.description) CONTAINS 'football team'
    MATCH (p)-[:PLACE_OF_BIRTH]->(city:Entity)
    RETURN count(*) AS cnt
""", {"tid": TURKEY_ID})
football_2hop_count = football_2hop[0]['cnt'] if football_2hop else 0

football_3hop = run("""
    MATCH (p:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(club:Entity)
          -[:COUNTRY]->(:Entity {entityId: $tid})
    WHERE toLower(club.description) CONTAINS 'football club'
       OR toLower(club.description) CONTAINS 'football team'
    MATCH (club)-[:HOME_VENUE]->(stadium:Entity)
    MATCH (stadium)-[:LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY|COUNTRY]->(place:Entity)
    RETURN count(*) AS cnt
""", {"tid": TURKEY_ID})
football_3hop_count = football_3hop[0]['cnt'] if football_3hop else 0

print(f"\n  Kulüp sayısı        : {football_club_count:,}")
print(f"  Futbolcu sayısı     : {football_player_count:,}")
print(f"  Relation türleri    : {football_rels}")
print(f"  2-hop path sayısı   : {football_2hop_count:,}")
print(f"  3-hop path sayısı   : {football_3hop_count:,}")


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 2: TÜRK SİNEMASI
# ══════════════════════════════════════════════════════════════════════════════
section("DOMAIN 2: Türk Sineması")

cinema_films = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    RETURN count(DISTINCT film) AS cnt
""", {"tid": TURKEY_ID})
cinema_film_count = cinema_films[0]['cnt'] if cinema_films else 0

cinema_directors = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(d:Entity)
    RETURN count(DISTINCT d) AS cnt
""", {"tid": TURKEY_ID})
cinema_director_count = cinema_directors[0]['cnt'] if cinema_directors else 0

cinema_actors = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:CAST_MEMBER]->(a:Entity)
    RETURN count(DISTINCT a) AS cnt
""", {"tid": TURKEY_ID})
cinema_actor_count = cinema_actors[0]['cnt'] if cinema_actors else 0

cinema_rel_types = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[r]->(target)
    RETURN DISTINCT type(r) AS rel
    LIMIT 50
""", {"tid": TURKEY_ID})
cinema_rels = [r['rel'] for r in cinema_rel_types]

cinema_2hop = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(d:Entity)
    MATCH (d)-[:PLACE_OF_BIRTH]->(city:Entity)
    RETURN count(*) AS cnt
""", {"tid": TURKEY_ID})
cinema_2hop_count = cinema_2hop[0]['cnt'] if cinema_2hop else 0

cinema_3hop = run("""
    MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
    MATCH (film)-[:DIRECTOR]->(d:Entity)
    MATCH (d)-[:PLACE_OF_BIRTH]->(city:Entity)
    MATCH (city)-[:COUNTRY]->(country:Entity)
    RETURN count(*) AS cnt
""", {"tid": TURKEY_ID})
cinema_3hop_count = cinema_3hop[0]['cnt'] if cinema_3hop else 0

print(f"\n  Film sayısı         : {cinema_film_count:,}")
print(f"  Yönetmen sayısı     : {cinema_director_count:,}")
print(f"  Oyuncu sayısı       : {cinema_actor_count:,}")
print(f"  Relation türleri    : {cinema_rels}")
print(f"  2-hop path sayısı   : {cinema_2hop_count:,}")
print(f"  3-hop path sayısı   : {cinema_3hop_count:,}")


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 3: TÜRK ŞİRKETLERİ
# ══════════════════════════════════════════════════════════════════════════════
section("DOMAIN 3: Türk Şirketleri")

companies = run("""
    MATCH (co:Entity)-[:HEADQUARTERS_LOCATION]->(city:Entity)
          -[:COUNTRY]->(:Entity {entityId: $tid})
    RETURN count(DISTINCT co) AS cnt
""", {"tid": TURKEY_ID})
company_count = companies[0]['cnt'] if companies else 0

company_rel_types = run("""
    MATCH (co:Entity)-[:HEADQUARTERS_LOCATION]->(city:Entity)
          -[:COUNTRY]->(:Entity {entityId: $tid})
    MATCH (co)-[r]->(target)
    RETURN DISTINCT type(r) AS rel
    LIMIT 50
""", {"tid": TURKEY_ID})
company_rels = [r['rel'] for r in company_rel_types]

company_2hop = run("""
    MATCH (co:Entity)-[:HEADQUARTERS_LOCATION]->(city:Entity)
          -[:COUNTRY]->(:Entity {entityId: $tid})
    MATCH (co)-[:INDUSTRY]->(ind:Entity)
    RETURN count(*) AS cnt
""", {"tid": TURKEY_ID})
company_2hop_count = company_2hop[0]['cnt'] if company_2hop else 0

company_3hop = run("""
    MATCH (co:Entity)-[:HEADQUARTERS_LOCATION]->(city:Entity)
          -[:COUNTRY]->(:Entity {entityId: $tid})
    MATCH (co)-[:SUBSIDIARY]->(sub:Entity)
    MATCH (sub)-[:HEADQUARTERS_LOCATION]->(city2:Entity)
    RETURN count(*) AS cnt
""", {"tid": TURKEY_ID})
company_3hop_count = company_3hop[0]['cnt'] if company_3hop else 0

print(f"\n  Şirket sayısı       : {company_count:,}")
print(f"  Relation türleri    : {company_rels}")
print(f"  2-hop path sayısı   : {company_2hop_count:,}")
print(f"  3-hop path sayısı   : {company_3hop_count:,}")


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 4: TÜRK MÜZİĞİ
# ══════════════════════════════════════════════════════════════════════════════
section("DOMAIN 4: Türk Müziği")

musicians = run("""
    MATCH (e:Entity)-[:COUNTRY_OF_CITIZENSHIP]->(:Entity {entityId: $tid})
    WHERE toLower(e.description) CONTAINS 'singer'
       OR toLower(e.description) CONTAINS 'musician'
       OR toLower(e.description) CONTAINS 'composer'
       OR toLower(e.description) CONTAINS 'rapper'
    RETURN count(DISTINCT e) AS cnt
""", {"tid": TURKEY_ID})
musician_count = musicians[0]['cnt'] if musicians else 0

music_rel_types = run("""
    MATCH (e:Entity)-[:COUNTRY_OF_CITIZENSHIP]->(:Entity {entityId: $tid})
    WHERE toLower(e.description) CONTAINS 'singer'
       OR toLower(e.description) CONTAINS 'musician'
       OR toLower(e.description) CONTAINS 'composer'
    MATCH (e)-[r]->(target)
    RETURN DISTINCT type(r) AS rel
    LIMIT 50
""", {"tid": TURKEY_ID})
music_rels = [r['rel'] for r in music_rel_types]

music_2hop = run("""
    MATCH (e:Entity)-[:COUNTRY_OF_CITIZENSHIP]->(:Entity {entityId: $tid})
    WHERE toLower(e.description) CONTAINS 'singer'
       OR toLower(e.description) CONTAINS 'musician'
    MATCH (e)-[:PLACE_OF_BIRTH]->(city:Entity)
    RETURN count(*) AS cnt
""", {"tid": TURKEY_ID})
music_2hop_count = music_2hop[0]['cnt'] if music_2hop else 0

music_3hop = run("""
    MATCH (e:Entity)-[:COUNTRY_OF_CITIZENSHIP]->(:Entity {entityId: $tid})
    WHERE toLower(e.description) CONTAINS 'singer'
       OR toLower(e.description) CONTAINS 'musician'
    MATCH (e)-[:PLACE_OF_BIRTH]->(city:Entity)
    MATCH (city)-[:COUNTRY]->(country:Entity)
    RETURN count(*) AS cnt
""", {"tid": TURKEY_ID})
music_3hop_count = music_3hop[0]['cnt'] if music_3hop else 0

print(f"\n  Müzisyen sayısı     : {musician_count:,}")
print(f"  Relation türleri    : {music_rels}")
print(f"  2-hop path sayısı   : {music_2hop_count:,}")
print(f"  3-hop path sayısı   : {music_3hop_count:,}")


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 5: TÜRK AKADEMİSİ
# ══════════════════════════════════════════════════════════════════════════════
section("DOMAIN 5: Türk Akademisi")

universities = run("""
    MATCH (u:Entity)-[:COUNTRY]->(:Entity {entityId: $tid})
    WHERE toLower(u.description) CONTAINS 'university'
       OR toLower(u.description) CONTAINS 'college'
    RETURN count(DISTINCT u) AS cnt
""", {"tid": TURKEY_ID})
university_count = universities[0]['cnt'] if universities else 0

academia_rel_types = run("""
    MATCH (u:Entity)-[:COUNTRY]->(:Entity {entityId: $tid})
    WHERE toLower(u.description) CONTAINS 'university'
       OR toLower(u.description) CONTAINS 'college'
    MATCH (u)-[r]->(target)
    RETURN DISTINCT type(r) AS rel
    LIMIT 50
""", {"tid": TURKEY_ID})
academia_rels = [r['rel'] for r in academia_rel_types]

academia_2hop = run("""
    MATCH (u:Entity)-[:COUNTRY]->(:Entity {entityId: $tid})
    WHERE toLower(u.description) CONTAINS 'university'
       OR toLower(u.description) CONTAINS 'college'
    MATCH (p:Entity)-[:EDUCATED_AT]->(u)
    RETURN count(*) AS cnt
""", {"tid": TURKEY_ID})
academia_2hop_count = academia_2hop[0]['cnt'] if academia_2hop else 0

academia_3hop = run("""
    MATCH (u:Entity)-[:COUNTRY]->(:Entity {entityId: $tid})
    WHERE toLower(u.description) CONTAINS 'university'
       OR toLower(u.description) CONTAINS 'college'
    MATCH (p:Entity)-[:EDUCATED_AT]->(u)
    MATCH (p)-[:PLACE_OF_BIRTH]->(city:Entity)
    RETURN count(*) AS cnt
""", {"tid": TURKEY_ID})
academia_3hop_count = academia_3hop[0]['cnt'] if academia_3hop else 0

print(f"\n  Üniversite sayısı   : {university_count:,}")
print(f"  Relation türleri    : {academia_rels}")
print(f"  2-hop path sayısı   : {academia_2hop_count:,}")
print(f"  3-hop path sayısı   : {academia_3hop_count:,}")


# ══════════════════════════════════════════════════════════════════════════════
# KARŞILAŞTIRMA TABLOSU
# ══════════════════════════════════════════════════════════════════════════════
section("KARŞILAŞTIRMA TABLOSU — 3.4 KRİTERLERİ")

domains = {
    "Futbol": {
        "seed_entities":    football_club_count,
        "related_entities": football_player_count,
        "relation_types":   len(football_rels),
        "rel_names":        football_rels,
        "2hop_paths":       football_2hop_count,
        "3hop_paths":       football_3hop_count,
    },
    "Sinema": {
        "seed_entities":    cinema_film_count,
        "related_entities": cinema_director_count + cinema_actor_count,
        "relation_types":   len(cinema_rels),
        "rel_names":        cinema_rels,
        "2hop_paths":       cinema_2hop_count,
        "3hop_paths":       cinema_3hop_count,
    },
    "Şirketler": {
        "seed_entities":    company_count,
        "related_entities": 0,
        "relation_types":   len(company_rels),
        "rel_names":        company_rels,
        "2hop_paths":       company_2hop_count,
        "3hop_paths":       company_3hop_count,
    },
    "Müzik": {
        "seed_entities":    musician_count,
        "related_entities": 0,
        "relation_types":   len(music_rels),
        "rel_names":        music_rels,
        "2hop_paths":       music_2hop_count,
        "3hop_paths":       music_3hop_count,
    },
    "Akademi": {
        "seed_entities":    university_count,
        "related_entities": 0,
        "relation_types":   len(academia_rels),
        "rel_names":        academia_rels,
        "2hop_paths":       academia_2hop_count,
        "3hop_paths":       academia_3hop_count,
    },
}

CRITERIA = {
    "seed_entities":  {"min": 10,  "label": "Seed Entity (≥10)"},
    "relation_types": {"min": 5,   "label": "Rel. Türü (≥5)"},
    "2hop_paths":     {"min": 30,  "label": "2-hop Path (≥30)"},
}

print(f"\n  {'Domain':12} | {'Seed':>6} | {'İlgili':>7} | {'Rel Türü':>8} | {'2-hop':>7} | {'3-hop':>7} | Kriter")
print("  " + "─" * 75)

for name, d in domains.items():
    criteria_ok = all([
        d['seed_entities']  >= CRITERIA['seed_entities']['min'],
        d['relation_types'] >= CRITERIA['relation_types']['min'],
        d['2hop_paths']     >= CRITERIA['2hop_paths']['min'],
    ])
    status = "✅ TÜM KRİTER" if criteria_ok else "⚠️  EKSİK"
    print(f"  {name:12} | {d['seed_entities']:>6,} | {d['related_entities']:>7,} | "
          f"{d['relation_types']:>8} | {d['2hop_paths']:>7,} | {d['3hop_paths']:>7,} | {status}")

print(f"\n  Minimum beklentiler: Seed ≥ 10 | Rel. Türü ≥ 5 | 2-hop ≥ 30")


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK
# ══════════════════════════════════════════════════════════════════════════════
section("GRAFİK ÜRETİLİYOR")

domain_names = list(domains.keys())
seed_counts  = [domains[d]['seed_entities']   for d in domain_names]
hop2_counts  = [domains[d]['2hop_paths']      for d in domain_names]
rel_counts   = [domains[d]['relation_types']  for d in domain_names]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#1a1a2e")

colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

for ax in axes:
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.spines['bottom'].set_color('#aaaaaa')
    ax.spines['left'].set_color('#aaaaaa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

bars1 = axes[0].bar(domain_names, seed_counts, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
axes[0].set_title("Seed Entity Sayısı", color="white", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Sayı", color="white")
axes[0].axhline(y=10, color='yellow', linestyle='--', linewidth=1.5, label='Min (10)')
axes[0].legend(facecolor='#2c2c54', labelcolor='white', fontsize=9)
for bar, val in zip(bars1, seed_counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:,}', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
axes[0].set_xticklabels(domain_names, rotation=15, color='white', fontsize=9)

bars2 = axes[1].bar(domain_names, hop2_counts, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
axes[1].set_title("2-Hop Path Sayısı", color="white", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Sayı", color="white")
axes[1].axhline(y=30, color='yellow', linestyle='--', linewidth=1.5, label='Min (30)')
axes[1].legend(facecolor='#2c2c54', labelcolor='white', fontsize=9)
for bar, val in zip(bars2, hop2_counts):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:,}', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
axes[1].set_xticklabels(domain_names, rotation=15, color='white', fontsize=9)

bars3 = axes[2].bar(domain_names, rel_counts, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
axes[2].set_title("Relation Türü Sayısı", color="white", fontsize=12, fontweight="bold")
axes[2].set_ylabel("Sayı", color="white")
axes[2].axhline(y=5, color='yellow', linestyle='--', linewidth=1.5, label='Min (5)')
axes[2].legend(facecolor='#2c2c54', labelcolor='white', fontsize=9)
for bar, val in zip(bars3, rel_counts):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val}', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
axes[2].set_xticklabels(domain_names, rotation=15, color='white', fontsize=9)

fig.suptitle("Phase 2: Türkiye Domain Karşılaştırması\n(Sarı kesik çizgi = minimum kriter)",
             color="white", fontsize=14, fontweight="bold")
plt.tight_layout()

chart_path = os.path.join(OUTPUT_DIR, "phase2_domain_overview_chart.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print(f"\n  ✔ Grafik kaydedildi: {chart_path}")


# ══════════════════════════════════════════════════════════════════════════════
# JSON KAYDET  ✔ phase1_summary JSON'dan okunuyor
# ══════════════════════════════════════════════════════════════════════════════
phase1_summary = _load_phase1_summary()

save_json({
    "phase1_summary":  phase1_summary,   # ✔ artık gerçek değerler
    "domain_analysis": domains,
    "criteria": {
        "min_seed_entities":  10,
        "min_relation_types": 5,
        "min_2hop_paths":     30,
    },
    "note": "Bu dosyaya bakarak domain seçimi yapılacak."
}, "phase2_domain_overview.json")

driver.close()

print("\n" + "=" * 65)
print("  ÖZET — Domain Karşılaştırması Tamamlandı")
print("=" * 65)
print(f"""
  Domain       | Seed  | 2-hop | Rel Türü
  ────────────────────────────────────────
  Futbol       | {football_club_count:>5,} | {football_2hop_count:>5,} | {len(football_rels):>8}
  Sinema       | {cinema_film_count:>5,} | {cinema_2hop_count:>5,} | {len(cinema_rels):>8}
  Şirketler    | {company_count:>5,} | {company_2hop_count:>5,} | {len(company_rels):>8}
  Müzik        | {musician_count:>5,} | {music_2hop_count:>5,} | {len(music_rels):>8}
  Akademi      | {university_count:>5,} | {academia_2hop_count:>5,} | {len(academia_rels):>8}

  → Çıktıya bakarak domain seçimini yap.
""")
