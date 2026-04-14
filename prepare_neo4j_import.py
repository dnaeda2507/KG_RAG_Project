"""
Adım 3.2-3.3: Wikidata5M verilerini Neo4j bulk import için CSV formatına dönüştürme.
nodes.csv ve relationships.csv dosyaları oluşturur.
"""
import os
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))
ENTITY_FILE = os.path.join(BASE, "wikidata5m_raw_data", "wikidata5m_alias", "wikidata5m_entity.txt")
RELATION_FILE = os.path.join(BASE, "wikidata5m_raw_data", "wikidata5m_alias", "wikidata5m_relation.txt")
TRIPLET_FILE = os.path.join(BASE, "wikidata5m_raw_data", "wikidata5m_all_triplet.txt")

# ── 1. Entity'leri oku ───────────────────────────────────────────────
print("Loading entities...")
entities = {}
with open(ENTITY_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Entities"):
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            entities[parts[0]] = parts[1]  # ilk alias = isim

print(f"  → {len(entities):,} entity yüklendi")

# ── 2. Relation'ları oku ─────────────────────────────────────────────
print("Loading relations...")
relations = {}
with open(RELATION_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            relations[parts[0]] = parts[1]

print(f"  → {len(relations):,} relation yüklendi")

# ── 3. nodes.csv oluştur ─────────────────────────────────────────────
nodes_path = os.path.join(BASE, "nodes.csv")
print(f"\nCreating {nodes_path} ...")

with open(nodes_path, "w", encoding="utf-8") as f:
    f.write("entityId:ID,name,description,:LABEL\n")
    for entity_id, name in tqdm(entities.items(), desc="Nodes CSV"):
        safe = name.replace('"', '""').replace("\n", " ").replace("\r", "")
        f.write(f'{entity_id},"{safe}","{safe}",Entity\n')

print(f"  → nodes.csv yazıldı ({len(entities):,} node)")

# ── 4. relationships.csv oluştur ─────────────────────────────────────
rels_path = os.path.join(BASE, "relationships.csv")
print(f"\nCreating {rels_path} ...")

count = 0
skipped = 0
with open(rels_path, "w", encoding="utf-8") as f:
    f.write(":START_ID,:END_ID,:TYPE,relationId\n")
    with open(TRIPLET_FILE, "r", encoding="utf-8") as triplets:
        for line in tqdm(triplets, desc="Rels CSV"):
            parts = line.strip().split("\t")
            if len(parts) == 3:
                subj, rel, obj = parts
                rel_name = relations.get(rel, rel)
                # Neo4j relationship type: büyük harf, boşluk → _, özel karakter temizle
                rel_type = rel_name.upper().replace(" ", "_").replace(",", "").replace("'", "").replace('"', "")
                f.write(f"{subj},{obj},{rel_type},{rel}\n")
                count += 1
            else:
                skipped += 1

print(f"  → relationships.csv yazıldı ({count:,} rel, {skipped} atlandı)")
print(f"\n✅ Tamamlandı! nodes.csv ve relationships.csv hazır.")
print(f"   Sonraki adım: Neo4j bulk import")
