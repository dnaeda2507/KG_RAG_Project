import pandas as pd
from tqdm import tqdm

print("Loading entities...")
entities = {}
with open('wikidata5m_raw_data/wikidata5m_alias/wikidata5m_entity.txt', 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            entities[parts[0]] = parts[1]

print("Loading relations...")
relations = {}
with open('wikidata5m_raw_data/wikidata5m_alias/wikidata5m_relation.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            relations[parts[0]] = parts[1]

print(f"Loaded {len(entities)} entities and {len(relations)} relations")

# Step 1: nodes.csv
print("Creating nodes CSV...")
with open('nodes.csv', 'w', encoding='utf-8') as f:
    f.write('entityId:ID,name,description,:LABEL\n')
    for entity_id, description in tqdm(entities.items()):
        safe_desc = description.replace('"', '""')
        f.write(f'{entity_id},"{safe_desc}","{safe_desc}",Entity\n')

# Step 2: relationships.csv
print("Creating relationships CSV...")
with open('relationships.csv', 'w', encoding='utf-8') as f:
    f.write(':START_ID,:END_ID,:TYPE,relationId\n')
    with open('wikidata5m_raw_data/wikidata5m_all_triplet.txt', 'r', encoding='utf-8') as triplets:
        for line in tqdm(triplets):
            parts = line.strip().split('\t', 2)
            if len(parts) == 3:
                subj, rel, obj = parts
                obj = obj.split('\t')[0]
                rel_name = relations.get(rel, rel).upper().replace(' ', '_').replace(',', '')  # ← virgül temizleme eklendi
                f.write(f'{subj},{obj},{rel_name},{rel}\n')
