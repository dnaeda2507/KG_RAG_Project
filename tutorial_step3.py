from tqdm import tqdm

# 1. NAME + ALIASES — entity.txt'den (ilk alias = name, tümü = aliases)
print("Loading names and aliases...")
entity_names   = {}   # entity_id -> first alias (primary name)
entity_aliases = {}   # entity_id -> pipe-separated all aliases
with open('wikidata5m_raw_data/wikidata5m_alias/wikidata5m_entity.txt', 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            eid = parts[0]
            entity_names[eid]   = parts[1]               # ilk alias = name
            entity_aliases[eid] = '|'.join(parts[1:])    # tümü pipe ile birleştirildi

# 2. DESCRIPTION — text.txt'den (gerçek Wikipedia özeti)
print("Loading descriptions...")
entity_descriptions = {}
with open('wikidata5m_raw_data/wikidata5m_text.txt', 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
            entity_descriptions[parts[0]] = parts[1]

# 3. RELATIONS — relation.txt'den (ilk alias = ilişki ismi)
print("Loading relations...")
relations = {}
with open('wikidata5m_raw_data/wikidata5m_alias/wikidata5m_relation.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            relations[parts[0]] = parts[1]

print(f"Names: {len(entity_names)}, Aliases: {len(entity_aliases)}, "
      f"Descriptions: {len(entity_descriptions)}, Relations: {len(relations)}")

# 4. nodes.csv
# aliases kolonu: '|' ile birleştirilmiş tüm alias listesi
# Neo4j import sonrası SPLIT(e.aliases, '|') ile dizi olarak kullanılabilir
print("Creating nodes CSV...")
all_ids = set(entity_names.keys()) | set(entity_descriptions.keys())

with open('nodes.csv', 'w', encoding='utf-8') as f:
    f.write('entityId:ID,name,description,aliases,:LABEL\n')
    for entity_id in tqdm(all_ids):
        name    = entity_names.get(entity_id, entity_id)
        desc    = entity_descriptions.get(entity_id, "")
        aliases = entity_aliases.get(entity_id, name)   # en azından name yazar
        safe_name    = name.replace('"', '""')
        safe_desc    = desc[:500].replace('"', '""')
        safe_aliases = aliases.replace('"', '""')
        f.write(f'"{entity_id}","{safe_name}","{safe_desc}","{safe_aliases}",Entity\n')

# 5. relationships.csv
print("Creating relationships CSV...")
with open('relationships.csv', 'w', encoding='utf-8') as f:
    f.write(':START_ID,:END_ID,:TYPE,relationId\n')
    with open('wikidata5m_raw_data/wikidata5m_all_triplet.txt', 'r', encoding='utf-8') as triplets:
        for line in tqdm(triplets):
            parts = line.strip().split('\t')
            if len(parts) == 3:
                subj, rel, obj = parts
                rel_name = relations.get(rel, rel).upper().replace(' ', '_').replace(',', '')
                f.write(f'"{subj}","{obj}",{rel_name},{rel}\n')

print("Bitti! Simdi neo4j-admin import calistir.")