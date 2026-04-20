import os

files = {
    "entity.txt (alias)": "wikidata5m_raw_data/wikidata5m_alias/wikidata5m_entity.txt",
    "relation.txt":        "wikidata5m_raw_data/wikidata5m_alias/wikidata5m_relation.txt",
    "all_triplet.txt":     "wikidata5m_raw_data/wikidata5m_all_triplet.txt",
    "text.txt (desc)":     "wikidata5m_raw_data/wikidata5m_text.txt",
}

for label, path in files.items():
    print(f"\n{'='*60}")
    print(f"DOSYA: {label}")
    print(f"PATH : {path}")
    if not os.path.exists(path):
        print("  !! DOSYA BULUNAMADI !!")
        continue
    size_mb = os.path.getsize(path) / (1024*1024)
    print(f"BOYUT: {size_mb:.1f} MB")
    print("ILK 3 SATIR:")
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            parts = line.strip().split('\t')
            print(f"  Satir {i+1}: {len(parts)} sutun")
            for j, p in enumerate(parts):
                preview = p[:80] + "..." if len(p) > 80 else p
                print(f"    [{j}] = '{preview}'")