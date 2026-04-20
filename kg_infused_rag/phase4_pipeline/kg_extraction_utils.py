import re

def extract_answer_from_kg_and_passages(query, kg_triples, passages, relation_patterns=None):
    """
    Soruya uygun relation'ı tespit eder, önce KG'den, sonra passage'dan çıkarım yapar.
    relation_patterns: {relation: regex}
    """
    if relation_patterns is None:
        relation_patterns = {
            'place_of_birth': r'was born in ([^.,;\n]+)',
            'date_of_birth': r'was born on ([^.,;\n]+)',
            'occupation': r'is a[n]? ([^.,;\n]+)',
            'citizenship': r'is a citizen of ([^.,;\n]+)',
            'award_received': r'received the ([^.,;\n]+) award',
            # Gerekirse diğer relation ve regex'ler eklenebilir
        }

    # 1. Relation tespiti (anahtar kelime ile basit)
    for rel, pattern in relation_patterns.items():
        if rel.replace('_', ' ') in query.lower():
            # 2. KG'den çıkarım
            for triple in kg_triples:
                if triple['relation'] == rel:
                    return triple['tail']
            # 3. Passage'dan çıkarım
            for p in passages:
                match = re.search(pattern, p.get('content', ''), re.IGNORECASE)
                if match:
                    return match.group(1)
            break
    # 4. LLM fallback
    return "Cannot be determined from available information."
