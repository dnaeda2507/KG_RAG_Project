import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neo4j import GraphDatabase

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OUTPUT_PHASE1


driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
os.makedirs(OUTPUT_PHASE1, exist_ok=True)


def run(cypher, params=None):
    with driver.session() as session:
        return [record.data() for record in session.run(cypher, params or {})]


def save_json(filename, data):
    path = os.path.join(OUTPUT_PHASE1, filename)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def detect_turkey_entity():
    # 2.3.1 Step 1: Search for candidate entities containing Turkey/Turkiye.
    candidates = run(
        """
        MATCH (e:Entity)
        WHERE toLower(e.name) IN ['turkiye', 'turkey', 'türkiye']
           OR (e.aliases IS NOT NULL AND ANY(
                 a IN split(e.aliases, '|')
                 WHERE toLower(a) IN ['turkey', 'türkiye', 'turkiye']
               ))
     RETURN e.entityId AS entityId,
               e.name AS name,
         e.description AS description
        ORDER BY size(e.entityId) ASC
        LIMIT 10
        """
    )

    if not candidates:
        raise RuntimeError("Turkey entity could not be found with a simple name search.")

    # Select the highest-priority candidate as the Turkey anchor entity.
    turkey = candidates[0]

    # 2.3.1 Step 2: Extract direct outgoing triples from Turkey.
    outgoing = run(
        """
        MATCH (e:Entity {entityId: $tid})-[r]->(target)
        RETURN e.entityId AS source_id,
               e.name AS source_name,
               type(r) AS relation,
               target.entityId AS target_id,
               target.name AS target_name
        """,
        {"tid": turkey["entityId"]},
    )

    # 2.3.1 Step 3: Extract direct incoming triples to Turkey.
    incoming = run(
        """
        MATCH (source:Entity)-[r]->(e:Entity {entityId: $tid})
        RETURN source.entityId AS source_id,
               source.name AS source_name,
               type(r) AS relation,
               e.entityId AS target_id,
               e.name AS target_name
        """,
        {"tid": turkey["entityId"]},
    )

    direct_connected_entities = run(
        """
        MATCH (e:Entity)-[r]->(:Entity {entityId: $tid})
        RETURN count(DISTINCT e) AS count
        """,
        {"tid": turkey["entityId"]},
    )[0]["count"]

    direct_outgoing_entities = run(
        """
        MATCH (:Entity {entityId: $tid})-[r]->(e:Entity)
        RETURN count(DISTINCT e) AS count
        """,
        {"tid": turkey["entityId"]},
    )[0]["count"]

    total_direct_incoming_triples = run(
        """
        MATCH (:Entity)-[r]->(:Entity {entityId: $tid})
        RETURN count(r) AS count
        """,
        {"tid": turkey["entityId"]},
    )[0]["count"]

    total_direct_outgoing_triples = run(
        """
        MATCH (:Entity {entityId: $tid})-[r]->(:Entity)
        RETURN count(r) AS count
        """,
        {"tid": turkey["entityId"]},
    )[0]["count"]

    outgoing_relation_counts = run(
        """
        MATCH (:Entity {entityId: $tid})-[r]->()
        RETURN type(r) AS relation_type, count(*) AS frequency
        ORDER BY frequency DESC
        """,
        {"tid": turkey["entityId"]},
    )

    incoming_relation_counts = run(
        """
        MATCH ()-[r]->(:Entity {entityId: $tid})
        RETURN type(r) AS relation_type, count(*) AS frequency
        ORDER BY frequency DESC
        LIMIT 20
        """,
        {"tid": turkey["entityId"]},
    )

    categorical_distribution = run(
        """
        MATCH (neighbor:Entity)-[]-(:Entity {entityId: $tid})
        OPTIONAL MATCH (neighbor)-[:INSTANCE_OF]->(category:Entity)
        WITH neighbor, coalesce(category.name, category.entityId, 'uncategorized') AS category_name
        RETURN category_name,
               count(DISTINCT neighbor) AS entity_count
        ORDER BY entity_count DESC
        LIMIT 30
        """,
        {"tid": turkey["entityId"]},
    )

    def build_result():
        sample_outgoing_triples = outgoing[:20]
        sample_incoming_triples = incoming[:20]

        print(f"Selected Turkey entity: {turkey['entityId']} - {turkey['name']}")
        print(f"Direct incoming entities: {direct_connected_entities:,}")
        print(f"Direct outgoing entities: {direct_outgoing_entities:,}")
        print(f"Total direct incoming triples: {total_direct_incoming_triples:,}")
        print(f"Total direct outgoing triples: {total_direct_outgoing_triples:,}")

        print("\nSample outgoing triples:")
        for triple in sample_outgoing_triples[:10]:
            print(
                f"  {triple['source_id']:12} --[{triple['relation']:30}]--> "
                f"{triple['target_id']}"
            )

        print("\nSample incoming triples:")
        for triple in sample_incoming_triples[:10]:
            print(
                f"  {triple['source_id']:12} --[{triple['relation']:30}]--> "
                f"{triple['target_id']}"
            )

        return {
            "turkey_candidates": [
                {
                    "entityId": item["entityId"],
                    "name": item["name"],
                    "description": item["description"],
                }
                for item in candidates
            ],
            "selected_turkey_entity": turkey,
            "direct_connected_entity_count": direct_connected_entities,
            "direct_outgoing_entity_count": direct_outgoing_entities,
            "total_incoming_entities": direct_connected_entities,
            "total_outgoing_entities": direct_outgoing_entities,
            "total_direct_incoming_triples": total_direct_incoming_triples,
            "total_direct_outgoing_triples": total_direct_outgoing_triples,
            "sample_outgoing_triples": sample_outgoing_triples,
            "sample_incoming_triples": sample_incoming_triples,
            "outgoing_triples": outgoing,
            "incoming_triples": incoming,
            "all_directly_connected_triples": {
                "incoming": incoming,
                "outgoing": outgoing,
            },
            "outgoing_relation_counts": outgoing_relation_counts,
            "incoming_relation_counts": incoming_relation_counts,
            "outgoing_relations": outgoing_relation_counts,
            "incoming_relations_top20": incoming_relation_counts,
            "categorical_distribution": categorical_distribution,
        }

    result = build_result()
    return turkey["entityId"], result

def analyze_turkish_cities(turkey_id):
    # 2.3.2 Step 1: Find cities whose COUNTRY relation points to Turkey.
    cities = run(
        """
        MATCH (city:Entity)-[:COUNTRY]->(:Entity {entityId: $tid})
        MATCH (city)-[:INSTANCE_OF]->(type:Entity)
        WHERE type.name CONTAINS 'city'
        RETURN DISTINCT city.entityId AS city_id,
               city.name AS city_name,
               city.description AS description
        ORDER BY city_name
        LIMIT 200
        """,
        {"tid": turkey_id},
    )

    # Identify major cities explicitly for the report.
    major_city_ids = run(
        """
        MATCH (city:Entity)-[:COUNTRY]->(:Entity {entityId: $tid})
        WHERE city.name CONTAINS 'Istanbul'
           OR city.name CONTAINS 'İstanbul'
           OR city.name CONTAINS 'Ankara'
           OR city.name CONTAINS 'Izmir'
           OR city.name CONTAINS 'İzmir'
        RETURN city.entityId AS city_id, city.name AS city_name, city.description AS description
        ORDER BY city_name
        """,
        {"tid": turkey_id},
    )

    # 2.3.2 Step 2: Count how many different entities are connected to each city.
    city_connections = run(
        """
        MATCH (entity:Entity)-[r]-(city:Entity)-[:COUNTRY]->(:Entity {entityId: $tid})
        MATCH (city)-[:INSTANCE_OF]->(type:Entity)
        WHERE type.name CONTAINS 'city'
        RETURN city.entityId AS city_id,
               city.name AS city_name,
               count(DISTINCT entity) AS connected_entities
        ORDER BY connected_entities DESC
        LIMIT 50
        """,
        {"tid": turkey_id},
    )

    city_birth_counts = run(
        """
        MATCH (person:Entity)-[:PLACE_OF_BIRTH]->(city:Entity)-[:COUNTRY]->(:Entity {entityId: $tid})
        MATCH (city)-[:INSTANCE_OF]->(type:Entity)
        WHERE type.name CONTAINS 'city'
        RETURN city.entityId AS city_id,
               city.name AS city_name,
               count(DISTINCT person) AS people_born_here
        ORDER BY people_born_here DESC
        LIMIT 20
        """,
        {"tid": turkey_id},
    )

    result = {
        "turkish_cities": cities,
        "major_cities": major_city_ids,
        "city_connection_counts": city_connections,
        "city_birth_counts": city_birth_counts,
    }
    return result


def analyze_relation_types(turkey_id):
    # 2.3.3 Step 1: Check the required relation types in the Turkey context.
    country_count = run(
        """
        MATCH (e:Entity)-[:COUNTRY]->(:Entity {entityId: $tid})
        RETURN count(DISTINCT e) AS count
        """,
        {"tid": turkey_id},
    )[0]["count"]

    place_of_birth_count = run(
        """
        MATCH (person:Entity)-[:PLACE_OF_BIRTH]->(place:Entity)
        WHERE (place)-[:COUNTRY]->(:Entity {entityId: $tid})
        RETURN count(DISTINCT person) AS count
        """,
        {"tid": turkey_id},
    )[0]["count"]

    headquarters_location_count = run(
        """
        MATCH (org:Entity)-[:HEADQUARTERS_LOCATION]->(place:Entity)
        WHERE (place)-[:COUNTRY]->(:Entity {entityId: $tid})
        RETURN count(DISTINCT org) AS count
        """,
        {"tid": turkey_id},
    )[0]["count"]

    member_of_sports_team_count = run(
        """
        MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(team:Entity)
        WHERE (team)-[:COUNTRY]->(:Entity {entityId: $tid})
        RETURN count(DISTINCT player) AS count
        """,
        {"tid": turkey_id},
    )[0]["count"]

    director_count = run(
        """
        MATCH (film:Entity)-[:DIRECTOR]->(director:Entity)
        WHERE (film)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
        RETURN count(DISTINCT director) AS count
        """,
        {"tid": turkey_id},
    )[0]["count"]

    educated_at_count = run(
        """
        MATCH (person:Entity)-[:EDUCATED_AT]->(inst:Entity)
        WHERE (inst)-[:COUNTRY]->(:Entity {entityId: $tid})
        RETURN count(DISTINCT person) AS count
        """,
        {"tid": turkey_id},
    )[0]["count"]

    # 2.3.3 Step 2: Find the most frequent relation types around Turkey.
    top_relation_types = run(
        """
        MATCH (e:Entity)-[r]->(target:Entity)
        WHERE target.entityId = $tid
           OR (target)-[:COUNTRY]->(:Entity {entityId: $tid})
        RETURN type(r) AS relation_type, count(*) AS frequency
        ORDER BY frequency DESC
        LIMIT 20
        """,
        {"tid": turkey_id},
    )

    relation_usage = {
        "country": country_count,
        "place_of_birth": place_of_birth_count,
        "headquarters_location": headquarters_location_count,
        "member_of_sports_team": member_of_sports_team_count,
        "director": director_count,
        "educated_at": educated_at_count,
    }

    result = {
        "required_relation_usage": relation_usage,
        "top_relation_types": top_relation_types,
    }
    return result


def analyze_domain_density(turkey_id):
    # 2.4 Question 4: Compare the main domains in the Turkey context.
    sports_count = run(
        """
        MATCH (player:Entity)-[:MEMBER_OF_SPORTS_TEAM]->(team:Entity)
        WHERE (team)-[:COUNTRY]->(:Entity {entityId: $tid})
        RETURN count(DISTINCT player) AS count
        """,
        {"tid": turkey_id},
    )[0]["count"]

    cinema_count = run(
        """
        MATCH (film:Entity)-[:COUNTRY_OF_ORIGIN]->(:Entity {entityId: $tid})
        RETURN count(DISTINCT film) AS count
        """,
        {"tid": turkey_id},
    )[0]["count"]

    business_count = run(
        """
        MATCH (org:Entity)-[:HEADQUARTERS_LOCATION]->(city:Entity)
        WHERE (city)-[:COUNTRY]->(:Entity {entityId: $tid})
        RETURN count(DISTINCT org) AS count
        """,
        {"tid": turkey_id},
    )[0]["count"]

    academia_count = run(
        """
        MATCH (person:Entity)-[:EDUCATED_AT]->(inst:Entity)
        WHERE (inst)-[:COUNTRY]->(:Entity {entityId: $tid})
        RETURN count(DISTINCT person) AS count
        """,
        {"tid": turkey_id},
    )[0]["count"]

    music_count = run(
        """
        MATCH (artist:Entity)-[:COUNTRY_OF_CITIZENSHIP]->(:Entity {entityId: $tid})
        WHERE toLower(coalesce(artist.description, '')) CONTAINS 'singer'
           OR toLower(coalesce(artist.description, '')) CONTAINS 'musician'
           OR toLower(coalesce(artist.description, '')) CONTAINS 'composer'
           OR toLower(coalesce(artist.description, '')) CONTAINS 'rapper'
        RETURN count(DISTINCT artist) AS count
        """,
        {"tid": turkey_id},
    )[0]["count"]

    domain_counts = {
        "sports": sports_count,
        "cinema": cinema_count,
        "business": business_count,
        "academia": academia_count,
        "music": music_count,
    }

    result = {
        "domain_counts": domain_counts,
        "most_dense_domain": max(domain_counts, key=domain_counts.get),
    }
    return result



def build_summary(turkey_data, city_data, relation_data, domain_data):
    # 2.5 Expected outputs summary for quick reporting.
    summary = {
        "turkey_entity_id": turkey_data["selected_turkey_entity"]["entityId"],
        "direct_connected_entity_count": turkey_data["direct_connected_entity_count"],
        "direct_outgoing_entity_count": turkey_data["direct_outgoing_entity_count"],
        "total_incoming_entities": turkey_data["total_incoming_entities"],
        "total_outgoing_entities": turkey_data["total_outgoing_entities"],
        "total_direct_incoming_triples": turkey_data["total_direct_incoming_triples"],
        "total_direct_outgoing_triples": turkey_data["total_direct_outgoing_triples"],
        "top_10_turkey_related_categories": turkey_data["categorical_distribution"][:10],
        "number_of_detected_cities": len(city_data["turkish_cities"]),
        "top_10_cities_by_connected_entities": city_data["city_connection_counts"][:10],
        "top_10_relation_types": relation_data["top_relation_types"][:10],
        "domain_counts": domain_data["domain_counts"],
        "most_dense_domain": domain_data["most_dense_domain"],
    }
    return summary


def write_outputs(turkey_data, city_data, relation_data, domain_data, summary):
    save_json("phase1_turkey_stats.json", turkey_data)
    save_json("phase1_city_report.json", city_data)
    save_json("phase1_relation_freq.json", relation_data)
    save_json("phase1_domain_counts.json", domain_data)
    save_json("phase1_summary.json", summary)


def generate_visuals(city_data, relation_data, domain_data):
    # Visual outputs required for Phase 1 reporting.
    colors = [
        "#e74c3c",
        "#e67e22",
        "#f1c40f",
        "#2ecc71",
        "#3498db",
        "#9b59b6",
        "#1abc9c",
        "#e91e63",
        "#34495e",
    ]

    figure, axes = plt.subplots(2, 2, figsize=(18, 13))
    figure.suptitle("Phase 1: Wikidata5M Turkey Domain Analysis", fontsize=16, fontweight="bold")

    domain_counts = domain_data["domain_counts"]
    domain_labels = list(domain_counts.keys())
    domain_values = list(domain_counts.values())
    bars_plot = axes[0, 0].barh(domain_labels, domain_values, color=colors[:len(domain_labels)])
    axes[0, 0].set_title("Entity Count by Domain", fontweight="bold")
    axes[0, 0].set_xlabel("Entity Count")
    max_value = max(domain_values) if any(value > 0 for value in domain_values) else 1
    for bar, value in zip(bars_plot, domain_values):
        axes[0, 0].text(
            bar.get_width() + max_value * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            fontsize=9,
        )
    axes[0, 0].tick_params(axis="y", labelsize=8)

    top_birth_cities = city_data["city_birth_counts"][:10]
    if top_birth_cities:
        city_labels = [item["city_name"][:20] for item in top_birth_cities]
        city_values = [item["people_born_here"] for item in top_birth_cities]
        axes[0, 1].bar(city_labels, city_values, color="#3498db", edgecolor="white")
        axes[0, 1].set_title("People Born in Each City", fontweight="bold")
        axes[0, 1].tick_params(axis="x", rotation=45, labelsize=8)
        axes[0, 1].set_ylabel("People Count")
    else:
        axes[0, 1].text(0.5, 0.5, "No data found", ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[0, 1].set_title("People Born in Each City", fontweight="bold")

    top_relations = relation_data["top_relation_types"][:12]
    if top_relations:
        relation_labels = [item["relation_type"] for item in top_relations]
        relation_values = [item["frequency"] for item in top_relations]
        axes[1, 0].barh(relation_labels[::-1], relation_values[::-1], color="#e74c3c")
        axes[1, 0].set_title("Top 12 Relations in the Turkey Context", fontweight="bold")
        axes[1, 0].set_xlabel("Frequency")
        axes[1, 0].tick_params(axis="y", labelsize=7)

    non_zero_domains = {key: value for key, value in domain_counts.items() if value > 0}
    if non_zero_domains:
        axes[1, 1].pie(
            non_zero_domains.values(),
            labels=non_zero_domains.keys(),
            autopct="%1.1f%%",
            startangle=140,
            colors=colors[:len(non_zero_domains)],
        )
        axes[1, 1].set_title("Domain Distribution (%)", fontweight="bold")
    else:
        axes[1, 1].text(0.5, 0.5, "No data found", ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Domain Distribution (%)", fontweight="bold")

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_PHASE1, "phase1_visuals.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


def main():
    print_section("Phase 1 - 2.3.1 Turkey Main Entity Detection")
    turkey_id, turkey_data = detect_turkey_entity()
    print(f"Selected Turkey entity: {turkey_id}")

    print_section("Phase 1 - 2.3.2 Turkish Cities Detection")
    city_data = analyze_turkish_cities(turkey_id)
    print(f"Detected Turkish cities: {len(city_data['turkish_cities'])}")

    print_section("Phase 1 - 2.3.3 Relevant Relation Types Analysis")
    relation_data = analyze_relation_types(turkey_id)
    print("Required relation analysis completed.")

    print_section("Phase 1 - 2.4 Research Questions")
    domain_data = analyze_domain_density(turkey_id)
    print(f"Most dense domain: {domain_data['most_dense_domain']}")

    print_section("Phase 1 - 2.5 Expected Outputs")
    summary = build_summary(turkey_data, city_data, relation_data, domain_data)
    write_outputs(turkey_data, city_data, relation_data, domain_data, summary)
    generate_visuals(city_data, relation_data, domain_data)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        driver.close()