from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "Nazperesed07."))

with driver.session() as session:
    result = session.run("RETURN 'Neo4j Connected!' AS message")
    print(result.single()["message"])

driver.close()