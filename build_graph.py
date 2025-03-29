# build_graph.py
import networkx as nx
from collections import defaultdict
from db.db import get_session
from db.models import Deck


def build_card_graph():
    """
    Baut einen Graphen auf, in dem Knoten Karten sind und eine gewichtete Kante
    zwischen zwei Karten besteht, wenn sie in mindestens einem Deck gemeinsam vorkommen.
    Das Gewicht entspricht der Anzahl der gemeinsamen Decks.
    """
    G = nx.Graph()
    edge_weights = defaultdict(int)

    with get_session() as session:
        decks = session.query(Deck).all()
        for deck in decks:
            # Sammle alle Karten-IDs, die in diesem Deck vorkommen.
            card_ids = [dc.card_id for dc in deck.deck_cards]
            # Füge für jedes Kartenpaar ein Gewicht hinzu.
            for i in range(len(card_ids)):
                for j in range(i + 1, len(card_ids)):
                    a, b = sorted((card_ids[i], card_ids[j]))
                    edge_weights[(a, b)] += 1
                    # Stelle sicher, dass die Knoten im Graphen existieren.
                    if not G.has_node(a):
                        G.add_node(a)
                    if not G.has_node(b):
                        G.add_node(b)

    # Füge die Kanten mit dem berechneten Gewicht dem Graphen hinzu.
    for (a, b), weight in edge_weights.items():
        G.add_edge(a, b, weight=weight)

    return G


if __name__ == "__main__":
    # Baue den Graphen aus den Deck-Daten
    G = build_card_graph()
    # Speichere den Graphen in eine GraphML-Datei
    nx.write_graphml(G, "card_graph.graphml")
    print("Graph wurde erfolgreich in 'card_graph.graphml' gespeichert.")
