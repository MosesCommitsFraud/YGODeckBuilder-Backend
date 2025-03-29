
# deckbuilder.py
import json
import os
import pickle
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from train_reranker import build_feature_vector, GraphManager
import numpy as np
import pickle

# 1) Importiere dein DB-Modell
from db.models import Card
from Emedding.graph.adder import init_deck_graph, add_card_to_graph
# 2) Importiere die Embedding-Funktionen aus embedding.topk
from Emedding.topl import (
    add_embeddings_to_qdrant,
    get_embedding_score,
    get_top30_similar_cards  # <-- Neue Funktion mit query_points (siehe unten)
)

# 3) Importiere die Rule-based-Funktionen aus rulemining.topl
from rulemining.topl import (
    get_candidate_scores,
    YDK_FOLDER,
    get_single_card_score,
    ARCHETYPES_FILE,
    ner_pipe,
    load_card_info_from_db,
    load_full_decks
)

##########################################
# Qdrant / RDF Setup
##########################################
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD, OWL

COLLECTION_NAME = "my_ygo_embeddings"






##########################################
# Demo-Funktionen
##########################################
def build_deck_suggestions(deck_card_ids, rule_scores, embeddings_dict, qdrant_client,
                           main_decks, archetypes):
    """
    Vereint die Rule-basierten Kandidaten (rule_scores) und die Embedding-basierten,
    baut eine Liste und berechnet ggf. fehlende Scores.

    WICHTIG: Wir übergeben hier "main_decks" und "archetypes",
    damit wir NICHT None übergeben müssen, wenn eine Karte
    nicht in den Top-30 rule-basierten war.
    """
    # 1) Hole Embedding-Kandidaten (Top-30) z. B. von der ersten Karte
    if not deck_card_ids:
        return []

    ref_id = int(deck_card_ids[0])  # z. B. die erste Karte als Query
    emb_candidates = get_top30_similar_cards(qdrant_client, COLLECTION_NAME, ref_id, embeddings_dict=embeddings_dict)
    # emb_candidates => [(c_id, embScore), ...]

    # 2) Vereinigung
    rule_ids = set(rule_scores.keys())  # str
    emb_ids = {str(cid) for cid, _ in emb_candidates}
    all_ids = rule_ids | emb_ids

    results = []
    # Dictionary zum schnellen Lookup der Embedding-Scores
    emb_dict = {str(cid): score for cid, score in emb_candidates}

    # 3) Für alle Kandidaten => Scores hinzufügen
    for cid in all_ids:
        candidate = {
            "card_id": cid,
            "rule_score": 0.0,
            "emb_score": 0.0
        }

        # (a) Rule-Score
        if cid in rule_scores:
            candidate["rule_score"] = rule_scores[cid]
        else:
            # Wir haben "main_decks" und "archetypes" hier parat,
            # also geben wir sie an get_single_card_score weiter
            card_info = load_card_info_from_db()
            candidate["rule_score"] = get_single_card_score(
                base_cards=deck_card_ids,
                candidate_card_id=cid,
                main_decks=main_decks,
                ner_pipe=ner_pipe,
                card_info=card_info,
                archetypes=archetypes
            )

        # (b) Embedding-Score
        if cid in emb_dict:
            # kam direkt aus den top-30
            candidate["emb_score"] = emb_dict[cid]
        else:
            # Falls nicht in top-30 => Score manuell ermitteln
            candidate["emb_score"] = get_embedding_score(deck_card_ids, cid, embeddings_dict)

        results.append(candidate)

    # Sortierung (Dummy: rule_score + emb_score)
    # -> In deinem echten Code: ML-ReRank
    results = sorted(results, key=lambda x: (x["rule_score"] + x["emb_score"]), reverse=True)
    return results


def main():
    # 1) DB-Verbindung aufbauen
    engine = create_engine("sqlite:///../data.sqlite")
    Session = sessionmaker(bind=engine)
    session = Session()

    # 2) Embeddings laden
    with open("Emedding/graph_embeddings.pkl", "rb") as f:
        embeddings_dict = pickle.load(f)

    # 3) Qdrant-Setup (In-Memory)
    qdrant_client = QdrantClient(":memory:", prefer_grpc=False)
    dim = len(list(embeddings_dict.values())[0])
    try:
        qdrant_client.delete_collection(COLLECTION_NAME)
    except:
        pass
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    add_embeddings_to_qdrant(qdrant_client, embeddings_dict, COLLECTION_NAME)

    # 4) Seed-Karten & RDF-Deck
    seed_ids = [89631139, 6983839]  # Blue-Eyes, Red-Eyes
    seed_cards = []
    for cid in seed_ids:
        co = session.query(Card).filter(Card.id == cid).one_or_none()
        if co:
            seed_cards.append(co)

    deck_graph, deck_uri = init_deck_graph(session=session, seed_cards=seed_cards, deck_id="my_dekc")
    # Optional: Füge die Seed-Karten zusätzlich in den Graphen ein (falls nötig)
    for cobj in seed_cards:
        add_card_to_graph(deck_graph, cobj, deck_uri)

    # Aktuelle Deck‑IDs als Strings
    deck_card_ids = [str(c.id) for c in seed_cards]

    # 5) Archetypes + Decks laden (für rule-based)
    with open(ARCHETYPES_FILE, "r", encoding="utf-8") as f:
        archetypes = json.load(f)
    full_decks = load_full_decks(YDK_FOLDER)
    print(f"Loaded {len(full_decks)} decks.")
    main_decks = [deck["main"] for deck in full_decks]

    # 6) Rule-based: Hole initiale Top-30 Score-Liste
    card_info = load_card_info_from_db()
    rule_scores = get_candidate_scores(
        base_cards=deck_card_ids,
        main_decks=main_decks,
        card_info=card_info,
        top_n=30,
        archetypes=archetypes,
        ner_pipe=ner_pipe
    )

    # 7) Embedding + Rule zusammenführen
    final_suggestions = build_deck_suggestions(
        deck_card_ids=deck_card_ids,
        rule_scores=rule_scores,
        embeddings_dict=embeddings_dict,
        qdrant_client=qdrant_client,
        main_decks=main_decks,
        archetypes=archetypes
    )

    # 8) Re-Ranking der Kandidaten mittels ML-Modell
    # Lade das trainierte Reranker-Modell
    with open("card_recommender.pkl", "rb") as f:
        reranker_model = pickle.load(f)
    # Lade Card-Infos für die Feature-Berechnung
    card_info_ml = load_card_info_from_db()
    # Initialisiere den GraphManager (stellt graphbasierte Features bereit)
    graph_manager = GraphManager()
    candidate_features = []
    for candidate in final_suggestions:
        candidate_id = int(candidate["card_id"])
        fv = build_feature_vector(
            deck_card_ids,
            candidate_id,
            rule_scores,
            card_info_ml,
            embeddings_dict,
            graph_manager
        )
        candidate_features.append(fv)
    X_candidates = np.array(candidate_features)
    rerank_scores = reranker_model.predict_proba(X_candidates)[:, 1]
    for i, candidate in enumerate(final_suggestions):
        candidate["rerank_score"] = rerank_scores[i]
    final_suggestions = sorted(final_suggestions, key=lambda x: x["rerank_score"], reverse=True)

    # 9) Ausgabe (Top-5)
    print("\n--- TOP-5 Vorschläge (Re-Ranking mit ML-Modell) ---")
    for i, cand in enumerate(final_suggestions[:5], start=1):
        print(f"{i}. Card={cand['card_id']} | Rule={cand['rule_score']:.2f} | Emb={cand['emb_score']:.2f} | Rerank={cand['rerank_score']:.2f}")

    # 10) Beispiel: Neue Karte hinzufügen (Dark Magician)
    new_card_id = 71703785
    new_card = session.query(Card).filter(Card.id == new_card_id).one_or_none()
    if new_card:
        add_card_to_graph(deck_graph, new_card, deck_uri)
        deck_card_ids.append(str(new_card_id))

    # 11) Graph speichern
    deck_graph.serialize("my_deck_abox.ttl", format="turtle")
    print("[INFO] Deck-Graph gespeichert in 'my_deck_abox.ttl'.")


if __name__ == "__main__":
    main()
