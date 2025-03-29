import os
import json
import random
import pickle
import math
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

from rdflib import Namespace, Graph
# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# DB-Modelle
from db.models import Base, Deck, DeckCard, Card
from generate_emeddings import get_embedding_score

# Regelbasierte Features
from rulemining.topl import (
    get_candidate_scores,
    get_single_card_score,
    load_card_info_from_db,
    load_full_decks,
    ner_pipe,
    ARCHETYPES_FILE,
    YDK_FOLDER
)

# ----------------- Embeddings und Graph -----------------
EMBEDDINGS_PICKLE = "Emedding/graph_embeddings.pkl"
RDF_GRAPH_FILE = "my_deck_abox.ttl"


class GraphManager:
    def __init__(self):
        self.graph = Graph()
        self.ns = Namespace("http://example.org/ygo#")
        self.load_graph()

    def load_graph(self):
        if os.path.exists(RDF_GRAPH_FILE):
            self.graph.parse(RDF_GRAPH_FILE, format="turtle")

    def get_card_distance(self, card1_id: int, card2_id: int) -> float:
        """Berechnet die kürzeste Pfaddistanz zwischen zwei Karten im RDF-Graph"""
        # Hier sollte die eigentliche Graph Traversal Logik implementiert werden
        # Platzhalter: Zufällige Distanz zwischen 0-3
        return random.uniform(0, 3)


graph_manager = GraphManager()


# ----------------- Datenladefunktionen -----------------
def load_decks_and_cards(db_url="sqlite:///../data.sqlite") -> Tuple[List[Deck], set]:
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    decks = session.query(Deck).all()
    all_card_ids = {dc.card_id for deck in decks for dc in deck.deck_cards}

    session.close()
    return decks, all_card_ids


def load_card_info(db_url="sqlite:///../data.sqlite") -> Dict[int, dict]:
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    cards = session.query(Card).all()
    card_info = {}

    for card in cards:
        effect_features = [
            int(bool(getattr(card, f"effect_{effect}")))
            for effect in [
                'search', 'destroy', 'negate', 'draw', 'special_summon',
                'banish', 'send_gy', 'recover_lp', 'inflict_damage', 'equip',
                'modify_stats', 'protect', 'discard', 'change_position',
                'return', 'shuffle', 'copy', 'counter', 'token_summon',
                'deck_manipulation'
            ]
        ]

        card_info[card.id] = {
            "id": card.id,
            "type": card.type,
            "race": card.race,
            "archetype": card.archetype,
            "ban_status": card.ban_tcg.lower() if card.ban_tcg else "unlimited",
            "is_staple": card.is_staple,
            "effect_features": effect_features,
            "referenced_cards": json.loads(card.referenced_cards) if card.referenced_cards else []
        }

    session.close()
    return card_info


# ----------------- Feature Engineering -----------------
def calculate_effect_presence(deck_card_ids: List[int], card_info: dict) -> List[float]:
    """Berechnet die Präsenz jedes Effekts im Deck"""
    effect_counts = [0] * 20
    for cid in deck_card_ids:
        effects = card_info.get(cid, {}).get("effect_features", [0] * 20)
        for i in range(20):
            effect_counts[i] += effects[i]
    return [count / len(deck_card_ids) if len(deck_card_ids) > 0 else 0 for count in effect_counts]


def build_feature_vector(
        deck_card_ids: List[int],
        candidate_id: int,
        rule_scores: dict,
        card_info: dict,
        embeddings_dict: dict,
        graph_manager: GraphManager
) -> List[float]:
    # Basis-Features
    features = []

    # 1. Embedding Score
    embedding_score = get_embedding_score(deck_card_ids, candidate_id, embeddings_dict)
    features.append(embedding_score)

    # 2. Rule-based Score
    features.append(rule_scores.get(str(candidate_id), 0.0))

    # 3. Graph Features
    min_distance = min(
        [graph_manager.get_card_distance(cid, candidate_id) for cid in deck_card_ids[:5]],
        default=3.0
    )
    features.extend([min_distance])

    # 4. Kartentyp-Verteilung im Deck
    type_counts = defaultdict(int)
    for cid in deck_card_ids:
        card_type = card_info.get(cid, {}).get("type", "")
        if "Spell" in card_type:
            type_counts["spell"] += 1
        elif "Trap" in card_type:
            type_counts["trap"] += 1
        else:
            type_counts["monster"] += 1
    total = len(deck_card_ids) or 1
    features.extend([
        type_counts["spell"] / total,
        type_counts["trap"] / total,
        type_counts["monster"] / total
    ])

    # 5. Effekt-Präsenz im Deck
    effect_presence = calculate_effect_presence(deck_card_ids, card_info)
    features.extend(effect_presence)

    # 6. Karten-spezifische Features
    candidate_info = card_info.get(candidate_id, {})
    features.extend([
        1 if candidate_info.get("is_staple") else 0,
        {"banned": 0, "limited": 1, "semi-limited": 2, "unlimited": 3}.get(
            candidate_info.get("ban_status", "unlimited"), 3
        )
    ])

    # 7. Effekt-Features der Kandidatenkarte
    features.extend(candidate_info.get("effect_features", [0] * 20))

    return features


# ----------------- Modelltraining -----------------
def prepare_training_data(
        decks: List[Deck],
        card_info: dict,
        embeddings_dict: dict,
        archetypes: dict,
        main_decks: list,
        negative_ratio: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    # Archetypen und Regelbasierte Scores vorberechnen
    with open(ARCHETYPES_FILE) as f:
        archetypes = json.load(f)

    for deck in decks:
        deck_card_ids = [str(dc.card_id) for dc in deck.deck_cards]
        int_card_ids = [int(cid) for cid in deck_card_ids]

        # Rule Scores berechnen
        rule_scores = get_candidate_scores(
            deck_card_ids,
            main_decks,
            card_info,
            top_n=30,
            archetypes=archetypes,
            ner_pipe=ner_pipe
        )

        # Positive Samples
        for cid in int_card_ids:
            fv = build_feature_vector(
                int_card_ids,
                cid,
                rule_scores,
                card_info,
                embeddings_dict,
                graph_manager
            )
            X.append(fv)
            y.append(1)

        # Negative Samples
        all_cards = list(card_info.keys())
        neg_samples = random.sample(
            [c for c in all_cards if c not in int_card_ids],
            k=int(len(int_card_ids) * negative_ratio)
        )
        for cid in neg_samples:
            fv = build_feature_vector(
                int_card_ids,
                cid,
                rule_scores,
                card_info,
                embeddings_dict,
                graph_manager
            )
            X.append(fv)
            y.append(0)

    return np.array(X), np.array(y)


def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict_proba(X_test)[:, 1]
    print(f"Modellperformance:\nAUC: {roc_auc_score(y_test, y_pred):.3f}")

    return model


# ----------------- Hauptprogramm -----------------
def main():
    # 1. Daten laden
    decks, all_cards = load_decks_and_cards()
    card_info = load_card_info()
    main_decks = [d["main"] for d in load_full_decks(YDK_FOLDER)]

    # 2. Embeddings laden
    with open(EMBEDDINGS_PICKLE, "rb") as f:
        embeddings_dict = pickle.load(f)
    with open(ARCHETYPES_FILE, "r", encoding="utf-8") as f:
        archetypes = json.load(f)

    # 3. Trainingsdaten vorbereiten
    X, y = prepare_training_data(decks, card_info, embeddings_dict, archetypes, main_decks)
    print(f"Trainingsdaten Shape: {X.shape}")

    # 4. Modell trainieren
    model = train_model(X, y)

    # 5. Modell speichern
    with open("card_recommender.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Modelltraining abgeschlossen!")


if __name__ == "__main__":
    main()