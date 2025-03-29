import os
import json
import pickle
import random
import numpy as np
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Importiere deine DB-Modelle (z.B. Card, Deck, DeckCard)
from db.models import Base, Card, Deck, DeckCard

# Importiere die neuen ML-Funktionen (Feature-Vektor-Berechnung, GraphManager) aus deinem Train-Reranker-Modul
from train_reranker import build_feature_vector as new_build_feature_vector, GraphManager

# Importiere die Rule-based-Funktionen aus rulemining.topl
try:
    from rulemining.topl import (
        get_candidate_scores,
        load_card_info_from_db,
        load_full_decks,
        ARCHETYPES_FILE,
        ner_pipe,
        YDK_FOLDER
    )
except ImportError:
    print("Fehler: Das Modul rulemining.topl wurde nicht gefunden. Bitte stelle sicher, dass es im PYTHONPATH ist.")
    exit(1)

# ----------------- KONFIGURATION -----------------
PICKLE_DATEI = "Emedding/graph_embeddings.pkl"    # Pfad zur Pickle-Datei mit den Embeddings
SQLITE_DB = "../data.sqlite"                        # Pfad zur SQLite-Datenbank
COLLECTION_NAME = "my_ygo_embeddings"               # Name der Collection in Qdrant
TOP_N = 5                                         # Anzahl final zurückgegebener Ergebnisse (nach Re-Ranking)
CANDIDATE_LIMIT = 30                              # Anzahl Kandidaten, die aus Qdrant abgerufen werden
MODEL_PICKLE = "card_recommender.pkl"             # Pfad zum trainierten Reranking-Modell (neues Modell)
FREQUENT_COMBOS_FILE = "frequent_combos.json"       # Datei mit frequent combos (optional)

DESIRED_MAIN_SIZE = 60    # Zielgröße des finalen Main Decks
DESIRED_EXTRA_SIZE = 15   # Zielgröße des finalen Extra Decks
DESIRED_SIDE_SIZE = 15    # Zielgröße des finalen Side Decks

# Falls benötigt, hier ggf. die Feature-Vektor-Dimension definieren
FEATURE_VECTOR_DIM = 50
# -------------------------------------------------

# --- Hilfsfunktionen zur Karteninfo ---

def load_all_card_info(db_url=f"sqlite:///{SQLITE_DB}"):
    """Lädt alle Karteninformationen aus der DB in ein Dictionary."""
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    card_info_dict = {}
    for card_obj in session.query(Card).all():
        ban_status = card_obj.ban_tcg
        if ban_status:
            if ban_status.lower() == "banned":
                ban_status_num = 0
            elif ban_status.lower() == "limited":
                ban_status_num = 1
            elif ban_status.lower() == "semi-limited":
                ban_status_num = 2
            else:
                ban_status_num = 3
        else:
            ban_status_num = 3

        try:
            referenced_cards = json.loads(card_obj.referenced_cards) if card_obj.referenced_cards else []
        except Exception:
            referenced_cards = []
        try:
            referenced_archetypes = json.loads(card_obj.referenced_archetypes) if card_obj.referenced_archetypes else []
        except Exception:
            referenced_archetypes = []
        try:
            referenced_races = json.loads(card_obj.referenced_races) if card_obj.referenced_races else []
        except Exception:
            referenced_races = []

        card_info_dict[card_obj.id] = {
            "id": card_obj.id,
            "name": card_obj.name,
            "type": card_obj.type or "",
            "human_readable_card_type": card_obj.human_readable_card_type,
            "frame_type": card_obj.frame_type,
            "desc": card_obj.desc,
            "race": card_obj.race,
            "archetype": card_obj.archetype,
            "ban_status_num": ban_status_num,
            "is_staple": card_obj.is_staple,
            "effect_search": card_obj.effect_search,
            "effect_destroy": card_obj.effect_destroy,
            "effect_negate": card_obj.effect_negate,
            "effect_draw": card_obj.effect_draw,
            "effect_special_summon": card_obj.effect_special_summon,
            "effect_banish": card_obj.effect_banish,
            "effect_send_gy": card_obj.effect_send_gy,
            "effect_recover_lp": card_obj.effect_recover_lp,
            "effect_inflict_damage": card_obj.effect_inflict_damage,
            "effect_equip": card_obj.effect_equip,
            "effect_modify_stats": card_obj.effect_modify_stats,
            "effect_protect": card_obj.effect_protect,
            "effect_discard": card_obj.effect_discard,
            "effect_change_position": card_obj.effect_change_position,
            "effect_return": card_obj.effect_return,
            "effect_shuffle": card_obj.effect_shuffle,
            "effect_copy": card_obj.effect_copy,
            "effect_counter": card_obj.effect_counter,
            "effect_token_summon": card_obj.effect_token_summon,
            "effect_deck_manipulation": card_obj.effect_deck_manipulation,
            "referenced_cards": referenced_cards,
            "referenced_archetypes": referenced_archetypes,
            "referenced_races": referenced_races,
            "effect_features": [
                1 if card_obj.effect_search else 0,
                1 if card_obj.effect_destroy else 0,
                1 if card_obj.effect_negate else 0,
                1 if card_obj.effect_draw else 0,
                1 if card_obj.effect_special_summon else 0,
                1 if card_obj.effect_banish else 0,
                1 if card_obj.effect_send_gy else 0,
                1 if card_obj.effect_recover_lp else 0,
                1 if card_obj.effect_inflict_damage else 0,
                1 if card_obj.effect_equip else 0,
                1 if card_obj.effect_modify_stats else 0,
                1 if card_obj.effect_protect else 0,
                1 if card_obj.effect_discard else 0,
                1 if card_obj.effect_change_position else 0,
                1 if card_obj.effect_return else 0,
                1 if card_obj.effect_shuffle else 0,
                1 if card_obj.effect_copy else 0,
                1 if card_obj.effect_counter else 0,
                1 if card_obj.effect_token_summon else 0,
                1 if card_obj.effect_deck_manipulation else 0
            ]
        }
    session.close()
    return card_info_dict

def default_card_info(c_id):
    """Standardwerte, falls für eine Karte keine Info vorhanden ist."""
    return {
        "id": c_id,
        "name": "",
        "type": "",
        "human_readable_card_type": "",
        "frame_type": "",
        "desc": "",
        "race": None,
        "archetype": None,
        "ban_status_num": 3,
        "is_staple": False,
        "effect_search": False,
        "effect_destroy": False,
        "effect_negate": False,
        "effect_draw": False,
        "effect_special_summon": False,
        "effect_banish": False,
        "effect_send_gy": False,
        "effect_recover_lp": False,
        "effect_inflict_damage": False,
        "effect_equip": False,
        "effect_modify_stats": False,
        "effect_protect": False,
        "effect_discard": False,
        "effect_change_position": False,
        "effect_return": False,
        "effect_shuffle": False,
        "effect_copy": False,
        "effect_counter": False,
        "effect_token_summon": False,
        "effect_deck_manipulation": False,
        "referenced_cards": [],
        "referenced_archetypes": [],
        "referenced_races": [],
        "effect_features": [0] * 20
    }

# --- Feature Engineering & Re-Ranking ---
# Wir verwenden hier die in "train_reranker" definierte Funktion new_build_feature_vector und den GraphManager.
def build_feature_vector(deck_cards, c_id, rule_scores, card_infos, embeddings_dict, graph_manager):
    # Für einen Kandidaten c_id wird der Feature-Vektor berechnet
    return new_build_feature_vector(deck_cards, c_id, rule_scores, card_infos, embeddings_dict, graph_manager)

def re_rank_with_model(deck_cards, candidate_ids, model, rule_scores, embeddings_dict, card_infos, graph_manager):
    """
    Filtert zunächst Kandidaten, die laut Ban-Liste bereits ihre maximale Anzahl im Deck erreicht haben,
    berechnet anschließend den Feature-Vektor und wendet das ML-Modell an.
    Die Kandidaten werden nach der zurückgegebenen Wahrscheinlichkeit sortiert.
    """
    legal_candidate_ids = [cid for cid in candidate_ids if can_add_card(deck_cards, cid, card_infos)]
    results = []
    for c_id in legal_candidate_ids:
        fv = build_feature_vector(deck_cards, c_id, rule_scores, card_infos, embeddings_dict, graph_manager)
        score = model.predict_proba([fv])[0][1]
        results.append((c_id, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# --- Qdrant-Funktionen ---

def add_embeddings_to_qdrant(client: QdrantClient, embeddings_dict, collection_name):
    points = []
    for k_id, vec in embeddings_dict.items():
        try:
            point_id = int(k_id)
        except ValueError:
            print(f"Überspringe Schlüssel '{k_id}' (nicht numerisch).")
            continue
        if isinstance(vec, np.ndarray):
            vec = vec.tolist()
        else:
            vec = list(vec)
        payload = {"original_id": str(k_id)}
        p = PointStruct(id=point_id, vector=vec, payload=payload)
        points.append(p)
    print(f"Füge {len(points)} Embeddings in Qdrant ein...")
    BATCH_SIZE = 1000
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i: i + BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"Batch {i // BATCH_SIZE + 1} mit {len(batch)} Einträgen hinzugefügt.")

def query_similar_cards(client: QdrantClient, collection_name: str, card_id: int, candidate_limit=30):
    points = client.retrieve(collection_name=collection_name, ids=[card_id], with_vectors=True)
    if not points:
        print(f"Keine Embedding für Karte {card_id} gefunden.")
        return []
    query_vector = points[0].vector
    if not query_vector:
        print(f"Für Karte {card_id} existiert kein Vektor in Qdrant.")
        return []
    search_result = client.search(collection_name=collection_name, query_vector=query_vector, limit=candidate_limit + 1)
    filtered = [(sp.id, sp.score) for sp in search_result if sp.id != card_id]
    return filtered[:candidate_limit]

def suggest_cards_for_deck(client: QdrantClient, collection_name: str, deck_card_ids, candidate_limit=30):
    points = client.retrieve(collection_name=collection_name, ids=deck_card_ids, with_vectors=True)
    vectors = [p.vector for p in points if p.vector is not None]
    if not vectors:
        print("Keine Vektoren verfügbar, Abbruch.")
        return []
    deck_embedding = np.mean(np.array(vectors), axis=0).tolist()
    search_result = client.search(
        collection_name=collection_name,
        query_vector=deck_embedding,
        limit=candidate_limit + len(deck_card_ids)
    )
    filtered = [(sp.id, sp.score) for sp in search_result if sp.id not in deck_card_ids]
    return filtered[:candidate_limit]

# --- Deck-Building-Funktionen ---

def max_copies_allowed(ban_status_num):
    if ban_status_num == 0:
        return 0
    elif ban_status_num == 1:
        return 1
    elif ban_status_num == 2:
        return 2
    else:
        return 3

def can_add_card(deck_cards, card_id, card_infos):
    info = card_infos.get(card_id, default_card_info(card_id))
    allowed = max_copies_allowed(info.get("ban_status_num", 3))
    current_count = deck_cards.count(card_id)
    return current_count < allowed

def is_extra_deck_card(card_id, card_infos):
    info = card_infos.get(card_id, default_card_info(card_id))
    card_type = info.get("type", "").lower()
    if any(keyword in card_type for keyword in ["fusion", "synchro", "xyz", "link", "pendulum"]):
        return True
    return False

def build_main_and_extra_decks_from_seed(client, collection_name, seed_ids_main, seed_ids_extra,
                                           model, rule_scores, embeddings_dict, card_infos, graph_manager,
                                           desired_main_size=DESIRED_MAIN_SIZE, desired_extra_size=DESIRED_EXTRA_SIZE,
                                           candidate_limit=CANDIDATE_LIMIT):
    main_deck = list(seed_ids_main)
    extra_deck = list(seed_ids_extra)
    while len(main_deck) < desired_main_size:
        candidates = suggest_cards_for_deck(client, collection_name, main_deck, candidate_limit=candidate_limit)
        if not candidates:
            print("Keine weiteren Kandidaten gefunden. Abbruch.")
            break
        candidate_ids = [cid for cid, score in candidates]
        re_ranked = re_rank_with_model(main_deck, candidate_ids, model, rule_scores, embeddings_dict, card_infos, graph_manager)
        added = False
        for c_id, score in re_ranked:
            if is_extra_deck_card(c_id, card_infos):
                if len(extra_deck) < desired_extra_size and can_add_card(extra_deck, c_id, card_infos):
                    extra_deck.append(c_id)
                    added = True
                    print(f"Extra Deck hinzugefügt: Karte {c_id}. Extra Deck Größe: {len(extra_deck)}/{desired_extra_size}")
                    break
                else:
                    continue
            else:
                if can_add_card(main_deck, c_id, card_infos):
                    main_deck.append(c_id)
                    added = True
                    print(f"Main Deck hinzugefügt: Karte {c_id}. Main Deck Größe: {len(main_deck)}/{desired_main_size}")
                    break
        if not added:
            print("Keine passende Karte mehr gefunden, die den Ban-/Limit-Regeln entspricht.")
            break
    return main_deck, extra_deck

def build_side_deck_from_seed(client, collection_name, seed_ids, model, rule_scores, embeddings_dict, card_infos, graph_manager,
                              deck_size=DESIRED_SIDE_SIZE, candidate_limit=CANDIDATE_LIMIT, main_deck_ids=None):
    if main_deck_ids is None:
        main_deck_ids = []
    deck = list(seed_ids)
    while len(deck) < deck_size:
        candidates = suggest_cards_for_deck(client, collection_name, deck, candidate_limit=candidate_limit)
        candidates = [(cid, score) for cid, score in candidates if cid not in main_deck_ids]
        if not candidates:
            print("Keine weiteren Kandidaten für Side Deck gefunden. Abbruch.")
            break
        candidate_ids = [cid for cid, score in candidates]
        re_ranked = re_rank_with_model(deck, candidate_ids, model, rule_scores, embeddings_dict, card_infos, graph_manager)
        added = False
        for c_id, score in re_ranked:
            if c_id not in main_deck_ids and can_add_card(deck, c_id, card_infos):
                deck.append(c_id)
                added = True
                print(f"Side Deck hinzugefügt: Karte {c_id}. Side Deck Größe: {len(deck)}/{deck_size}")
                break
        if not added:
            print("Keine passende Side Deck Karte mehr gefunden, die den Regeln entspricht.")
            break
    return deck

def build_ydk_string(main_deck_cards, extra_deck_cards, side_deck_cards):
    # Sortiere die Karten in jedem Abschnitt (z. B. numerisch aufsteigend)
    main_sorted = sorted(main_deck_cards)
    extra_sorted = sorted(extra_deck_cards)
    side_sorted = sorted(side_deck_cards)
    lines = []
    lines.append("#created by Qdrant-Deck-Builder")
    lines.append("#main")
    for c_id in main_sorted:
        lines.append(str(c_id))
    lines.append("#extra")
    for c_id in extra_sorted:
        lines.append(str(c_id))
    lines.append("!side")
    for c_id in side_sorted:
        lines.append(str(c_id))
    return "\n".join(lines)

# --- MAIN-Prozess mit interaktiver Auswahl ---

def main():
    # DB-Setup
    engine = create_engine(f"sqlite:///{SQLITE_DB}")
    Session = sessionmaker(bind=engine)
    session = Session()

    # Embeddings laden
    try:
        with open(PICKLE_DATEI, "rb") as f:
            embeddings_dict = pickle.load(f)
        print(f"Es wurden {len(embeddings_dict)} Embeddings aus '{PICKLE_DATEI}' geladen.")
    except FileNotFoundError:
        print(f"Die Datei '{PICKLE_DATEI}' wurde nicht gefunden.")
        return
    except Exception as e:
        print("Fehler beim Laden der Pickle-Datei:", e)
        return

    if not embeddings_dict:
        print("Das embeddings_dict ist leer. Abbruch.")
        return

    # Qdrant-Client (In-Memory)
    client = QdrantClient(":memory:", prefer_grpc=False)
    beispiel_id = next(iter(embeddings_dict))
    dim = len(embeddings_dict[beispiel_id])
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    print(f"Collection '{COLLECTION_NAME}' wurde neu erstellt (Dimension={dim}).")
    add_embeddings_to_qdrant(client, embeddings_dict, COLLECTION_NAME)

    # Reranking-Modell laden
    try:
        with open(MODEL_PICKLE, "rb") as f:
            model = pickle.load(f)
        print("Re-Ranking Modell erfolgreich geladen.")
    except Exception as e:
        print("Fehler beim Laden des Re-Ranking Modells:", e)
        return

    # Zusätzliche Daten laden für rule-based scoring
    try:
        with open(ARCHETYPES_FILE, "r", encoding="utf-8") as f:
            archetypes = json.load(f)
        full_decks = load_full_decks(YDK_FOLDER)
        main_decks_rule = [deck["main"] for deck in full_decks]
        card_info_for_rule = load_card_info_from_db()
    except Exception as e:
        print("Fehler beim Laden der rule-based Daten:", e)
        return

    # Für ML und Re-Ranking laden wir alle Karteninformationen
    card_infos = load_all_card_info(f"sqlite:///{SQLITE_DB}")

    # Initialisiere den GraphManager (liefert graph-basierte Features)
    graph_manager = GraphManager()

    # Interaktive Schleife
    while True:
        print("\nWähle:")
        print(" 1) Suche Top-N ähnliche Karten zu EINER Karten-ID (mit Re-Ranking)")
        print(" 2) Suche Top-N ähnliche Karten zu EINEM GANZEN DECK (mit Re-Ranking)")
        print(" 3) Beenden")
        print(" 4) Erstelle ein komplettes Deck (Main, Extra, Side) aus Seed-Karten und gebe .ydk aus")
        auswahl = input("Eingabe: ").strip()

        if auswahl == "1":
            card_id_str = input("Gib eine Karten-ID ein (z.B. '35283277'): ").strip()
            if not card_id_str:
                print("Keine ID eingegeben.")
                continue
            try:
                card_id = int(card_id_str)
            except ValueError:
                print("Fehler: Bitte eine rein numerische ID eingeben.")
                continue

            # Berechne rule_scores basierend auf dem aktuellen Karten-Kontext (Einzelne Karte)
            rule_scores = get_candidate_scores(
                base_cards=[str(card_id)],
                main_decks=main_decks_rule,
                card_info=card_info_for_rule,
                top_n=30,
                archetypes=archetypes,
                ner_pipe=ner_pipe
            )

            candidates = query_similar_cards(client, COLLECTION_NAME, card_id, candidate_limit=CANDIDATE_LIMIT)
            if candidates:
                candidate_ids = [cid for cid, score in candidates]
                deck_context = [card_id]
                re_ranked = re_rank_with_model(deck_context, candidate_ids, model, rule_scores, embeddings_dict, card_infos, graph_manager)
                final_results = re_ranked[:TOP_N]
                print(f"\nTop {TOP_N} Ergebnisse für Karte {card_id} (nach Re-Ranking):")
                result_ids = [res[0] for res in final_results]
                cards_in_db = session.query(Card).filter(Card.id.in_(result_ids)).all()
                card_map = {c.id: c for c in cards_in_db}
                for i, (cid, score) in enumerate(final_results, start=1):
                    card_obj = card_map.get(cid)
                    card_name = card_obj.name if card_obj else "Unbekannt"
                    print(f"{i}. ID={cid} ({card_name}), Score={score:.4f}")
            else:
                print("Keine Ergebnisse oder Karte nicht in Qdrant gefunden.")

        elif auswahl == "2":
            deck_input = input("Gib mehrere Karten-IDs (komma-getrennt) ein: ").strip()
            if not deck_input:
                print("Keine IDs eingegeben.")
                continue

            raw_ids = [x.strip() for x in deck_input.split(",") if x.strip()]
            try:
                deck_ids = [int(x) for x in raw_ids]
            except ValueError:
                print("Fehler: Bitte nur rein numerische IDs eingeben.")
                continue

            # Berechne rule_scores basierend auf dem aktuellen Deck
            rule_scores = get_candidate_scores(
                base_cards=[str(x) for x in deck_ids],
                main_decks=main_decks_rule,
                card_info=card_info_for_rule,
                top_n=30,
                archetypes=archetypes,
                ner_pipe=ner_pipe
            )

            candidates = suggest_cards_for_deck(client, COLLECTION_NAME, deck_ids, candidate_limit=CANDIDATE_LIMIT)
            if candidates:
                candidate_ids = [cid for cid, score in candidates]
                deck_context = deck_ids
                re_ranked = re_rank_with_model(deck_context, candidate_ids, model, rule_scores, embeddings_dict, card_infos, graph_manager)
                final_results = re_ranked[:TOP_N]
                print(f"\nTop {TOP_N} Ergebnisse (Deck-Embedding, nach Re-Ranking):")
                suggestion_ids = [s[0] for s in final_results]
                cards_in_db = session.query(Card).filter(Card.id.in_(suggestion_ids)).all()
                card_map = {c.id: c for c in cards_in_db}
                for i, (cid, score) in enumerate(final_results, start=1):
                    card_obj = card_map.get(cid)
                    card_name = card_obj.name if card_obj else "Unbekannt"
                    print(f"{i}. ID={cid} ({card_name}), Score={score:.4f}")
            else:
                print("Keine Vorschläge oder keine Embeddings gefunden.")

        elif auswahl == "3":
            print("Programm wird beendet.")
            break

        elif auswahl == "4":
            seed_input_main = input("Gib deine Seed-Karten-IDs für das Main Deck (komma-getrennt) ein: ").strip()
            seed_input_extra = input("Gib deine Seed-Karten-IDs für das Extra Deck (komma-getrennt, optional) ein: ").strip()
            seed_input_side = input("Gib deine Seed-Karten-IDs für das Side Deck (komma-getrennt, optional) ein: ").strip()

            if not seed_input_main:
                print("Für das Main Deck müssen Seed-Karten angegeben werden. Abbruch.")
                continue

            try:
                seed_ids_main = [int(x.strip()) for x in seed_input_main.split(",") if x.strip()]
            except ValueError:
                print("Fehler: Bitte nur rein numerische IDs für das Main Deck eingeben.")
                continue

            try:
                seed_ids_extra = [int(x.strip()) for x in seed_input_extra.split(",") if x.strip()]
            except ValueError:
                seed_ids_extra = []
            try:
                seed_ids_side = [int(x.strip()) for x in seed_input_side.split(",") if x.strip()]
            except ValueError:
                seed_ids_side = []

            print(f"Starte Main/Extra Deck Building mit Seeds: Main: {seed_ids_main} | Extra: {seed_ids_extra} ...")
            # Berechne rule_scores basierend auf den Seed-Karten für das Main Deck
            rule_scores = get_candidate_scores(
                base_cards=[str(x) for x in seed_ids_main],
                main_decks=main_decks_rule,
                card_info=card_info_for_rule,
                top_n=30,
                archetypes=archetypes,
                ner_pipe=ner_pipe
            )
            final_main_deck, final_extra_deck = build_main_and_extra_decks_from_seed(
                client, COLLECTION_NAME, seed_ids_main, seed_ids_extra,
                model, rule_scores, embeddings_dict, card_infos, graph_manager,
                desired_main_size=DESIRED_MAIN_SIZE, desired_extra_size=DESIRED_EXTRA_SIZE
            )
            print(f"Main Deck fertig mit {len(final_main_deck)} Karten.")
            print(f"Extra Deck fertig mit {len(final_extra_deck)} Karten.")

            print("Starte Side Deck Building...")
            # Für das Side Deck erneut rule_scores basierend auf dem Main Deck berechnen
            rule_scores_side = get_candidate_scores(
                base_cards=[str(x) for x in final_main_deck],
                main_decks=main_decks_rule,
                card_info=card_info_for_rule,
                top_n=30,
                archetypes=archetypes,
                ner_pipe=ner_pipe
            )
            final_side_deck = build_side_deck_from_seed(
                client, COLLECTION_NAME, seed_ids_side,
                model, rule_scores_side, embeddings_dict, card_infos, graph_manager,
                deck_size=DESIRED_SIDE_SIZE, main_deck_ids=final_main_deck
            )
            print(f"Side Deck fertig mit {len(final_side_deck)} Karten.")

            final_deck = {
                "main": final_main_deck,
                "extra": final_extra_deck,
                "side": final_side_deck
            }
            ydk_content = build_ydk_string(final_deck["main"], final_deck["extra"], final_deck["side"])
            print("\n===== .ydk-Inhalt (sortiert) =====\n")
            print(ydk_content)
            print("\n===== Ende der .ydk-Datei =====\n")
            with open("my_deck.ydk", "w", encoding="utf-8") as f:
                f.write(ydk_content)
            print("Deck in my_deck.ydk gespeichert.")

        else:
            print("Ungültige Eingabe. Bitte 1, 2, 3 oder 4 eingeben.")

    session.close()

if __name__ == "__main__":
    main()
