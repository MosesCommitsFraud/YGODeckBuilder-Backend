import pickle
import numpy as np
import random
import json
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Importiere deine DB-Modelle (z. B. Card, Deck, DeckCard)
from db.models import Base, Card, Deck, DeckCard

# ----------------- KONFIGURATION -----------------
PICKLE_DATEI = "graph_embeddings.pkl"  # Pfad zur Pickle-Datei mit den Embeddings
SQLITE_DB = "data.sqlite"  # Pfad zur SQLite-Datenbank
COLLECTION_NAME = "my_ygo_embeddings"  # Name der Collection in Qdrant
TOP_N = 5  # Anzahl final zurückgegebener Ergebnisse (nach Re-Ranking)
CANDIDATE_LIMIT = 30  # Anzahl Kandidaten, die aus Qdrant abgerufen werden
MODEL_PICKLE = "my_random_forest.pkl"  # Pfad zum trainierten Modell
FREQUENT_COMBOS_FILE = "frequent_combos.json"  # Datei mit frequent combos (optional)

DESIRED_MAIN_SIZE = 40   # Größe des finalen Main Decks
DESIRED_EXTRA_SIZE = 15  # Größe des finalen Extra Decks
DESIRED_SIDE_SIZE = 15   # Größe des finalen Side Decks
# -------------------------------------------------

# --- Helper-Funktionen zur Karteninfo ---

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

# --- Feature-Berechnung und Re-Ranking ---

def get_embedding_score(deck_cards, c_id, embeddings_dict):
    """Berechnet die Cosinus-Ähnlichkeit zwischen dem durchschnittlichen Deck-Embedding und dem Candidate-Embedding."""
    if c_id not in embeddings_dict:
        return 0.0
    candidate_vector = np.array(embeddings_dict[c_id])
    deck_vectors = [np.array(embeddings_dict[x]) for x in deck_cards if x in embeddings_dict]
    if not deck_vectors:
        return 0.0
    deck_avg = np.mean(deck_vectors, axis=0)
    dot = np.dot(candidate_vector, deck_avg)
    norm_candidate = np.linalg.norm(candidate_vector)
    norm_deck = np.linalg.norm(deck_avg)
    if norm_candidate == 0 or norm_deck == 0:
        return 0.0
    return float(dot / (norm_candidate * norm_deck))

def build_feature_vector(deck_cards, c_id, usage_stats, combos, embeddings_dict, card_infos):
    """
    Erzeugt den Feature-Vektor für die (Deck, Candidate)-Kombination.
    Die Features umfassen den Embedding-Score, usage-Statistik, Synergie, Deck-Zusammensetzung u.a.
    """
    deck_set = set(deck_cards)
    embedding_score = get_embedding_score(deck_cards, c_id, embeddings_dict)
    usage_val = usage_stats.get(c_id, 0.0)

    # Synergy: Zähle erfüllte Combos
    synergy_count = 0
    new_deck = deck_set.union({c_id})
    for combo_dict in combos:
        combo_items = combo_dict.get("itemsets", [])
        combo_set = set(map(int, combo_items))
        if combo_set.issubset(new_deck):
            synergy_count += 1
    synergy_norm = synergy_count / len(deck_cards) if deck_cards else 0

    # Deck-Zusammensetzung
    spell_count = 0
    trap_count = 0
    monster_count = 0
    deck_archetypes = []
    deck_races = []
    for dcard in deck_cards:
        info = card_infos.get(dcard, default_card_info(dcard))
        ctype = info.get("type", "")
        if "spell" in ctype.lower():
            spell_count += 1
        elif "trap" in ctype.lower():
            trap_count += 1
        else:
            monster_count += 1
        arch = info.get("archetype")
        if arch:
            deck_archetypes.append(arch)
        race = info.get("race")
        if race:
            deck_races.append(race)
    total_cards = len(deck_cards) if deck_cards else 1
    spell_pct = spell_count / total_cards
    trap_pct = trap_count / total_cards
    monster_pct = monster_count / total_cards

    cand_info = card_infos.get(c_id, default_card_info(c_id))
    ban_stat = cand_info.get("ban_status_num", 3)
    cand_arche = cand_info.get("archetype")
    cand_race = cand_info.get("race")
    same_arch_count = sum(1 for a in deck_archetypes if a == cand_arche) if cand_arche else 0
    same_race_count = sum(1 for r in deck_races if r == cand_race) if cand_race else 0

    # Referenzierte Karten/Archetypen/Races
    referenced_cards = cand_info.get("referenced_cards", [])
    num_ref_cards = len(referenced_cards)
    count_ref_cards_in_deck = 0
    if num_ref_cards > 0:
        for ref in referenced_cards:
            try:
                if int(ref) in deck_set:
                    count_ref_cards_in_deck += 1
            except ValueError:
                continue
    norm_ref_cards = count_ref_cards_in_deck / num_ref_cards if num_ref_cards > 0 else 0

    referenced_archetypes = cand_info.get("referenced_archetypes", [])
    num_ref_arch = len(referenced_archetypes)
    count_ref_arch_in_deck = sum(1 for ref in referenced_archetypes if ref in deck_archetypes) if num_ref_arch > 0 else 0
    norm_ref_arch = count_ref_arch_in_deck / num_ref_arch if num_ref_arch > 0 else 0

    referenced_races = cand_info.get("referenced_races", [])
    num_ref_races = len(referenced_races)
    count_ref_races_in_deck = sum(1 for ref in referenced_races if ref in deck_races) if num_ref_races > 0 else 0
    norm_ref_races = count_ref_races_in_deck / num_ref_races if num_ref_races > 0 else 0

    effect_features = cand_info.get("effect_features", [0] * 20)

    feat_vec = [
        embedding_score,  # Feature 0
        usage_val,        # Feature 1
        synergy_norm,     # Feature 2
        spell_pct,        # Feature 3
        trap_pct,         # Feature 4
        monster_pct,      # Feature 5
        ban_stat,         # Feature 6
        same_arch_count,  # Feature 7
        same_race_count,  # Feature 8
        norm_ref_cards,   # Feature 9
        norm_ref_arch,    # Feature 10
        norm_ref_races,   # Feature 11
        1 if cand_info.get("is_staple") else 0  # Feature 12
    ]
    feat_vec.extend(effect_features)  # Features 13-32
    return feat_vec

def re_rank_with_model(deck_cards, candidate_ids, model, combos, embeddings_dict, card_infos):
    """
    Berechnet für jede Candidate-Karte den Feature-Vektor und wendet das trainierte Modell an.
    Anschließend werden die Kandidaten nach der vom Modell zurückgegebenen Wahrscheinlichkeit sortiert.
    """
    results = []
    for c_id in candidate_ids:
        fv = build_feature_vector(deck_cards, c_id, usage_stats, combos, embeddings_dict, card_infos)
        score = model.predict_proba([fv])[0][1]
        results.append((c_id, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def load_usage_stats(session, deck_list):
    """Berechnet für alle Karten die relative Häufigkeit (usage_stats) in den Decks."""
    usage_stats = defaultdict(int)
    for deck in deck_list:
        deck_cards = [dc.card_id for dc in deck.deck_cards]
        for c_id in set(deck_cards):
            usage_stats[c_id] += 1
    for c_id in usage_stats:
        usage_stats[c_id] = usage_stats[c_id] / len(deck_list) if len(deck_list) else 0.0
    return usage_stats

def load_combos():
    """Lädt frequent combos aus der JSON-Datei (falls vorhanden)."""
    try:
        with open(FREQUENT_COMBOS_FILE, "r") as f:
            combos = json.load(f)
        return combos
    except FileNotFoundError:
        print("Keine frequent_combos.json gefunden. Combos bleiben leer.")
        return []

# --- Qdrant-Funktionen ---

def add_embeddings_to_qdrant(client: QdrantClient, embeddings_dict, collection_name):
    """Fügt alle Embeddings als Points (in Batches) in Qdrant ein."""
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
    """
    Sucht zu einer einzelnen Karten-ID die ähnlichsten Kandidaten in Qdrant.
    Es werden candidate_limit Ergebnisse (ohne die eigene Karte) abgerufen.
    """
    points = client.retrieve(collection_name=collection_name, ids=[card_id], with_vectors=True)
    if not points:
        print(f"Keine Embedding für Karte {card_id} gefunden.")
        return []
    query_vector = points[0].vector
    if not query_vector:
        print(f"Für Karte {card_id} existiert kein Vector in Qdrant.")
        return []
    search_result = client.search(collection_name=collection_NAME, query_vector=query_vector, limit=candidate_limit + 1)
    filtered = [(sp.id, sp.score) for sp in search_result if sp.id != card_id]
    return filtered[:candidate_limit]

def suggest_cards_for_deck(client: QdrantClient, collection_name: str, deck_card_ids, candidate_limit=30):
    """
    Bildet das Durchschnitts-Embedding eines Decks und sucht in Qdrant
    nach candidate_limit ähnlichen Karten (ausgenommen die im Deck enthaltenen).
    """
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
    """
    Gibt die maximal erlaubte Anzahl Kopien einer Karte basierend auf ban_status_num zurück.
      0 → banned (0 Kopien), 1 → limited (1 Kopie), 2 → semi-limited (2 Kopien), 3 → unlimited (3 Kopien).
    """
    if ban_status_num == 0:
        return 0
    elif ban_status_num == 1:
        return 1
    elif ban_status_num == 2:
        return 2
    else:
        return 3

def can_add_card(deck_cards, card_id, card_infos):
    """
    Überprüft, ob eine weitere Kopie von card_id in das Deck darf
    (Ban-Status und Anzahl bereits enthaltener Kopien).
    """
    info = card_infos.get(card_id, default_card_info(card_id))
    ban_status_num = info.get("ban_status_num", 3)
    allowed = max_copies_allowed(ban_status_num)
    if allowed == 0:
        return False
    current_count = deck_cards.count(card_id)
    return current_count < allowed

def is_extra_deck_card(card_id, card_infos):
    """
    Prüft, ob eine Karte als Extra-Deck-Kandidat gelten kann,
    basierend auf dem Kartentyp (Fusion, Synchro, XYZ, Link, Pendulum).
    """
    info = card_infos.get(card_id, default_card_info(card_id))
    card_type = info.get("type", "").lower()
    if any(keyword in card_type for keyword in ["fusion", "synchro", "xyz", "link", "pendulum"]):
        return True
    return False

def build_main_and_extra_decks_from_seed(client, collection_name, seed_ids_main, seed_ids_extra,
                                           model, combos, embeddings_dict, card_infos,
                                           desired_main_size=DESIRED_MAIN_SIZE, desired_extra_size=DESIRED_EXTRA_SIZE,
                                           candidate_limit=CANDIDATE_LIMIT):
    """
    Baut in einem gemeinsamen Prozess das Main und Extra Deck aus Seed-Karten.
    Ist der Top-Kandidat ein Extra-Deck-Kandidat, wird er ins Extra Deck aufgenommen (falls dort noch Platz ist).
    Wenn das Extra Deck voll ist, wird dieser Kandidat übersprungen und es wird eine Karte gesucht, die in das Main Deck passt.
    """
    main_deck = list(seed_ids_main)
    extra_deck = list(seed_ids_extra)
    while len(main_deck) < desired_main_size:
        candidates = suggest_cards_for_deck(client, collection_name, main_deck, candidate_limit=candidate_limit)
        if not candidates:
            print("Keine weiteren Kandidaten gefunden. Abbruch.")
            break
        candidate_ids = [cid for cid, score in candidates]
        re_ranked = re_rank_with_model(main_deck, candidate_ids, model, combos, embeddings_dict, card_infos)
        added = False
        for c_id, score in re_ranked:
            if is_extra_deck_card(c_id, card_infos):
                # Wenn Kandidat extra-Deck-typisch ist:
                if len(extra_deck) < desired_extra_size and can_add_card(extra_deck, c_id, card_infos):
                    extra_deck.append(c_id)
                    added = True
                    print(f"Extra Deck hinzugefügt: Karte {c_id}. Extra Deck Größe: {len(extra_deck)}/{desired_extra_size}")
                    break
                else:
                    # Extra Deck voll – Kandidat überspringen
                    continue
            else:
                # Kandidat passt ins Main Deck
                if can_add_card(main_deck, c_id, card_infos):
                    main_deck.append(c_id)
                    added = True
                    print(f"Main Deck hinzugefügt: Karte {c_id}. Main Deck Größe: {len(main_deck)}/{desired_main_size}")
                    break
        if not added:
            print("Keine passende Karte mehr gefunden, die den Ban-/Limit-Regeln entspricht.")
            break
    return main_deck, extra_deck

def build_side_deck_from_seed(client, collection_name, seed_ids, model, usage_stats, combos, embeddings_dict, card_infos,
                              deck_size=DESIRED_SIDE_SIZE, candidate_limit=CANDIDATE_LIMIT, main_deck_ids=None):
    """
    Baut ein Side Deck (bis deck_size) ausgehend von einer Seed-Liste.
    Dabei werden Karten bevorzugt, die nicht bereits im Main Deck enthalten sind.
    """
    if main_deck_ids is None:
        main_deck_ids = []
    deck = list(seed_ids)
    while len(deck) < deck_size:
        candidates = suggest_cards_for_deck(client, collection_name, deck, candidate_limit=candidate_limit)
        # Filtere Kandidaten, die bereits im Main Deck sind
        candidates = [(cid, score) for cid, score in candidates if cid not in main_deck_ids]
        if not candidates:
            print("Keine weiteren Kandidaten für Side Deck gefunden. Abbruch.")
            break
        candidate_ids = [cid for cid, score in candidates]
        re_ranked = re_rank_with_model(deck, candidate_ids, model, usage_stats, combos, embeddings_dict, card_infos)
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
    """
    Erzeugt einen YDK-String mit Main, Extra und Side Deck.
    """
    lines = []
    lines.append("#created by Qdrant-Deck-Builder")
    lines.append("#main")
    for c_id in main_deck_cards:
        lines.append(str(c_id))
    lines.append("#extra")
    for c_id in extra_deck_cards:
        lines.append(str(c_id))
    lines.append("!side")
    for c_id in side_deck_cards:
        lines.append(str(c_id))
    return "\n".join(lines)

# --- MAIN-Prozess mit interaktiver Auswahl ---

def main():
    # DB-Setup
    engine = create_engine(f"sqlite:///../{SQLITE_DB}")
    Session = sessionmaker(bind=engine)
    session = Session()

    # (Optional) Tabellen erstellen, falls noch nicht vorhanden:
    # Base.metadata.create_all(engine)

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

    # Qdrant-Client (In-Memory, HTTP)
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

    # Embeddings in Qdrant einfügen
    add_embeddings_to_qdrant(client, embeddings_dict, COLLECTION_NAME)

    # Re-Ranking Modell laden
    try:
        with open(MODEL_PICKLE, "rb") as f:
            model = pickle.load(f)
        print("Re-Ranking Modell erfolgreich geladen.")
    except Exception as e:
        print("Fehler beim Laden des Re-Ranking Modells:", e)
        return

    # Zusätzliche Daten für das Re-Ranking laden:
    card_infos = load_all_card_info(f"sqlite:///{SQLITE_DB}")

    # Für usage_stats laden wir alle Decks aus der DB
    deck_list = session.query(Deck).all()
    usage_stats = load_usage_stats(session, deck_list)

    # Frequent Combos laden
    combos = load_combos()

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

            candidates = query_similar_cards(client, COLLECTION_NAME, card_id, candidate_limit=CANDIDATE_LIMIT)
            if candidates:
                candidate_ids = [cid for cid, score in candidates]
                deck_context = [card_id]
                re_ranked = re_rank_with_model(
                    deck_context, candidate_ids, model,
                    usage_stats, combos, embeddings_dict, card_infos
                )
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

            candidates = suggest_cards_for_deck(client, COLLECTION_NAME, deck_ids, candidate_limit=CANDIDATE_LIMIT)
            if candidates:
                candidate_ids = [cid for cid, score in candidates]
                deck_context = deck_ids
                re_ranked = re_rank_with_model(
                    deck_context, candidate_ids, model,
                    usage_stats, combos, embeddings_dict, card_infos
                )
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
            # Seed-Eingabe für Main, Extra und Side Deck
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
            final_main_deck, final_extra_deck = build_main_and_extra_decks_from_seed(
                client, COLLECTION_NAME, seed_ids_main, seed_ids_extra,
                model, usage_stats, combos, embeddings_dict, card_infos,
                desired_main_size=DESIRED_MAIN_SIZE, desired_extra_size=DESIRED_EXTRA_SIZE
            )
            print(f"Main Deck fertig mit {len(final_main_deck)} Karten.")
            print(f"Extra Deck fertig mit {len(final_extra_deck)} Karten.")

            print("Starte Side Deck Building...")
            final_side_deck = build_side_deck_from_seed(
                client, COLLECTION_NAME, seed_ids_side,
                model, usage_stats, combos, embeddings_dict, card_infos,
                deck_size=DESIRED_SIDE_SIZE, main_deck_ids=final_main_deck
            )
            print(f"Side Deck fertig mit {len(final_side_deck)} Karten.")

            final_deck = {
                "main": final_main_deck,
                "extra": final_extra_deck,
                "side": final_side_deck
            }
            ydk_content = build_ydk_string(final_deck["main"], final_deck["extra"], final_deck["side"])
            print("\n===== .ydk-Inhalt =====\n")
            print(ydk_content)
            print("\n===== Ende der .ydk-Datei =====\n")
            with open("my_deck.ydk", "w", encoding="utf-8") as f:
                f.write(ydk_content)
            print("Deck in my_deck.ydk gespeichert.")

        else:
            print("Ungültige Eingabe. Bitte 1, 2, 3 oder 4 eingeben.")

if __name__ == "__main__":
    main()
