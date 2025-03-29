import pickle
import random
import math
import json
import numpy as np
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# DB-Modelle
from db.models import Base, Deck, DeckCard, Card

# ----------------- Embeddings laden -----------------
EMBEDDINGS_PICKLE = "graph_embeddings.pkl"  # Pfad zur Pickle-Datei mit den Embeddings

try:
    with open(EMBEDDINGS_PICKLE, "rb") as f:
        raw_embeddings = pickle.load(f)
    # Konvertiere alle Embeddings in NumPy-Arrays, um spätere Umwandlungen zu vermeiden
    embeddings_dict = {k: np.array(v) for k, v in raw_embeddings.items()}
    print(f"Es wurden {len(embeddings_dict)} Embeddings aus '{EMBEDDINGS_PICKLE}' geladen.")
except Exception as e:
    print(f"Fehler beim Laden der Embeddings: {e}")
    embeddings_dict = {}

def get_embedding_score(deck_cards, c_id):
    """
    Berechnet den Embedding Score als Cosinus-Ähnlichkeit zwischen dem
    durchschnittlichen Deck-Embedding und dem Embedding der Kandidatenkarte.
    """
    if c_id not in embeddings_dict:
        return 0.0
    candidate_vector = embeddings_dict[c_id]
    deck_vectors = [embeddings_dict[x] for x in deck_cards if x in embeddings_dict]
    if not deck_vectors:
        return 0.0
    deck_avg = np.mean(deck_vectors, axis=0)
    dot = np.dot(candidate_vector, deck_avg)
    norm_candidate = np.linalg.norm(candidate_vector)
    norm_deck = np.linalg.norm(deck_avg)
    if norm_candidate == 0 or norm_deck == 0:
        return 0.0
    return float(dot / (norm_candidate * norm_deck))

# ---------------------------------------------------------
# 1) Daten aus DB laden
# ---------------------------------------------------------
def load_decks_and_cards_from_db(db_url="sqlite:///mydb.db"):
    """
    Lädt alle Decks aus der DB.
    Gibt zurück:
      - deck_list: Liste von Deck-Objekten
      - all_cards: Set aller card_ids, die in den Decks vorkommen.
    """
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    deck_list = session.query(Deck).all()
    all_cards = set()
    for deck in deck_list:
        for dc in deck.deck_cards:
            all_cards.add(dc.card_id)

    session.close()
    return deck_list, all_cards

# ---------------------------------------------------------
# 2) Alle Karteninformationen einmalig laden
# ---------------------------------------------------------
def load_all_card_info(db_url="sqlite:///mydb.db"):
    """
    Lädt alle relevanten Informationen aller Karten aus der DB und speichert sie in einem Dictionary.
    """
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
    """Gibt Standardwerte zurück, falls keine Karteninformation gefunden wurde."""
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

# ---------------------------------------------------------
# 3) Hilfsfunktionen (Feature-Building etc.)
# ---------------------------------------------------------
def build_feature_vector(deck_cards, c_id, usage_stats, combos, get_embedding_score, card_infos):
    """
    Erzeugt einen Feature-Vektor für die (Deck, Candidate)-Kombination.
    """
    deck_set = set(deck_cards)
    embedding_score = get_embedding_score(deck_cards, c_id)
    usage_val = usage_stats.get(c_id, 0.0)

    # Synergy: Zähle, wie viele frequent combos (mit Aufnahme der Candidate) erfüllt sind.
    synergy_count = 0
    new_deck = deck_set.union({c_id})
    for combo_dict in combos:
        combo_items = combo_dict["itemsets"]
        combo_set = set(map(int, combo_items))
        if combo_set.issubset(new_deck):
            synergy_count += 1
    synergy_norm = synergy_count / len(deck_cards) if deck_cards else 0

    # Deck-Komposition und Sammlung von Archetypen/Races
    spell_count = 0
    trap_count = 0
    monster_count = 0
    deck_archetypes = []
    deck_races = []
    for dcard in deck_cards:
        info = card_infos.get(dcard, default_card_info(dcard))
        ctype = info.get("type", "")
        if "Spell" in ctype:
            spell_count += 1
        elif "Trap" in ctype:
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

    # Informationen zur Candidate-Karte
    cand_info = card_infos.get(c_id, default_card_info(c_id))
    ban_stat = cand_info.get("ban_status_num", 3)
    cand_arche = cand_info.get("archetype")
    cand_race = cand_info.get("race")
    same_arch_count = sum(1 for a in deck_archetypes if a == cand_arche) if cand_arche else 0
    same_race_count = sum(1 for r in deck_races if r == cand_race) if cand_race else 0

    # Normalisierte referenzierte Karten
    referenced_cards = cand_info.get("referenced_cards", [])
    num_ref_cards = len(referenced_cards)
    count_ref_cards_in_deck = 0
    for ref in referenced_cards:
        try:
            ref_int = int(ref)
        except ValueError:
            continue
        if ref_int in deck_set:
            count_ref_cards_in_deck += 1
    norm_ref_cards = count_ref_cards_in_deck / num_ref_cards if num_ref_cards > 0 else 0

    # Normalisierte referenzierte Archetypen
    referenced_archetypes = cand_info.get("referenced_archetypes", [])
    num_ref_arch = len(referenced_archetypes)
    count_ref_arch_in_deck = sum(1 for ref in referenced_archetypes if ref in deck_archetypes)
    norm_ref_arch = count_ref_arch_in_deck / num_ref_arch if num_ref_arch > 0 else 0

    # Normalisierte referenzierte Races
    referenced_races = cand_info.get("referenced_races", [])
    num_ref_races = len(referenced_races)
    count_ref_races_in_deck = sum(1 for ref in referenced_races if ref in deck_races)
    norm_ref_races = count_ref_races_in_deck / num_ref_races if num_ref_races > 0 else 0


    # One-Hot Encoded Effekt-Features (20 Features)
    effect_features = cand_info.get("effect_features", [0] * 20)

    # Zusammenbau des Feature-Vektors (insgesamt 34 Features)
    feat_vec = [
        embedding_score,      # 0
        usage_val,            # 1
        synergy_norm,         # 2
        spell_pct,            # 3
        trap_pct,             # 4
        monster_pct,          # 5
        ban_stat,             # 6
        same_arch_count,      # 7
        same_race_count,      # 8
        norm_ref_cards,       # 9
        norm_ref_arch,        # 10
        norm_ref_races,       # 11
        1 if cand_info.get("is_staple") else 0  # 12
    ]
    feat_vec.extend(effect_features)  # Features 13-32
        # Feature 33

    return feat_vec

def build_training_data(deck_list, all_cards, usage_stats, combos, get_embedding_score, card_infos, negative_ratio=1):
    """
    Erzeugt (X, y)-Trainingsdaten:
      - Positive Beispiele: Karte c_id ist im Deck (Label=1)
      - Negative Beispiele: Zufällig gewählte Karten, die nicht im Deck sind (Label=0)
    """
    X = []
    y = []

    for deck in deck_list:
        deck_cards = [dc.card_id for dc in deck.deck_cards]
        deck_card_set = set(deck_cards)

        # Positive Beispiele
        for c_id in deck_card_set:
            fv = build_feature_vector(deck_cards, c_id, usage_stats, combos, get_embedding_score, card_infos)
            X.append(fv)
            y.append(1)

        # Negative Beispiele
        not_in_deck = list(all_cards - deck_card_set)
        sample_size = min(len(not_in_deck), negative_ratio * len(deck_cards))
        if sample_size > 0:
            neg_cards = random.sample(not_in_deck, sample_size)
            for c_id in neg_cards:
                fv = build_feature_vector(deck_cards, c_id, usage_stats, combos, get_embedding_score, card_infos)
                X.append(fv)
                y.append(0)

    return np.array(X), np.array(y)

def train_random_forest(X, y):
    """
    Trainiert einen RandomForestClassifier, evaluiert ihn und speichert das Modell.
    """
    if len(X) == 0:
        print("Keine Trainingsdaten (X ist leer). Abbruch.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if len(X_train) == 0 or len(X_test) == 0:
        print("Zu wenige Samples für Train/Test-Split. Abbruch.")
        return None

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # Nutzt alle verfügbaren CPU-Kerne
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, (y_pred_proba > 0.5).astype(int))
    print(f"RandomForest -> Test AUC = {auc:.4f}, Accuracy = {acc:.4f}")

    # Modell speichern
    with open("my_random_forest.pkl", "wb") as f:
        pickle.dump(model, f)

    return model

def re_rank_with_model(deck_cards, candidate_ids, model, usage_stats, combos, get_embedding_score, card_infos):
    """
    Wendet das trainierte RandomForest-Modell an und sortiert die Kandidaten absteigend nach der
    Wahrscheinlichkeit, dass die Karte im Deck sein sollte.
    """
    if model is None:
        print("Kein Modell vorhanden. Abbruch.")
        return []

    results = []
    for c_id in candidate_ids:
        fv = build_feature_vector(deck_cards, c_id, usage_stats, combos, get_embedding_score, card_infos)
        score = model.predict_proba([fv])[0][1]
        results.append((c_id, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ---------------------------------------------------------
# 4) MAIN
# ---------------------------------------------------------
def main():
    db_url = "sqlite:///data.sqlite"  # Passe den Pfad zur DB an
    deck_list, all_cards = load_decks_and_cards_from_db(db_url)
    print(f"Decks geladen: {len(deck_list)}")
    print(f"Insgesamt Karten-IDs (in Decks): {len(all_cards)}")

    # Alle Karteninformationen einmalig laden
    card_infos = load_all_card_info(db_url)

    # Berechne usage_stats: relative Häufigkeit, in wie vielen Decks eine Karte vorkommt
    usage_stats = defaultdict(int)
    for deck in deck_list:
        unique_deck_cards = set(dc.card_id for dc in deck.deck_cards)
        for c_id in unique_deck_cards:
            usage_stats[c_id] += 1
    for c_id in usage_stats:
        usage_stats[c_id] = usage_stats[c_id] / len(deck_list)

    # Combos laden (frequent_combos.json), falls vorhanden
    try:
        with open("frequent_combos.json", "r") as f:
            combos = json.load(f)
    except FileNotFoundError:
        combos = []
        print("Keine frequent_combos.json gefunden. combos bleibt leer.")

    # Erstelle Trainingsdaten
    X, y = build_training_data(
        deck_list,
        all_cards,
        usage_stats,
        combos,
        get_embedding_score,
        card_infos,
        negative_ratio=1
    )
    print(f"Trainingssamples: {X.shape[0]}")
    if len(X) > 0:
        print(f"Positive = {sum(y)}, Negative = {len(y) - sum(y)}")

    # Trainiere das Modell
    model = train_random_forest(X, y)
    if not model:
        return

    # Beispiel: Re-Ranking für das erste Deck aus deck_list
    if deck_list:
        example_deck = deck_list[0]
        example_deck_cards = [dc.card_id for dc in example_deck.deck_cards]

        # Zufällige 10 Karten aus all_cards als Kandidaten
        candidate_ids = random.sample(all_cards, min(10, len(all_cards)))
        results = re_rank_with_model(example_deck_cards, candidate_ids, model,
                                     usage_stats, combos,
                                     get_embedding_score, card_infos)
        print("\nRe-Ranked Candidates:")
        for (cid, sc) in results:
            print(f"Card {cid} -> Probability {sc:.3f}")

if __name__ == "__main__":
    main()
