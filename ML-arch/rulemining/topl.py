import os
import json
import itertools
from collections import Counter, defaultdict
import pandas as pd
import warnings
import pickle

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Card

from transformers import pipeline

warnings.filterwarnings("ignore", category=RuntimeWarning)

###################################
# Configuration – IMPORTANT VARIABLES
###################################
SYNERGY_BOOST_FACTOR = 1.5
EXTRA_SYNERGY_BOOST = 1.5
ARCHETYPE_BOOST_FACTOR = 1.5

MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.5
MAX_ITEMSET_LENGTH = 3

TARGET_MIN_MAIN = 40
TARGET_MAX_MAIN = 60
TARGET_EXTRA_SIZE = 15
TARGET_SIDE_SIZE = 15

DROP_THRESHOLD = 0.5
MAIN_DUPLICATE_LIMIT = 3
EXTRA_DUPLICATE_LIMIT = 1

ARCHETYPES_FILE = "archetypes.json"
YDK_FOLDER = "../ydk_download"
TRAINED_MODEL_FILE = "trained_model.pkl"
OUTPUT_DECK_FILE = "generated_deck.ydk"

###################################
# Transformers for NER
###################################
ner_pipe = pipeline("token-classification", model="dslim/bert-base-NER")

###################################
# Card Info Loading & Helpers (DB-Version)
###################################
def load_card_info_from_db(db_url="sqlite:///../data.sqlite"):
    """
    Lädt alle Kartendaten aus der Datenbank und erstellt ein Dictionary.
    Die Keys sind die Card-IDs (als String) und die Werte enthalten alle relevanten
    Informationen, inklusive referenzierter Karten/Archetypen/Races und den Effekt-Flags.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    card_info = {}
    for card in session.query(Card).all():
        ban_status = card.ban_tcg
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
            referenced_cards = json.loads(card.referenced_cards) if card.referenced_cards else []
        except Exception:
            referenced_cards = []
        try:
            referenced_archetypes = json.loads(card.referenced_archetypes) if card.referenced_archetypes else []
        except Exception:
            referenced_archetypes = []
        try:
            referenced_races = json.loads(card.referenced_races) if card.referenced_races else []
        except Exception:
            referenced_races = []

        card_info[str(card.id)] = {
            "id": card.id,
            "name": card.name,
            "type": card.type or "",
            "human_readable_card_type": card.human_readable_card_type,
            "frame_type": card.frame_type,
            "desc": card.desc,
            "race": card.race,
            "archetype": card.archetype,
            "ban_status_num": ban_status_num,
            "is_staple": card.is_staple,
            "atk": card.atk,
            "defense": card.defense,
            "level": card.level,
            "attribute": card.attribute,
            "referenced_cards": referenced_cards,
            "referenced_archetypes": referenced_archetypes,
            "referenced_races": referenced_races,
            "effect_search": card.effect_search,
            "effect_destroy": card.effect_destroy,
            "effect_negate": card.effect_negate,
            "effect_draw": card.effect_draw,
            "effect_special_summon": card.effect_special_summon,
            "effect_banish": card.effect_banish,
            "effect_send_gy": card.effect_send_gy,
            "effect_recover_lp": card.effect_recover_lp,
            "effect_inflict_damage": card.effect_inflict_damage,
            "effect_equip": card.effect_equip,
            "effect_modify_stats": card.effect_modify_stats,
            "effect_protect": card.effect_protect,
            "effect_discard": card.effect_discard,
            "effect_change_position": card.effect_change_position,
            "effect_return": card.effect_return,
            "effect_shuffle": card.effect_shuffle,
            "effect_copy": card.effect_copy,
            "effect_counter": card.effect_counter,
            "effect_token_summon": card.effect_token_summon,
            "effect_deck_manipulation": card.effect_deck_manipulation
        }
    session.close()
    return card_info

###################################
# Einige Helper-Funktionen
###################################
def parse_ydk_file(file_path):
    """Parses a .ydk file into a dict with keys 'main', 'extra', and 'side'."""
    main_deck, extra_deck, side_deck = [], [], []
    current_section = None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "main" in line.lower():
                    current_section = "main"
                elif "extra" in line.lower():
                    current_section = "extra"
                else:
                    current_section = None
            elif line.startswith("!"):
                if "side" in line.lower():
                    current_section = "side"
                else:
                    current_section = None
            elif line.isdigit() and current_section:
                if current_section == "main":
                    main_deck.append(line)
                elif current_section == "extra":
                    extra_deck.append(line)
                elif current_section == "side":
                    side_deck.append(line)
    return {"main": main_deck, "extra": extra_deck, "side": side_deck}


def load_full_decks(ydk_folder=YDK_FOLDER):
    """
    Lädt alle .ydk-Dateien in einem Ordner.
    Nur Decks mit Main-Deck-Größe zwischen 40 und 60 werden behalten.
    """
    decks = []
    for filename in os.listdir(ydk_folder):
        if filename.endswith(".ydk"):
            path = os.path.join(ydk_folder, filename)
            deck = parse_ydk_file(path)
            if 40 <= len(deck["main"]) <= 60:
                decks.append(deck)
    return decks


###########################
# Assoziationsanalyse
###########################
def build_transaction_df(deck_lists):
    """Erzeugt ein One-Hot-encoded DataFrame aus einer Liste von Decks (Listen)."""
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(deck_lists).transform(deck_lists)
    return pd.DataFrame(te_ary, columns=te.columns_)


def mine_rules(df, min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE, max_len=MAX_ITEMSET_LENGTH):
    """Frequent itemsets und Regeln mit Apriori minen."""
    from mlxtend.frequent_patterns import apriori, association_rules
    freq_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=max_len)
    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)
    return freq_itemsets, rules

def recommend_main_deck_contextual(input_cards, main_decks, top_n=10, min_decks=5):
    """
    Kontextbasierte Empfehlungen (association rules) aus Decks, die mind. eine input_card enthalten.
    """
    from collections import defaultdict
    filtered = [deck for deck in main_decks if any(card in deck for card in input_cards)]
    if len(filtered) < min_decks:
        return []
    df_context = build_transaction_df(filtered)
    _, rules_context = mine_rules(df_context)
    recs = defaultdict(list)
    input_set = set(input_cards)
    for _, rule in rules_context.iterrows():
        if input_set.intersection(set(rule['antecedents'])):
            for card in set(rule['consequents']) - input_set:
                recs[card].append(rule['confidence'])
    ranked = [(card, sum(confs) / len(confs)) for card, confs in recs.items()]
    return sorted(ranked, key=lambda x: x[1], reverse=True)[:top_n]

def recommend_main_deck_by_input(input_cards, main_decks, top_n=10):
    """
    Frequency-basierte Empfehlung: Wir zählen, welche Karten in Decks vorkommen, die (mind. eine) input_card enthalten.
    """
    from collections import Counter
    filtered = [deck for deck in main_decks if any(card in deck for card in input_cards)]
    freq = Counter(card for deck in filtered for card in deck if card not in input_cards)
    return freq.most_common(top_n)

def combine_recommendations(assoc, freq):
    """
    Vereint assoziationsbasierte (assoc) und frequency-basierte (freq) Empfehlungen.
    assoc => Liste[(card, conf)]
    freq => Liste[(card, count)]
    """
    combined = {}
    if freq:
        max_freq = max(freq, key=lambda x: x[1])[1]
    else:
        max_freq = 1

    # assoc in dict
    for card, conf in assoc:
        combined[card] = conf

    # freq integrieren
    for card, count in freq:
        norm = count / max_freq
        if card not in combined:
            combined[card] = norm
        else:
            # Mittelwert aus existing conf und freq
            combined[card] = (combined[card] + norm) / 2

    # Sortieren
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)

############################
# NER-/Archetype-Boost
############################
def get_aggregated_entities(card_list, card_info, ner_pipe):
    """
    Extrahiert Named Entities aus den Beschreibungen aller Karten in card_list.
    """
    entities = set()
    for card_id in card_list:
        if card_id in card_info:
            desc = card_info[card_id].get("desc", "")
            if desc:
                results = ner_pipe(desc)
                for res in results:
                    entities.add(res["word"].lower())
    return entities

def boost_with_archetypes(candidate_score, candidate, input_cards, card_info, archetypes, boost_factor=ARCHETYPE_BOOST_FACTOR):
    """
    Erhöht den Score, wenn eine Übereinstimmung beim Archetype in den Input-Karten-Beschreibungen auftritt.
    """
    input_archetypes = set()
    for card_id in input_cards:
        if card_id in card_info:
            name_lower = card_info[card_id]["name"].lower()
            desc_lower = card_info[card_id].get("desc", "").lower()
            for archetype in archetypes:
                arch_l = archetype.lower().strip('"')
                if arch_l in name_lower or arch_l in desc_lower:
                    input_archetypes.add(arch_l)

    if candidate in card_info:
        cand_text = card_info[candidate]["name"].lower() + " " + card_info[candidate].get("desc", "").lower()
        for arch in input_archetypes:
            if arch in cand_text:
                candidate_score *= boost_factor
                break

    return candidate_score

############################
# Haupt-Funktion, um eine Menge von Kandidaten (Top-N) zu scoren
############################
def get_candidate_scores(base_cards, main_decks, card_info, archetypes, ner_pipe, top_n=30):
    """
    Holt top_n Kontexte (association rules), top_n Frequenz, kombiniert sie,
    boostet die Scores und gibt ein Dictionary {card_id: final_score}.
    """
    context_recs = recommend_main_deck_contextual(base_cards, main_decks, top_n=top_n, min_decks=5)
    freq_recs = recommend_main_deck_by_input(base_cards, main_decks, top_n=top_n)
    combined_recs = combine_recommendations(context_recs, freq_recs)

    # NER-Boost
    initial_context_entities = get_aggregated_entities(base_cards, card_info, ner_pipe)

    boosted_scores = {}
    for card, score in combined_recs:
        new_score = score
        # Falls Named Entities in der Card-Beschreibung o. Ä.
        if card in card_info:
            # Schlichtes Beispiel: Falls ENTWEDER der Name der Card
            # in den Entities ist, boosten wir => SYNERGY_BOOST_FACTOR
            if any(entity in card_info[card]["name"].lower() for entity in initial_context_entities):
                new_score *= SYNERGY_BOOST_FACTOR

        # Archetype-Boost
        new_score = boost_with_archetypes(new_score, card, base_cards, card_info, archetypes)
        boosted_scores[card] = new_score

    return boosted_scores

############################
# NEU: Score für EINE Karte berechnen
############################
def get_single_card_score(base_cards, candidate_card_id, main_decks, card_info, archetypes, ner_pipe):
    """
    Errechnet den synergy-basierten Score nur für eine einzelne Kandidaten-Karte.

    Vorgehen (vereinfacht):
    1) Wir holen uns alle "Kandidaten" wie in get_candidate_scores (Top-Listen),
       damit wir die relativen Scores haben.
    2) Prüfen, ob unser candidate_card_id dort auftaucht,
       andernfalls fügen wir ihn mit score=0 in die Liste.
    3) Wenden den gleichen NER-/Archetype-Boost an.
    4) Geben den finalen Score für candidate_card_id zurück.
    """
    top_n = 9999  # hohes Limit, damit wir quasi alle relevanten Karten erwischen
    context_recs = recommend_main_deck_contextual(base_cards, main_decks, top_n=top_n, min_decks=5)
    freq_recs = recommend_main_deck_by_input(base_cards, main_decks, top_n=top_n)
    combined = combine_recommendations(context_recs, freq_recs)

    # dictionary: card -> base_score
    base_dict = dict(combined)

    # Falls candidate nicht in base_dict => Score=0
    if candidate_card_id not in base_dict:
        base_dict[candidate_card_id] = 0.0

    # NER-Boost
    initial_context_entities = get_aggregated_entities(base_cards, card_info, ner_pipe)
    base_score = base_dict[candidate_card_id]

    # Step: Named-Entity Boost
    if candidate_card_id in card_info:
        if any(entity in card_info[candidate_card_id]["name"].lower() for entity in initial_context_entities):
            base_score *= SYNERGY_BOOST_FACTOR

    # Archetype-Boost
    final_score = boost_with_archetypes(base_score, candidate_card_id, base_cards, card_info, archetypes)
    return final_score

############################
# Main
############################
def main():
    # 1) Lade die Kartendaten
    card_info = load_card_info_from_db()
    print("Card Info aus DB geladen. Anzahl Karten:", len(card_info))

    # 2) Archetypen
    with open(ARCHETYPES_FILE, "r", encoding="utf-8") as f:
        archetypes = json.load(f)

    # 3) Lade Decks (z. B. für Assoziationsanalyse)
    print("Loading full decks...")
    full_decks = load_full_decks(YDK_FOLDER)
    print(f"Loaded {len(full_decks)} decks.")
    main_decks = [deck["main"] for deck in full_decks]

    # 4) Frage nach Basis-Karten (Kontext)
    base_input = input("Enter your base card IDs (comma separated): ")
    base_cards = [c.strip() for c in base_input.split(",") if c.strip()]
    if not base_cards:
        print("No base cards provided. Exiting.")
        return

    # 5) Frage nach einer einzelnen Kandidaten-Karte
    candidate_card_id = input("\nEnter a single candidate card ID: ").strip()
    if not candidate_card_id:
        print("No candidate card provided. Exiting.")
        return

    # 6) Score für diese eine Karte berechnen
    synergy_score = get_single_card_score(
        base_cards=base_cards,
        candidate_card_id=candidate_card_id,
        main_decks=main_decks,
        card_info=card_info,
        archetypes=archetypes,
        ner_pipe=ner_pipe
    )

    # 7) Ausgabe
    card_name = card_info[candidate_card_id]["name"] if candidate_card_id in card_info else "Unknown"
    print(f"\nSynergy-based score for card {candidate_card_id} ({card_name}): {synergy_score:.2f}")


if __name__ == "__main__":
    main()
