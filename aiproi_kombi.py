# aiproi_kombi.py

import math
from collections import defaultdict
import pandas as pd
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# mlxtend
from mlxtend.frequent_patterns import apriori

# Deine Modelle
from db.models import Base, Deck, DeckCard


def load_decks_tf(db_url="sqlite:///mydb.db"):
    """
    Lädt alle Decks aus der DB und gibt zurück:
      - deck_list: Liste aller Deck-IDs
      - deck_card_counts: Dict[deck_id, Dict[card_id, count]] (Anzahl Kopien pro Karte in diesem Deck)
    """
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    deck_card_counts = {}
    deck_list = []

    all_decks = session.query(Deck).all()
    print(f"Anzahl gefundener Decks: {len(all_decks)}")

    for deck in all_decks:
        deck_list.append(deck.id)
        local_counts = defaultdict(int)
        for dc in deck.deck_cards:
            # Falls die DB wirklich pro Kopie mehrere Einträge hat,
            # addieren wir 1 pro Eintrag:
            local_counts[dc.card_id] += 1
        deck_card_counts[deck.id] = dict(local_counts)

    session.close()
    return deck_list, deck_card_counts


def compute_idf(deck_list, deck_card_counts):
    """
    Berechnet IDF pro Karte, basierend darauf,
    in wie vielen Decks die Karte vorkommt.
    idf(card) = ln( (N+1) / (df(card)+1) ).
    """
    N = len(deck_list)
    df_counter = defaultdict(int)

    # df(card) = Anzahl der Decks, in denen card vorkommt
    for deck_id in deck_list:
        card_map = deck_card_counts[deck_id]
        for c_id in card_map.keys():
            df_counter[c_id] += 1

    # IDF
    idf = {}
    for c_id, df_val in df_counter.items():
        idf_val = math.log((N + 1) / (df_val + 1))
        idf[c_id] = idf_val

    return idf


def build_tfidf_decks(deck_list, deck_card_counts, idf, threshold=0.1):
    """
    Erzeugt eine Liste von Sets,
    wobei pro Deck nur Karten enthalten sind,
    deren tfidf(deck, card) > threshold.
    tf(deck, card) = Kopienanzahl
    idf(card)
    """
    tfidf_decks = []

    for deck_id in deck_list:
        card_count_map = deck_card_counts[deck_id]
        deck_set = set()

        for c_id, copies in card_count_map.items():
            if c_id not in idf:
                continue
            tf_val = float(copies)           # z.B. 1..3
            tfidf_val = tf_val * idf[c_id]  # tf * idf
            if tfidf_val > threshold:
                # In binärer Darstellung: 1 -> wir nehmen die Karte
                deck_set.add(str(c_id))  # Konvertieren in String

        if deck_set:
            tfidf_decks.append(deck_set)

    return tfidf_decks


def create_onehot_dataframe(decks_as_itemsets):
    """
    Nimmt eine Liste[Set[str]] (jede Listezeile = ein Deck)
    und erzeugt ein DataFrame mit binären Spalten (1/0).
    """
    all_ids = set()
    for deckset in decks_as_itemsets:
        all_ids.update(deckset)

    all_ids = sorted(all_ids)
    data_matrix = []
    for deckset in decks_as_itemsets:
        row = [1 if card in deckset else 0 for card in all_ids]
        data_matrix.append(row)

    df = pd.DataFrame(data_matrix, columns=all_ids, dtype=bool)
    return df


def run_apriori_tfidf_decks(tfidf_decks, min_support=0.02, max_len=5):
    """
    Erstellt das One-Hot-DataFrame und führt Apriori aus.
    """
    df_onehot = create_onehot_dataframe(tfidf_decks)
    freq_itemsets = apriori(
        df_onehot,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len
    )
    freq_itemsets.sort_values("support", ascending=False, inplace=True)
    return freq_itemsets


def main():
    # ------------------
    # 1) Decks + Counts
    # ------------------
    deck_list, deck_card_counts = load_decks_tf("sqlite:///data.sqlite")
    print(f"Gefundene Decks: {len(deck_list)}")
    if not deck_list:
        print("Keine Decks gefunden, Abbruch.")
        return

    # -------------
    # 2) IDF
    # -------------
    idf_map = compute_idf(deck_list, deck_card_counts)
    print(f"Anzahl unterschiedlicher Karten: {len(idf_map)}")

    # ---------------------------
    # 3) TF–IDF => binär filtern
    # ---------------------------
    threshold = 0.1  # anpassen
    tfidf_decks = build_tfidf_decks(deck_list, deck_card_counts, idf_map, threshold=threshold)
    print(f"Decks nach TF–IDF-Filter: {len(tfidf_decks)}")

    # ------------
    # 4) Apriori
    # ------------
    min_support = 0.02
    freq_itemsets = run_apriori_tfidf_decks(tfidf_decks, min_support=min_support, max_len=5)
    freq_itemsets["length"] = freq_itemsets["itemsets"].apply(lambda x: len(x))

    # ------------------------------
    # 5) Nur Itemsets >= 2 Karten
    # ------------------------------
    final_itemsets = freq_itemsets[freq_itemsets["length"] >= 2]
    print(final_itemsets.head(20))

    # -----------------------------
    # 6) In JSON serialisierbares Format
    # -----------------------------
    # frozenset -> list
    final_itemsets["itemsets"] = final_itemsets["itemsets"].apply(lambda fs: list(fs))

    # DataFrame -> list of dict
    final_itemsets_dict = final_itemsets.to_dict(orient="records")

    # -----------------------------
    # 7) JSON speichern
    # -----------------------------
    with open("frequent_combos.json", "w") as f:
        json.dump(final_itemsets_dict, f, indent=2)

    print("Frequente Kombos wurden in 'frequent_combos.json' gespeichert.")


if __name__ == "__main__":
    main()
