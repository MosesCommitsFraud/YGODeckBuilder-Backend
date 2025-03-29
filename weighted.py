import math
from collections import defaultdict
import pandas as pd
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mlxtend.frequent_patterns import apriori

# Dein Datenbankmodell
from db.models import Base, Deck, DeckCard


def load_decks_tf(db_url="sqlite:///mydb.db"):
    """
    Lädt alle Decks aus der DB und gibt zurück:
      - deck_ids: Liste aller Deck-IDs
      - deck_card_counts: dict[deck_id, dict[card_id, int]]
        -> Anzahl Kopien je Karte im Deck
    """
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    deck_card_counts = {}
    deck_ids = []

    all_decks = session.query(Deck).all()
    print(f"Anzahl gefundener Decks: {len(all_decks)}")

    for deck in all_decks:
        deck_ids.append(deck.id)
        local_counts = defaultdict(int)
        for dc in deck.deck_cards:
            # Falls die DB pro Kopie mehrere Einträge speichert,
            # addieren wir hier 1 pro Eintrag:
            local_counts[dc.card_id] += 1

        deck_card_counts[deck.id] = dict(local_counts)

    session.close()
    return deck_ids, deck_card_counts


def compute_idf(deck_ids, deck_card_counts):
    """
    Berechnet IDF pro Karte:
    idf(card) = ln((N+1)/(df(card)+1))
    wo df(card) = Anzahl Decks, die card enthalten.
    """
    N = len(deck_ids)
    df_counter = defaultdict(int)

    for d_id in deck_ids:
        card_map = deck_card_counts[d_id]
        # Jede Karte -> einmal pro Deck zählen
        for c_id in card_map.keys():
            df_counter[c_id] += 1

    idf = {}
    for c_id, df_val in df_counter.items():
        idf_val = math.log((N + 1) / (df_val + 1))
        idf[c_id] = idf_val

    return idf


def build_tfidf_itemsets(deck_ids, deck_card_counts, idf, threshold=0.1):
    """
    Erstellt eine Liste von Sets, wobei pro Deck nur Karten enthalten sind,
    deren TF–IDF > threshold. TF(d,c) = Anzahl Kopien, IDF(c) = idf[c].
    """
    all_deck_sets = []

    for d_id in deck_ids:
        card_map = deck_card_counts[d_id]
        deck_set = set()

        for c_id, copies in card_map.items():
            if c_id not in idf:
                continue
            tf_val = float(copies)
            tfidf_val = tf_val * idf[c_id]
            if tfidf_val > threshold:
                # Karte in dieses Deck-Set übernehmen
                deck_set.add(str(c_id))  # als String

        if deck_set:
            all_deck_sets.append(deck_set)

    return all_deck_sets


def create_onehot_dataframe(deck_itemsets):
    """
    Nimmt eine Liste[Set[str]] (jede Zeile = ein Deck)
    und baut ein DataFrame für Apriori (binär: 1/0).
    """
    all_cards = set()
    for s in deck_itemsets:
        all_cards.update(s)
    all_cards = sorted(all_cards)

    data_matrix = []
    for deck_set in deck_itemsets:
        row = [1 if c in deck_set else 0 for c in all_cards]
        data_matrix.append(row)

    df = pd.DataFrame(data_matrix, columns=all_cards, dtype=bool)
    return df


def main():
    # 1) Lade Decks & Kopien
    deck_ids, deck_card_counts = load_decks_tf("sqlite:///data.sqlite")
    if not deck_ids:
        print("Keine Decks gefunden, Abbruch.")
        return

    # 2) IDF berechnen
    idf_map = compute_idf(deck_ids, deck_card_counts)

    # 3) TF-IDF => binäres Itemset
    threshold = 0.1
    deck_itemsets = build_tfidf_itemsets(deck_ids, deck_card_counts, idf_map, threshold=threshold)
    print(f"Itemsets nach TF–IDF-Filter: {len(deck_itemsets)} Decks mit >=1 Karte")

    # 4) One-Hot Encoding
    df_bin = create_onehot_dataframe(deck_itemsets)
    print(f"Form des DataFrames: {df_bin.shape}")

    # 5) Apriori
    min_support = 0.02
    freq_itemsets = apriori(
        df_bin,
        min_support=min_support,
        use_colnames=True,
        max_len=5
    )

    freq_itemsets["length"] = freq_itemsets["itemsets"].apply(len)
    freq_itemsets.sort_values("support", ascending=False, inplace=True)

    print("Erste gefundene Itemsets:")
    print(freq_itemsets.head(20))

    # 6) Filtern (z.B. nur Itemsets mit >= 2 Karten)
    filtered = freq_itemsets[freq_itemsets["length"] >= 2]
    print(f"\nGefilterte Itemsets (mind. 2 Karten): {filtered.shape[0]} Stück")
    print(filtered.head(20))

    # 7) JSON-Export vorbereiten
    #    a) frozenset -> list
    filtered["itemsets"] = filtered["itemsets"].apply(lambda fs: list(fs))
    #    b) DataFrame -> list of dict
    filtered_dict = filtered.to_dict(orient="records")

    # 8) JSON-Dump
    with open("frequent_combos_weighted.json", "w") as f:
        json.dump(filtered_dict, f, indent=2)

    print("\nfrequent_combos_weighted.json wurde erfolgreich erzeugt.")


if __name__ == "__main__":
    main()
