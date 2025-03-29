import math
from collections import defaultdict
import pandas as pd
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mlxtend.frequent_patterns import apriori

# Importiere deine DB-Modelle; wichtig: Card muss das Feld is_staple enthalten
from db.models import Base, Deck, DeckCard, Card


def load_staple_cards(db_url="sqlite:///data.sqlite"):
    """
    Lädt alle Card-IDs, bei denen is_staple=True.
    Gibt ein Set mit diesen Card-IDs zurück.
    """
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    staple_set = set()
    # Wir nehmen an, das Feld heißt is_staple (Boolean)
    cards_staple = session.query(Card).filter_by(is_staple=True).all()
    for c in cards_staple:
        staple_set.add(c.id)
    session.close()
    return staple_set


def load_decks_tf(db_url="sqlite:///data.sqlite"):
    """
    Lädt alle Decks und erzeugt:
      - deck_ids: Liste aller Deck-IDs
      - deck_card_counts: dict[deck_id, dict[card_id, int]]
          -> Anzahl Kopien je Karte im Deck (Staple-Karten werden ausgeschlossen).
      - deck_size: dict[deck_id, int] -> Summe Kopien pro Deck
    """
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Staple-Karten ermitteln:
    staple_set = load_staple_cards(db_url)

    deck_ids = []
    deck_card_counts = {}
    deck_size = {}

    all_decks = session.query(Deck).all()
    for deck in all_decks:
        deck_ids.append(deck.id)
        local_counts = defaultdict(int)

        for dc in deck.deck_cards:
            # Wenn die Karte als Staple markiert ist, überspringen
            if dc.card_id in staple_set:
                continue
            local_counts[dc.card_id] += 1

        deck_card_counts[deck.id] = dict(local_counts)
        total_copies = sum(local_counts.values())
        deck_size[deck.id] = total_copies

    session.close()
    print(f"Anzahl gefundener Decks: {len(all_decks)}")
    return deck_ids, deck_card_counts, deck_size


def compute_df(deck_ids, deck_card_counts):
    """
    Berechnet df_counter[c_id] = in wie vielen Decks Karte c_id vorkommt.
    """
    df_counter = defaultdict(int)
    for d_id in deck_ids:
        for c_id in deck_card_counts[d_id].keys():
            df_counter[c_id] += 1
    return df_counter


def compute_bm25_scores(deck_ids, deck_card_counts, deck_size, k=1.2, b=0.75):
    """
    Berechnet BM25-Score pro (Deck, Card) und gibt bm25_map[deck_id][c_id] zurück.
    """
    df_counter = compute_df(deck_ids, deck_card_counts)
    N = len(deck_ids)
    avg_deck_size = sum(deck_size[d_id] for d_id in deck_ids) / float(N)

    def bm25_idf(df_val):
        return math.log((N - df_val + 0.5) / (df_val + 0.5) + 1)

    idf_map = {}
    for c_id, df_val in df_counter.items():
        idf_map[c_id] = bm25_idf(df_val)

    bm25_map = {}
    for d_id in deck_ids:
        bm25_map[d_id] = {}
        d_len = deck_size[d_id]
        for c_id, tf_val in deck_card_counts[d_id].items():
            idf_c = idf_map[c_id]
            numerator = tf_val * (k + 1)
            denominator = tf_val + k * (1 - b + b * (d_len / avg_deck_size))
            score = idf_c * (numerator / denominator)
            bm25_map[d_id][c_id] = score
    return bm25_map


def build_bm25_itemsets(deck_ids, bm25_map, threshold=0.1):
    """
    Erzeugt eine Liste von Sets (als Strings) für jedes Deck:
    Eine Karte wird aufgenommen, wenn BM25(d,c) > threshold.
    """
    all_deck_sets = []
    for d_id in deck_ids:
        deck_set = set()
        for c_id, score in bm25_map[d_id].items():
            if score > threshold:
                deck_set.add(str(c_id))
        if deck_set:
            all_deck_sets.append(deck_set)
    return all_deck_sets


def create_onehot_dataframe(deck_itemsets):
    """
    Konvertiert eine Liste von Sets (jede Zeile = ein Deck) in ein binäres DataFrame.
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
    # 1) Decks laden (Staple-Karten werden ausgeschlossen)
    deck_ids, deck_card_counts, deck_size = load_decks_tf("sqlite:///data.sqlite")
    if not deck_ids:
        print("Keine Decks gefunden. Abbruch.")
        return

    # 2) BM25 berechnen
    k = 1.2
    b = 0.75
    bm25_map = compute_bm25_scores(deck_ids, deck_card_counts, deck_size, k=k, b=b)

    # 3) Binarisierung der BM25-Scores
    threshold = 0.1  # Experimenteller Wert
    bm25_deck_itemsets = build_bm25_itemsets(deck_ids, bm25_map, threshold=threshold)
    print(f"Anzahl Decks nach BM25-Filter: {len(bm25_deck_itemsets)}")

    # 4) One-Hot-Encoding
    df_bin = create_onehot_dataframe(bm25_deck_itemsets)
    print(f"DataFrame: {df_bin.shape}")

    # 5) Apriori ausführen
    min_support = 0.02
    freq_itemsets = apriori(df_bin, min_support=min_support, use_colnames=True, max_len=5)
    freq_itemsets["length"] = freq_itemsets["itemsets"].apply(len)
    freq_itemsets.sort_values("support", ascending=False, inplace=True)
    print("Erste gefundene Itemsets:")
    print(freq_itemsets.head(20))

    # 6) Filtern (nur Itemsets mit mindestens 2 Karten)
    final_itemsets = freq_itemsets[freq_itemsets["length"] >= 2]
    print(f"Itemsets >=2 Karten: {final_itemsets.shape[0]}")
    print(final_itemsets.head(20))

    # 7) JSON-Export (konvertiere frozenset in Liste)
    final_itemsets["itemsets"] = final_itemsets["itemsets"].apply(lambda fs: list(fs))
    combos_dict = final_itemsets.to_dict(orient="records")
    with open("frequent_combos_bm25.json", "w") as f:
        json.dump(combos_dict, f, indent=2)

    print("BM25-basiertes Apriori beendet. Ergebnisse in 'frequent_combos_bm25.json'.")


if __name__ == "__main__":
    main()
