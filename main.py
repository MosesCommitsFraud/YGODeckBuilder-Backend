import os
import glob
import pandas as pd
from collections import Counter
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def parse_ydk_file(filepath):
    """
    Parse a .ydk file and return a dictionary with keys:
      - "main": list of card IDs in the main deck.
      - "extra": list of card IDs in the extra deck.
      - "side": list of card IDs in the side deck.

    Enforces that in the main deck each card appears at most 3 times.
    """
    main_deck = []
    extra_deck = []
    side_deck = []
    current_section = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Identify section headers.
            if line.startswith("#"):
                if line.lower().startswith("#main"):
                    current_section = "main"
                elif line.lower().startswith("#extra"):
                    current_section = "extra"
                else:
                    current_section = None
                continue
            if line.startswith("!"):
                if line.lower().startswith("!side"):
                    current_section = "side"
                else:
                    current_section = None
                continue

            # Append card to the appropriate section.
            if current_section == "main":
                main_deck.append(line)
            elif current_section == "extra":
                extra_deck.append(line)
            elif current_section == "side":
                side_deck.append(line)

    # Enforce maximum 3 copies per card in the main deck.
    main_deck_trimmed = []
    counts = Counter()
    for card in main_deck:
        if counts[card] < 3:
            main_deck_trimmed.append(card)
            counts[card] += 1
    return {"main": main_deck_trimmed, "extra": extra_deck, "side": side_deck}


def load_decks_from_folder(folder="ydk_download"):
    """
    Load and parse all .ydk files from the specified folder.
    Only decks with a main deck size between 40 and 60 cards (after enforcing the 3-copy rule)
    are retained.
    """
    deck_files = glob.glob(os.path.join(folder, "*.ydk"))
    decks = []
    for filepath in deck_files:
        deck = parse_ydk_file(filepath)
        if 40 <= len(deck["main"]) <= 60:
            decks.append(deck)
    return decks


def prepare_transactions(decks):
    """
    Prepare a list of transactions for association rule mining.
    Each transaction is the set of unique card IDs from a deck’s main deck.
    """
    transactions = []
    for deck in decks:
        transactions.append(list(set(deck["main"])))
    return transactions


def mine_association_rules(transactions, min_support=0.01, min_confidence=0.5, max_len=3):
    """
    Mine association rules from the transaction data using the Apriori algorithm.

    To help with memory:
      - Columns (cards) with support below min_support are filtered out.
      - Candidate itemsets are limited to max_len.
    """
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Filter out rare items.
    support_series = df.mean()
    df = df.loc[:, support_series >= min_support]

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=max_len)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules


def create_main_deck(seed_cards, rules, min_main=40, max_main=60):
    """
    Automatically build a main deck based on the seed_cards and association rules.

    - Starts with the seed cards (each added once).
    - Iteratively adds recommended cards from rules whose antecedents are
      a subset of the current deck until at least min_main cards are reached.
    - Enforces a maximum of 3 copies per card.
    - Truncates to max_main cards if the deck gets too large.
    """
    # Initialize deck counts from seed cards.
    deck_counts = {card: 1 for card in seed_cards}
    current_length = sum(deck_counts.values())

    # Sort rules by confidence and lift (descending).
    sorted_rules = rules.sort_values(by=['confidence', 'lift'], ascending=False)

    updated = True
    # Iteratively add recommended cards until reaching the minimum required.
    while current_length < min_main and updated:
        updated = False
        for idx, row in sorted_rules.iterrows():
            if row['antecedents'].issubset(set(deck_counts.keys())):
                for card in row['consequents']:
                    if deck_counts.get(card, 0) < 3:
                        deck_counts[card] = deck_counts.get(card, 0) + 1
                        current_length += 1
                        updated = True
                        if current_length >= min_main:
                            break
            if current_length >= min_main:
                break
        # If no rules add any new card, exit the loop.
        if not updated:
            break

    # If still under min_main, try adding extra copies of the seed cards.
    while current_length < min_main:
        added_any = False
        for card in seed_cards:
            if deck_counts.get(card, 0) < 3:
                deck_counts[card] = deck_counts.get(card, 0) + 1
                current_length += 1
                added_any = True
                if current_length >= min_main:
                    break
        if not added_any:
            break  # No further additions possible.

    # Convert deck_counts to a list of card IDs.
    main_deck = []
    for card, count in deck_counts.items():
        main_deck.extend([card] * count)

    # Truncate if the deck exceeds max_main cards.
    if len(main_deck) > max_main:
        main_deck = main_deck[:max_main]

    return main_deck


def create_deck(seed_cards):
    """
    Automatically create a full deck (main, extra, side) based on the given seed_cards.
    Currently, the extra and side decks are left empty.
    """
    # Load decks and mine association rules from your dataset.
    decks = load_decks_from_folder("ydk_download")
    transactions = prepare_transactions(decks)
    rules = mine_association_rules(transactions, min_support=0.01, min_confidence=0.5, max_len=3)

    main_deck = create_main_deck(seed_cards, rules)
    extra_deck = []  # You can extend this with your own logic.
    side_deck = []  # You can extend this with your own logic.
    return {"main": main_deck, "extra": extra_deck, "side": side_deck}


def write_ydk_file(deck, filename):
    """
    Write the deck to a .ydk file with the following structure:
      #main
      <card IDs...>
      #extra
      <card IDs...>
      !side
      <card IDs...>
    """
    with open(filename, 'w') as f:
        f.write("#main\n")
        for card in deck["main"]:
            f.write(card + "\n")
        f.write("#extra\n")
        for card in deck["extra"]:
            f.write(card + "\n")
        f.write("!side\n")
        for card in deck["side"]:
            f.write(card + "\n")
    print(f"Deck saved as {filename}")


if __name__ == '__main__':
    # Get seed cards (comma separated) from the user.
    seed_input = input("Enter seed card IDs for deck creation (comma separated): ")
    seed_cards = {s.strip() for s in seed_input.split(",") if s.strip()}

    # Automatically create a deck based on the given seed cards.
    deck = create_deck(seed_cards)

    # Let the user name the deck and create the .ydk file.
    deck_name = input("Enter a name for your deck (without extension): ").strip()
    filename = deck_name + ".ydk"
    write_ydk_file(deck, filename)
