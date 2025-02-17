import os
import json
import itertools
from collections import Counter, defaultdict
import pandas as pd
import requests
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


##############################
# Part 1: Data Loading & Parsing
##############################

def parse_ydk_file(file_path):
    """
    Parses a .ydk file into a dictionary with keys "main", "extra", and "side".
    """
    main_deck = []
    extra_deck = []
    side_deck = []
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


def load_full_decks(ydk_folder):
    """
    Loads all .ydk files from a folder and returns a list of decks,
    each as a dict with keys "main", "extra", and "side".
    Only decks with a main deck of 40–60 cards are included.
    """
    decks = []
    for filename in os.listdir(ydk_folder):
        if filename.endswith(".ydk"):
            file_path = os.path.join(ydk_folder, filename)
            deck = parse_ydk_file(file_path)
            if 40 <= len(deck["main"]) <= 60:
                decks.append(deck)
    return decks


##############################
# Part 2: Association Rule Mining (Main Deck Only)
##############################

def build_transaction_df(decks):
    """
    Converts a list of main decks (each a list of card IDs) into a one-hot encoded DataFrame.
    """
    te = TransactionEncoder()
    te_ary = te.fit(decks).transform(decks)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df


def mine_rules(df, min_support=0.05, min_confidence=0.5, max_len=3):
    """
    Mines frequent itemsets and association rules using Apriori.
    Returns both the frequent itemsets and the rules DataFrame.
    """
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=max_len)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return frequent_itemsets, rules


def recommend_cards(input_cards, rules, top_n=5):
    """
    Given a list of input card IDs and association rules,
    recommends additional main deck cards.
    If any input card appears in a rule's antecedents, the consequent cards are recommended.
    Returns a list of tuples (card_id, avg_confidence).
    """
    recommendations = defaultdict(list)
    input_set = set(input_cards)
    matched_rule_count = 0

    for idx, rule in rules.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])
        if input_set.intersection(antecedents):
            matched_rule_count += 1
            new_cards = consequents - input_set
            for card in new_cards:
                recommendations[card].append(rule['confidence'])

    print(f"Found {matched_rule_count} association rules matching the input cards.")

    ranked = []
    for card, confs in recommendations.items():
        avg_conf = sum(confs) / len(confs)
        ranked.append((card, avg_conf))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def fallback_recommendations(input_cards, main_decks, top_n=5):
    """
    If no association rules match the input, fall back on co-occurrence frequency.
    Returns a list of tuples (card_id, frequency) from the main decks.
    """
    input_set = set(input_cards)
    co_occur = Counter()
    for deck in main_decks:
        if input_set.intersection(set(deck)):
            for card in deck:
                if card not in input_set:
                    co_occur[card] += 1
    ranked = co_occur.most_common(top_n)
    return ranked


##############################
# Part 3: Deck Construction (Main, Extra, & Side)
##############################

def build_new_deck(input_cards, recommendations, main_decks, extra_counter, side_counter,
                   target_main=40, target_extra=15, target_side=15):
    """
    Constructs a complete deck containing main, extra, and side decks.

    - Main deck: starts with input cards, then adds recommended cards (max 3 copies each),
      and fills remaining slots using the most frequent main deck cards.
    - Extra deck: filled with the most frequent extra deck cards (unique; max 1 copy per card).
    - Side deck: filled with the most frequent side deck cards (max 3 copies each).

    Returns a dictionary with keys "main", "extra", and "side".
    """
    deck_counter = Counter()  # for main deck (max 3 copies per card)
    new_deck = {"main": [], "extra": [], "side": []}

    # Build main deck
    for card in input_cards:
        new_deck["main"].append(card)
        deck_counter[card] += 1

    # Add recommended main deck cards
    for card, _ in recommendations:
        while deck_counter[card] < 3 and len(new_deck["main"]) < target_main:
            new_deck["main"].append(card)
            deck_counter[card] += 1
            if len(new_deck["main"]) >= target_main:
                break

    # If still not enough, fill with most frequent main deck cards from training data
    if len(new_deck["main"]) < target_main:
        all_main_cards = []
        for d in main_decks:
            all_main_cards.extend(d)
        card_freq = Counter(all_main_cards)
        for card, _ in card_freq.most_common():
            while deck_counter[card] < 3 and len(new_deck["main"]) < target_main:
                new_deck["main"].append(card)
                deck_counter[card] += 1
            if len(new_deck["main"]) >= target_main:
                break

    # Build extra deck (use frequency counts; only 1 copy per card)
    extra_deck = []
    for card, _ in extra_counter.most_common():
        if len(extra_deck) < target_extra:
            if card not in new_deck["main"] and card not in extra_deck:
                extra_deck.append(card)
        else:
            break
    new_deck["extra"] = extra_deck

    # Build side deck (allow up to 3 copies per card)
    side_deck = []
    side_deck_counter = Counter()
    for card, _ in side_counter.most_common():
        while side_deck_counter[card] < 3 and len(side_deck) < target_side:
            # Avoid duplicating cards already in main or extra decks
            if card not in new_deck["main"] and card not in new_deck["extra"]:
                side_deck.append(card)
                side_deck_counter[card] += 1
            if len(side_deck) >= target_side:
                break
    new_deck["side"] = side_deck

    return new_deck


def write_ydk(deck, filename):
    """
    Writes the complete deck (with main, extra, and side sections) to a .ydk file.
    """
    with open(filename, "w") as f:
        f.write("#main\n")
        for card in deck["main"]:
            f.write(card + "\n")
        f.write("#extra\n")
        for card in deck["extra"]:
            f.write(card + "\n")
        f.write("!side\n")
        for card in deck["side"]:
            f.write(card + "\n")
    print(f"Deck saved to {filename}")


##############################
# Part 4: Helper for JSON Serialization
##############################

def convert_frozensets(obj):
    """
    Recursively converts frozenset objects to lists for JSON serialization.
    """
    if isinstance(obj, frozenset):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_frozensets(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_frozensets(item) for item in obj]
    else:
        return obj


##############################
# Part 5: Main Function
##############################

def main():
    # ---- Settings ----
    # Folder containing your .ydk files (update this path as needed)
    ydk_folder = "ydk_download"
    # Association rule mining parameters (for main deck)
    min_support = 0.05
    min_confidence = 0.5
    max_itemset_length = 3
    # Target deck sizes
    target_main_size = 40
    target_extra_size = 15
    target_side_size = 15
    # Output file names
    trained_model_file = "trained_model.json"
    new_deck_file = "generated_deck.ydk"

    # ---- Load Full Decks ----
    print("Loading full decks (main, extra, side)...")
    full_decks = load_full_decks(ydk_folder)
    print(f"Loaded {len(full_decks)} decks.")

    # Separate main decks for association rule mining (using only the main deck cards)
    main_decks = [deck["main"] for deck in full_decks]

    # Compute frequency counters for extra and side decks from full decks
    extra_counter = Counter()
    side_counter = Counter()
    for deck in full_decks:
        for card in deck["extra"]:
            extra_counter[card] += 1
        for card in deck["side"]:
            side_counter[card] += 1

    # ---- Build Transaction DataFrame & Mine Rules (Main Deck) ----
    print("Building transaction DataFrame for main decks...")
    df = build_transaction_df(main_decks)
    print("Mining association rules on main decks...")
    frequent_itemsets, rules = mine_rules(df, min_support=min_support, min_confidence=min_confidence,
                                          max_len=max_itemset_length)
    print(f"Mined {len(rules)} association rules.")

    # ---- Output Trained Model ----
    freq_itemsets_dict = convert_frozensets(frequent_itemsets.to_dict(orient="records"))
    assoc_rules_dict = convert_frozensets(rules.to_dict(orient="records"))
    model = {
        "frequent_itemsets": freq_itemsets_dict,
        "association_rules": assoc_rules_dict
    }
    with open(trained_model_file, "w") as f:
        json.dump(model, f, indent=4)
    print(f"Trained model saved to {trained_model_file}")

    # ---- Get User Input for Base Cards ----
    user_input = input("Enter base card IDs (separated by commas): ")
    input_cards = [card.strip() for card in user_input.split(",") if card.strip()]
    if not input_cards:
        print("No input cards provided. Exiting.")
        return

    # ---- Get Recommendations for Main Deck ----
    recs = recommend_cards(input_cards, rules, top_n=10)
    if not recs:
        print("No recommendations found using association rules. Falling back to co-occurrence counts.")
        recs = fallback_recommendations(input_cards, main_decks, top_n=10)
        # recs from fallback are (card, frequency) tuples.
    else:
        print("Recommendations from association rules for main deck:")
        for card, conf in recs:
            print(f"Card {card} with avg confidence {conf:.2f}")

    # ---- Build New Complete Deck (Main, Extra, Side) ----
    new_deck = build_new_deck(input_cards, recs, main_decks, extra_counter, side_counter,
                              target_main=target_main_size,
                              target_extra=target_extra_size,
                              target_side=target_side_size)
    print(f"New deck built:")
    print("Main deck ({} cards):".format(len(new_deck["main"])))
    print("\n".join(new_deck["main"]))
    print("Extra deck ({} cards):".format(len(new_deck["extra"])))
    print("\n".join(new_deck["extra"]))
    print("Side deck ({} cards):".format(len(new_deck["side"])))
    print("\n".join(new_deck["side"]))

    # ---- Write the Complete Deck to a .ydk File ----
    write_ydk(new_deck, new_deck_file)


if __name__ == "__main__":
    main()
