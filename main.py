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
    Loads all .ydk files from a folder and returns a list of decks (dict with keys "main", "extra", "side").
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
# Part 2: Association Rule Mining (Generic for any deck section)
##############################

def build_transaction_df(deck_lists):
    """
    Converts a list of decks (each a list of card IDs) into a one-hot encoded DataFrame.
    """
    te = TransactionEncoder()
    te_ary = te.fit(deck_lists).transform(deck_lists)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df

def mine_rules(df, min_support=0.03, min_confidence=0.4, max_len=4):
    """
    Mines frequent itemsets and association rules using Apriori.
    Returns both the frequent itemsets and the rules DataFrame.
    """
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=max_len)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return frequent_itemsets, rules

def recommend_cards(input_cards, rules, top_n=10):
    """
    Given a list of input card IDs and association rules,
    recommends additional main deck cards.
    (If any input card appears in a rule's antecedents, the consequent cards are recommended.)
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

def fallback_recommendations(input_cards, main_decks, top_n=10):
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
# Part 3: Statistical Averages for Copy Counts
##############################

def compute_average_copies_main(main_decks):
    """
    Computes the average number of copies (rounded to nearest integer, max capped at 3)
    for each card in the main decks (only for decks where that card appears).
    """
    total = defaultdict(int)
    count = defaultdict(int)
    for deck in main_decks:
        freq = Counter(deck)
        for card, copies in freq.items():
            total[card] += copies
            count[card] += 1
    avg = {}
    for card in total:
        avg[card] = min(3, round(total[card] / count[card]))
    return avg

def compute_average_copies_side(full_decks):
    """
    Computes the average number of copies (rounded, capped at 3) for each card in the side decks.
    """
    total = defaultdict(int)
    count = defaultdict(int)
    for deck in full_decks:
        freq = Counter(deck["side"])
        for card, copies in freq.items():
            total[card] += copies
            count[card] += 1
    avg = {}
    for card in total:
        avg[card] = min(3, round(total[card] / count[card]))
    return avg

##############################
# Part 4: Build Sub-Decks via Association Analysis
##############################

def build_extra_deck_association(extra_decks, target_extra):
    """
    Uses extra deck transactions to mine frequent itemsets and picks a coherent set.
    If association rules exist, choose the largest frequent itemset (by support);
    then, if needed, fill remaining slots by frequency.
    (Extra deck: typically 1 copy per card.)
    """
    if not extra_decks:
        return []
    df_extra = build_transaction_df(extra_decks)
    freq_itemsets_extra = apriori(df_extra, min_support=0.02, use_colnames=True, max_len=2)
    if freq_itemsets_extra.empty:
        # Fall back on frequency counts
        extra_counter = Counter(card for deck in extra_decks for card in deck)
        recommended = [card for card, _ in extra_counter.most_common(target_extra)]
        return recommended
    # Choose the frequent itemset with maximum length and highest support
    freq_itemsets_extra['length'] = freq_itemsets_extra['itemsets'].apply(lambda x: len(x))
    max_length = freq_itemsets_extra['length'].max()
    candidate_sets = freq_itemsets_extra[freq_itemsets_extra['length'] == max_length]
    candidate_sets = candidate_sets.sort_values('support', ascending=False)
    chosen_itemset = list(candidate_sets.iloc[0]['itemsets'])
    # Fill remaining slots using frequency if needed
    extra_counter = Counter(card for deck in extra_decks for card in deck)
    recommended = chosen_itemset[:]
    for card, _ in extra_counter.most_common():
        if card not in recommended and len(recommended) < target_extra:
            recommended.append(card)
    return recommended[:target_extra]

def build_side_deck_association(side_decks, target_side, avg_side):
    """
    Uses side deck transactions to mine frequent itemsets and picks a coherent set.
    Then fills remaining slots (unique cards) by frequency.
    """
    if not side_decks:
        return []
    df_side = build_transaction_df(side_decks)
    freq_itemsets_side = apriori(df_side, min_support=0.02, use_colnames=True, max_len=3)
    if freq_itemsets_side.empty:
        side_counter = Counter(card for deck in side_decks for card in deck)
        recommended = []
        for card, _ in side_counter.most_common():
            if card not in recommended:
                recommended.append(card)
            if len(recommended) >= target_side:
                break
        return recommended[:target_side]
    freq_itemsets_side['length'] = freq_itemsets_side['itemsets'].apply(lambda x: len(x))
    max_length = freq_itemsets_side['length'].max()
    candidate_sets = freq_itemsets_side[freq_itemsets_side['length'] == max_length]
    candidate_sets = candidate_sets.sort_values('support', ascending=False)
    chosen_itemset = list(candidate_sets.iloc[0]['itemsets'])
    recommended = chosen_itemset[:]
    side_counter = Counter(card for deck in side_decks for card in deck)
    for card, _ in side_counter.most_common():
        if card not in recommended and len(recommended) < target_side:
            recommended.append(card)
    return recommended[:target_side]

##############################
# Part 5: Main Deck Construction (Using Statistical Averages)
##############################

def build_main_deck(input_cards, recommendations, main_decks, avg_main, target_main):
    """
    Constructs the main deck using:
     - The input cards (each added using its average copy count),
     - Then recommended cards (added up to that card's average count),
     - Finally, if needed, fill remaining slots using the most frequent main deck cards.
    """
    deck_counter = Counter()
    new_main = []
    # Add input cards using their average (at least one copy)
    for card in input_cards:
        copies = avg_main.get(card, 1)
        for _ in range(copies):
            if len(new_main) < target_main:
                new_main.append(card)
                deck_counter[card] += 1
            else:
                break

    # Add recommended cards using their average copy count
    for card, _ in recommendations:
        copies = avg_main.get(card, 1)
        while deck_counter[card] < copies and len(new_main) < target_main:
            new_main.append(card)
            deck_counter[card] += 1

    # If still not full, fill with most frequent main deck cards (using average counts)
    if len(new_main) < target_main:
        all_main_cards = []
        for d in main_decks:
            all_main_cards.extend(d)
        card_freq = Counter(all_main_cards)
        for card, _ in card_freq.most_common():
            copies = avg_main.get(card, 1)
            while deck_counter[card] < copies and len(new_main) < target_main:
                new_main.append(card)
                deck_counter[card] += 1
            if len(new_main) >= target_main:
                break
    return new_main

##############################
# Part 6: Write Deck to File
##############################

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
# Part 7: Helper for JSON Serialization
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
# Part 8: Main Function
##############################

def main():
    # ---- Settings ----
    ydk_folder = "ydk_download"  # UPDATE this path as needed
    # Parameters for association rule mining on main decks
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

    # Separate main decks for association rule mining
    main_decks = [deck["main"] for deck in full_decks]

    # Also extract extra and side deck lists
    extra_decks = [deck["extra"] for deck in full_decks if deck["extra"]]
    side_decks = [deck["side"] for deck in full_decks if deck["side"]]

    # ---- Compute Frequency Counters (for fallback if needed) ----
    extra_counter = Counter(card for deck in extra_decks for card in deck)
    side_counter = Counter(card for deck in side_decks for card in deck)

    # ---- Build Transaction DataFrame & Mine Rules (Main Deck) ----
    print("Building transaction DataFrame for main decks...")
    df = build_transaction_df(main_decks)
    print("Mining association rules on main decks...")
    frequent_itemsets, rules = mine_rules(df, min_support=min_support, min_confidence=min_confidence,
                                            max_len=max_itemset_length)
    print(f"Mined {len(rules)} association rules for main deck.")

    # ---- Output Trained Model (Main Deck Rules) ----
    freq_itemsets_dict = convert_frozensets(frequent_itemsets.to_dict(orient="records"))
    assoc_rules_dict = convert_frozensets(rules.to_dict(orient="records"))
    model = {
        "frequent_itemsets": freq_itemsets_dict,
        "association_rules": assoc_rules_dict
    }
    with open(trained_model_file, "w") as f:
        json.dump(model, f, indent=4)
    print(f"Trained model saved to {trained_model_file}")

    # ---- Get User Input for Base Main Deck Cards ----
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

    # ---- Compute Average Copies for Main & Side Decks ----
    avg_main = compute_average_copies_main(main_decks)
    avg_side = compute_average_copies_side(full_decks)

    # ---- Build Main Deck (Using Statistical Averages) ----
    new_main_deck = build_main_deck(input_cards, recs, main_decks, avg_main, target_main=target_main_size)

    # ---- Build Extra Deck via Association Analysis ----
    new_extra_deck = build_extra_deck_association(extra_decks, target_extra_size)

    # ---- Build Side Deck via Association Analysis ----
    new_side_deck = build_side_deck_association(side_decks, target_side_size, avg_side)

    # ---- Combine into New Complete Deck ----
    new_deck = {
        "main": new_main_deck,
        "extra": new_extra_deck,
        "side": new_side_deck
    }

    # ---- Output the New Deck ----
    print("New deck built:")
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
