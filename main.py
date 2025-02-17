import os
import json
import itertools
from collections import Counter, defaultdict
import pandas as pd
import requests
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


##############################
# Card Info Loading & Helpers
##############################

def load_card_info(filename="cardinfo.json"):
    """Loads card information from a local JSON file or downloads it if not present."""
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            card_data = json.load(f)
    else:
        url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
        response = requests.get(url)
        response.raise_for_status()
        card_data = response.json()
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(card_data, f, indent=4)
    card_info = {}
    for card in card_data.get("data", []):
        card_id_str = str(card["id"])
        card_info[card_id_str] = card
    return card_info


def get_card_category(card, card_info):
    """Returns 'Monster', 'Spell', or 'Trap' based on the card's type field.
       Defaults to 'Monster' if not found."""
    if card in card_info:
        type_str = card_info[card].get("type", "")
        if "Spell" in type_str:
            return "Spell"
        elif "Trap" in type_str:
            return "Trap"
        else:
            return "Monster"
    else:
        return "Monster"


##############################
# Part 1: Data Loading & Parsing
##############################

def parse_ydk_file(file_path):
    """Parses a .ydk file into a dict with keys 'main', 'extra', and 'side'."""
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
    Loads all .ydk files from a folder and returns a list of decks
    (each a dict with keys "main", "extra", "side").
    Only decks with a main deck size between 40 and 60 are kept.
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
# Part 2: Association Rule Mining (Generic for a Deck Section)
##############################

def build_transaction_df(deck_lists):
    """Converts a list of decks (each a list of card IDs) into a one‐hot encoded DataFrame."""
    te = TransactionEncoder()
    te_ary = te.fit(deck_lists).transform(deck_lists)
    return pd.DataFrame(te_ary, columns=te.columns_)


def mine_rules(df, min_support=0.05, min_confidence=0.5, max_len=3):
    """Mines frequent itemsets and association rules using Apriori."""
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=max_len)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return frequent_itemsets, rules


def recommend_cards(input_cards, rules, top_n=10):
    """
    Uses association rules (mined globally) to recommend cards for a section.
    Returns list of (card, avg_confidence).
    """
    recommendations = defaultdict(list)
    input_set = set(input_cards)
    for idx, rule in rules.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])
        if input_set.intersection(antecedents):
            for card in consequents - input_set:
                recommendations[card].append(rule['confidence'])
    ranked = []
    for card, confs in recommendations.items():
        avg_conf = sum(confs) / len(confs)
        ranked.append((card, avg_conf))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def fallback_recommendations(input_cards, decks, top_n=10):
    """Fallback: Use frequency counts from decks that contain any input card."""
    input_set = set(input_cards)
    freq = Counter()
    for deck in decks:
        if input_set.intersection(set(deck)):
            for card in deck:
                if card not in input_set:
                    freq[card] += 1
    return freq.most_common(top_n)


def recommend_main_deck_by_input(input_cards, main_decks, top_n=10):
    """Recommends main deck cards by filtering decks that contain any input card and tallying frequencies."""
    filtered = [deck for deck in main_decks if any(card in deck for card in input_cards)]
    freq = Counter(card for deck in filtered for card in deck if card not in input_cards)
    return freq.most_common(top_n)


def combine_recommendations(assoc, freq):
    """
    Combines two recommendation lists:
      - 'assoc' from association rules (card, confidence) with values [0,1]
      - 'freq' from frequency counts (card, count)
    Frequency is normalized to [0,1] and averaged.
    Returns sorted list of (card, combined_score).
    """
    combined = {}
    if freq:
        max_freq = max(freq, key=lambda x: x[1])[1]
    else:
        max_freq = 1
    for card, conf in assoc:
        combined[card] = conf
    for card, count in freq:
        norm = count / max_freq
        if card in combined:
            combined[card] = (combined[card] + norm) / 2
        else:
            combined[card] = norm
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


def recommend_main_deck_contextual(input_cards, main_decks, top_n=10, min_decks=5):
    """
    Filters main decks to only those containing any input card,
    mines association rules on that subset, and returns recommendations.
    """
    filtered = [deck for deck in main_decks if any(card in deck for card in input_cards)]
    if len(filtered) < min_decks:
        return []
    df_context = build_transaction_df(filtered)
    _, rules_context = mine_rules(df_context, min_support=0.05, min_confidence=0.5, max_len=3)
    recommendations = defaultdict(list)
    input_set = set(input_cards)
    for idx, rule in rules_context.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])
        if input_set.intersection(antecedents):
            for card in consequents - input_set:
                recommendations[card].append(rule['confidence'])
    ranked = []
    for card, confs in recommendations.items():
        avg_conf = sum(confs) / len(confs)
        ranked.append((card, avg_conf))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


##############################
# Part 3: Statistical Averages for Copy Counts & Distribution
##############################

def compute_average_copies_main(main_decks):
    """Computes average copies (rounded, capped at 3) for each card in main decks."""
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
    """Computes average copies (rounded, capped at 3) for each card in side decks."""
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


def compute_desired_distribution(decks, card_info, target_size):
    """
    Computes the average distribution of card categories (Monster, Spell, Trap)
    from a list of decks and scales it to a target size.
    """
    totals = {"Monster": 0, "Spell": 0, "Trap": 0}
    count = len(decks)
    for deck in decks:
        counts = {"Monster": 0, "Spell": 0, "Trap": 0}
        for card in deck:
            cat = get_card_category(card, card_info)
            counts[cat] += 1
        for cat in totals:
            totals[cat] += counts[cat]
    if count > 0:
        avg = {cat: totals[cat] / count for cat in totals}
    else:
        avg = {"Monster": target_size, "Spell": 0, "Trap": 0}
    total_avg = sum(avg.values())
    if total_avg == 0:
        return {"Monster": target_size, "Spell": 0, "Trap": 0}
    ratio = {cat: avg[cat] / total_avg for cat in avg}
    desired = {cat: round(ratio[cat] * target_size) for cat in ratio}
    diff = target_size - sum(desired.values())
    # Adjust by adding/subtracting from Monster count
    desired["Monster"] += diff
    return desired


##############################
# Part 4: Section Deck Recommendations (Extra & Side) with Context & Balance
##############################

def recommend_extra_deck(input_cards, full_decks, target_extra):
    """
    Recommends extra deck cards based on decks whose main contains an input card.
    Returns a list (unique, 1 copy each) sorted by frequency.
    """
    extra_counter = Counter()
    for deck in full_decks:
        if any(card in deck["main"] for card in input_cards):
            for card in deck["extra"]:
                extra_counter[card] += 1
    return [card for card, _ in extra_counter.most_common(target_extra)]


def recommend_side_deck(input_cards, full_decks, target_side):
    """
    Recommends side deck cards based on decks whose main contains an input card.
    Returns a list (unique) sorted by frequency.
    """
    side_counter = Counter()
    for deck in full_decks:
        if any(card in deck["main"] for card in input_cards):
            for card in deck["side"]:
                side_counter[card] += 1
    return [card for card, _ in side_counter.most_common(target_side)]


def build_section_deck(candidate_list, target_size, desired_distribution, card_info):
    """
    Builds a deck section (extra or side) from a candidate list,
    trying to match the desired distribution of card types.
    """
    new_section = []
    current_counts = {"Monster": 0, "Spell": 0, "Trap": 0}
    # First pass: add candidates if they help match distribution
    for card in candidate_list:
        if len(new_section) >= target_size:
            break
        cat = get_card_category(card, card_info)
        if current_counts[cat] < desired_distribution.get(cat, target_size):
            new_section.append(card)
            current_counts[cat] += 1
    # If still not full, fill with remaining candidates (unique)
    i = 0
    unique_candidates = list(dict.fromkeys(candidate_list))
    while len(new_section) < target_size and i < len(unique_candidates):
        if unique_candidates[i] not in new_section:
            new_section.append(unique_candidates[i])
        i += 1
    # As a fallback, repeat the first candidate if needed
    while len(new_section) < target_size and unique_candidates:
        new_section.append(unique_candidates[0])
    return new_section


##############################
# Part 5: Main Deck Construction (Balanced by Type)
##############################

def build_main_deck_balanced(input_cards, combined_recs, main_decks, avg_main, target_main, desired_distribution,
                             card_info):
    """
    Constructs a balanced main deck using:
      - Input cards (force-added)
      - Recommended cards (only if they help meet the desired type distribution)
      - Fallback from overall frequency, all while not exceeding average copy counts.
    """
    deck_counter = Counter()
    new_main = []
    current_types = {"Monster": 0, "Spell": 0, "Trap": 0}

    def try_add(card):
        cat = get_card_category(card, card_info)
        if current_types[cat] < desired_distribution.get(cat, target_main) and len(new_main) < target_main:
            new_main.append(card)
            deck_counter[card] += 1
            current_types[cat] += 1
            return True
        return False

    # Force add input cards
    for card in input_cards:
        cat = get_card_category(card, card_info)
        new_main.append(card)
        deck_counter[card] += 1
        current_types[cat] += 1

    # Add recommended cards (from combined contextual recommendations)
    for card, _ in combined_recs:
        copies = avg_main.get(card, 1)
        while deck_counter[card] < copies and len(new_main) < target_main:
            if try_add(card):
                pass
            else:
                break

    # Fill remaining slots from overall frequency
    if len(new_main) < target_main:
        all_main = []
        for deck in main_decks:
            all_main.extend(deck)
        freq = Counter(all_main)
        for card, _ in freq.most_common():
            copies = avg_main.get(card, 1)
            while deck_counter[card] < copies and len(new_main) < target_main:
                if try_add(card):
                    pass
                else:
                    break
            if len(new_main) >= target_main:
                break
    # If still not full, add the first input card repeatedly (shouldn't happen)
    while len(new_main) < target_main:
        new_main.append(input_cards[0])
    return new_main


##############################
# Part 6: Write Deck to File
##############################

def write_ydk(deck, filename):
    """Writes the complete deck (main, extra, side) to a .ydk file."""
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
    """Recursively converts frozenset objects to lists for JSON serialization."""
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
    min_support = 0.05
    min_confidence = 0.5
    max_itemset_length = 3
    target_main_size = 40
    target_extra_size = 15
    target_side_size = 15
    trained_model_file = "trained_model.json"
    new_deck_file = "generated_deck.ydk"

    # Load card info (for card types)
    card_info = load_card_info()

    # ---- Load Full Decks ----
    print("Loading full decks (main, extra, side)...")
    full_decks = load_full_decks(ydk_folder)
    print(f"Loaded {len(full_decks)} decks.")

    # Separate main, extra, and side decks
    main_decks = [deck["main"] for deck in full_decks]
    extra_decks = [deck["extra"] for deck in full_decks if deck["extra"]]
    side_decks = [deck["side"] for deck in full_decks if deck["side"]]

    # ---- Global Main Deck Rule Mining ----
    print("Building global transaction DataFrame for main decks...")
    df = build_transaction_df(main_decks)
    print("Mining global association rules on main decks...")
    frequent_itemsets, rules = mine_rules(df, min_support=min_support, min_confidence=min_confidence,
                                          max_len=max_itemset_length)
    print(f"Mined {len(rules)} global association rules for main deck.")
    freq_itemsets_dict = convert_frozensets(frequent_itemsets.to_dict(orient="records"))
    assoc_rules_dict = convert_frozensets(rules.to_dict(orient="records"))
    model = {"frequent_itemsets": freq_itemsets_dict, "association_rules": assoc_rules_dict}
    with open(trained_model_file, "w") as f:
        json.dump(model, f, indent=4)
    print(f"Trained model saved to {trained_model_file}")

    # ---- Get User Input ----
    user_input = input("Enter base card IDs (separated by commas): ")
    input_cards = [card.strip() for card in user_input.split(",") if card.strip()]
    if not input_cards:
        print("No input cards provided. Exiting.")
        return

    # ---- Contextual Main Deck Recommendations ----
    context_recs = recommend_main_deck_contextual(input_cards, main_decks, top_n=10, min_decks=5)
    freq_recs = recommend_main_deck_by_input(input_cards, main_decks, top_n=10)
    combined_recs = combine_recommendations(context_recs, freq_recs)
    print("Combined contextual main deck recommendations:")
    for card, score in combined_recs:
        print(f"Card {card} with score {score:.2f}")

    # ---- Compute Average Copies & Desired Distribution ----
    avg_main = compute_average_copies_main(main_decks)
    avg_side = compute_average_copies_side(full_decks)
    filtered_main = [deck for deck in main_decks if any(card in deck for card in input_cards)]
    desired_main_dist = compute_desired_distribution(filtered_main, card_info, target_main_size)
    filtered_extra = [deck["extra"] for deck in full_decks if
                      deck["extra"] and any(card in deck["main"] for card in input_cards)]
    desired_extra_dist = compute_desired_distribution(filtered_extra, card_info, target_extra_size)
    filtered_side = [deck["side"] for deck in full_decks if
                     deck["side"] and any(card in deck["main"] for card in input_cards)]
    desired_side_dist = compute_desired_distribution(filtered_side, card_info, target_side_size)

    print("Desired distribution for main deck:", desired_main_dist)
    print("Desired distribution for extra deck:", desired_extra_dist)
    print("Desired distribution for side deck:", desired_side_dist)

    # ---- Build Balanced Main Deck ----
    new_main_deck = build_main_deck_balanced(input_cards, combined_recs, main_decks, avg_main, target_main_size,
                                             desired_main_dist, card_info)

    # ---- Build Extra Deck (Contextual & Balanced) ----
    candidate_extra = recommend_extra_deck(input_cards, full_decks, target_extra_size * 2)  # get extra candidates
    new_extra_deck = build_section_deck(candidate_extra, target_extra_size, desired_extra_dist, card_info)

    # ---- Build Side Deck (Contextual & Balanced) ----
    candidate_side = recommend_side_deck(input_cards, full_decks, target_side_size * 2)
    new_side_deck = build_section_deck(candidate_side, target_side_size, desired_side_dist, card_info)

    # ---- Combine and Output New Deck ----
    new_deck = {"main": new_main_deck, "extra": new_extra_deck, "side": new_side_deck}
    print("New deck built:")
    print("Main deck ({} cards):".format(len(new_deck["main"])))
    print("\n".join(new_deck["main"]))
    print("Extra deck ({} cards):".format(len(new_deck["extra"])))
    print("\n".join(new_deck["extra"]))
    print("Side deck ({} cards):".format(len(new_deck["side"])))
    print("\n".join(new_deck["side"]))
    write_ydk(new_deck, new_deck_file)


if __name__ == "__main__":
    main()
