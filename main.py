import os
import json
import itertools
from collections import Counter, defaultdict
import pandas as pd
import requests
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Transformers for NER
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Initialize the NER pipeline
ner_pipe = pipeline("token-classification", model="dslim/bert-base-NER")


###################################
# Card Info Loading & Helpers
###################################

def load_card_info(filename="cardinfo.json"):
    """Loads card info from a local JSON file (or downloads if missing)."""
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
    """Returns 'Monster', 'Spell', or 'Trap' based on the card's type."""
    if card in card_info:
        t = card_info[card].get("type", "")
        if "Spell" in t:
            return "Spell"
        elif "Trap" in t:
            return "Trap"
        else:
            return "Monster"
    return "Monster"


def get_extra_category(card, card_info):
    """Categorizes extra deck cards (Fusion, Synchro, XYZ, Link, Pendulum, Other)."""
    if card in card_info:
        t = card_info[card].get("type", "").lower()
        if "fusion" in t:
            return "Fusion"
        elif "synchro" in t:
            return "Synchro"
        elif "xyz" in t:
            return "XYZ"
        elif "link" in t:
            return "Link"
        elif "pendulum" in t:
            return "Pendulum"
        else:
            return "Other"
    return "Other"


###################################
# Helper: Extract Referenced Card IDs from Descriptions
###################################

def extract_referenced_card_ids(description, name_to_id):
    """
    Scans a description (lowercased) for substrings matching known card names.
    Returns a set of card IDs that are referenced.
    """
    referenced = set()
    desc_lower = description.lower()
    for name, card_id in name_to_id.items():
        if name in desc_lower:
            referenced.add(card_id)
    return referenced


###################################
# New Helper: Filter Candidates by Synergy Dependencies
###################################

def filter_synergy_candidates(boosted_recs, card_info):
    """
    Checks each candidate's description for required synergy partners.
    If a candidate's description references other cards (via a substring match)
    and none of those referenced cards appear among the candidate recommendations,
    then that candidate is dropped.
    """
    # Build mapping of lower-case card names to IDs for all cards
    name_to_id = {card_info[c]["name"].lower(): c for c in card_info if "name" in card_info[c]}
    candidate_ids = set(card for card, score in boosted_recs)
    filtered = []
    for card, score in boosted_recs:
        if card not in card_info:
            filtered.append((card, score))
            continue
        desc = card_info[card].get("desc", "")
        dependencies = extract_referenced_card_ids(desc, name_to_id)
        if dependencies:
            if dependencies.intersection(candidate_ids):
                filtered.append((card, score))
            else:
                # Drop candidate because its synergy partner is missing
                continue
        else:
            filtered.append((card, score))
    return filtered


###################################
# Part 1: Data Loading & Parsing
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


def load_full_decks(ydk_folder):
    """
    Loads all .ydk files from a folder and returns a list of decks (dict with keys 'main', 'extra', 'side').
    Only decks with a main deck size between 40 and 60 are kept.
    """
    decks = []
    for filename in os.listdir(ydk_folder):
        if filename.endswith(".ydk"):
            path = os.path.join(ydk_folder, filename)
            deck = parse_ydk_file(path)
            if 40 <= len(deck["main"]) <= 60:
                decks.append(deck)
    return decks


###################################
# Part 2: Association Rule Mining (Generic)
###################################

def build_transaction_df(deck_lists):
    """Converts a list of decks (each a list of card IDs) into a one‑hot encoded DataFrame."""
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(deck_lists).transform(deck_lists)
    return pd.DataFrame(te_ary, columns=te.columns_)


def mine_rules(df, min_support=0.05, min_confidence=0.5, max_len=3):
    """Mines frequent itemsets and association rules using Apriori."""
    from mlxtend.frequent_patterns import apriori, association_rules
    freq_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=max_len)
    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)
    return freq_itemsets, rules


def recommend_cards(input_cards, rules, top_n=10):
    """
    Uses association rules to recommend cards.
    Returns a list of (card, avg_confidence) for candidates whose antecedents intersect input_cards.
    """
    recs = defaultdict(list)
    input_set = set(input_cards)
    for idx, rule in rules.iterrows():
        if input_set.intersection(set(rule['antecedents'])):
            for card in set(rule['consequents']) - input_set:
                recs[card].append(rule['confidence'])
    ranked = [(card, sum(confs) / len(confs)) for card, confs in recs.items()]
    return sorted(ranked, key=lambda x: x[1], reverse=True)[:top_n]


def fallback_recommendations(input_cards, decks, top_n=10):
    """Fallback: Uses frequency counts from decks that contain any input card."""
    input_set = set(input_cards)
    freq = Counter()
    for deck in decks:
        if input_set.intersection(set(deck)):
            for card in deck:
                if card not in input_set:
                    freq[card] += 1
    return freq.most_common(top_n)


def recommend_main_deck_by_input(input_cards, main_decks, top_n=10):
    """Recommends main deck cards by filtering decks containing input cards and tallying frequencies."""
    filtered = [deck for deck in main_decks if any(card in deck for card in input_cards)]
    freq = Counter(card for deck in filtered for card in deck if card not in input_cards)
    return freq.most_common(top_n)


def combine_recommendations(assoc, freq):
    """
    Combines two recommendation lists:
      - 'assoc': list of (card, confidence)
      - 'freq': list of (card, count)
    Normalizes frequency to [0,1] and averages if card appears in both.
    Returns a sorted list of (card, combined_score).
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
    Filters main decks to those containing input cards, mines association rules on that subset,
    and returns recommendations.
    """
    filtered = [deck for deck in main_decks if any(card in deck for card in input_cards)]
    if len(filtered) < min_decks:
        return []
    df_context = build_transaction_df(filtered)
    _, rules_context = mine_rules(df_context, min_support=0.05, min_confidence=0.5, max_len=3)
    recs = defaultdict(list)
    input_set = set(input_cards)
    for idx, rule in rules_context.iterrows():
        if input_set.intersection(set(rule['antecedents'])):
            for card in set(rule['consequents']) - input_set:
                recs[card].append(rule['confidence'])
    ranked = [(card, sum(confs) / len(confs)) for card, confs in recs.items()]
    return sorted(ranked, key=lambda x: x[1], reverse=True)[:top_n]


###################################
# Part 3: Statistical Averages for Copy Counts
###################################

def compute_average_copies_main(main_decks):
    """Computes average copies (rounded, capped at 3) for each card in main decks."""
    total = defaultdict(int)
    cnt = defaultdict(int)
    for deck in main_decks:
        freq = Counter(deck)
        for card, copies in freq.items():
            total[card] += copies
            cnt[card] += 1
    return {card: min(3, round(total[card] / cnt[card])) for card in total}


def compute_average_copies_side(full_decks):
    """Computes average copies (rounded, capped at 3) for each card in side decks."""
    total = defaultdict(int)
    cnt = defaultdict(int)
    for deck in full_decks:
        freq = Counter(deck["side"])
        for card, copies in freq.items():
            total[card] += copies
            cnt[card] += 1
    return {card: min(3, round(total[card] / cnt[card])) for card in total}


###################################
# Part 4: Extra & Side Deck Recommendations (Contextual)
###################################

def fallback_extra_deck(input_cards, full_decks, top_n=10):
    extra_counter = Counter()
    for deck in full_decks:
        if any(card in deck["main"] for card in input_cards):
            for card in deck["extra"]:
                extra_counter[card] += 1
    return extra_counter.most_common(top_n)


def recommend_extra_deck_contextual(input_cards, full_decks, top_n=10, min_decks=3):
    filtered = [deck["extra"] for deck in full_decks if
                deck["extra"] and any(card in deck["main"] for card in input_cards)]
    if len(filtered) < min_decks:
        return fallback_extra_deck(input_cards, full_decks, top_n)
    df_extra = build_transaction_df(filtered)
    _, rules_extra = mine_rules(df_extra, min_support=0.02, min_confidence=0.4, max_len=2)
    recs = defaultdict(list)
    for idx, rule in rules_extra.iterrows():
        for card in set(rule['consequents']):
            recs[card].append(rule['confidence'])
    ranked = [(card, sum(confs) / len(confs)) for card, confs in recs.items()]
    return sorted(ranked, key=lambda x: x[1], reverse=True)[:top_n]


def fallback_side_deck(input_cards, full_decks, top_n=10):
    side_counter = Counter()
    for deck in full_decks:
        if any(card in deck["main"] for card in input_cards):
            for card in deck["side"]:
                side_counter[card] += 1
    return side_counter.most_common(top_n)


def recommend_side_deck_contextual(input_cards, full_decks, top_n=10, min_decks=3):
    filtered = [deck["side"] for deck in full_decks if
                deck["side"] and any(card in deck["main"] for card in input_cards)]
    if len(filtered) < min_decks:
        return fallback_side_deck(input_cards, full_decks, top_n)
    df_side = build_transaction_df(filtered)
    _, rules_side = mine_rules(df_side, min_support=0.02, min_confidence=0.4, max_len=2)
    recs = defaultdict(list)
    for idx, rule in rules_side.iterrows():
        for card in set(rule['consequents']):
            recs[card].append(rule['confidence'])
    ranked = [(card, sum(confs) / len(confs)) for card, confs in recs.items()]
    return sorted(ranked, key=lambda x: x[1], reverse=True)[:top_n]


def build_section_deck_simple(candidate_list, target_size):
    """Builds a section deck (extra or side) by taking unique candidates up to target_size."""
    unique = list(dict.fromkeys(candidate_list))
    return unique[:target_size]


def fill_section_deck(current, fallback, target_size):
    """Fills a section deck up to target_size using fallback candidates not already in current."""
    deck = list(current)
    for card in fallback:
        if card not in deck:
            deck.append(card)
        if len(deck) >= target_size:
            break
    return deck[:target_size]


###################################
# Part 5: Iterative Main Deck Construction With Extended Capacity and Score Drop Threshold
###################################

def get_aggregated_entities(card_list, card_info, ner_pipe):
    """
    Aggregates named entities from the descriptions of all cards in card_list.
    Returns a set of lowercased tokens.
    """
    entities = set()
    for card in card_list:
        if card in card_info:
            desc = card_info[card].get("desc", "")
            if desc:
                ner_results = ner_pipe(desc)
                for res in ner_results:
                    entities.add(res["word"].lower())
    return entities


def extract_references_from_deck(card_list, card_info, name_to_id):
    """
    Extracts referenced card IDs from the descriptions of all cards in card_list.
    """
    refs = set()
    for card in card_list:
        if card in card_info:
            desc = card_info[card].get("desc", "")
            refs |= extract_referenced_card_ids(desc, name_to_id)
    return refs


def boost_with_context(candidate_score, candidate, aggregated_entities, aggregated_refs, card_info, boost_factor=1.2,
                       extra_boost=1.3):
    """
    Boosts candidate score if candidate's name matches tokens in aggregated_entities
    or if candidate is explicitly referenced in aggregated_refs.
    """
    candidate_name = card_info[candidate]["name"].lower() if candidate in card_info else ""
    new_score = candidate_score
    if any(entity in candidate_name or candidate_name in entity for entity in aggregated_entities):
        new_score *= boost_factor
    if candidate in aggregated_refs:
        new_score *= extra_boost
    return new_score


def build_main_deck_extended(input_cards, candidate_recs, main_decks, avg_main, card_info, ner_pipe, target_min=40,
                             target_max=60, drop_threshold=0.7):
    """
    Iteratively builds the main deck between target_min and target_max cards.
    Starts with forced input cards and then adds candidates from candidate_recs.
    Candidate scores are boosted using aggregated NER context and explicit reference extraction.
    If synergy filtering removes too many cards, the deck is filled from fallback candidates.
    """
    # Build mapping of lower-case card names to IDs
    name_to_id = {card_info[c]["name"].lower(): c for c in card_info if "name" in card_info[c]}

    new_main_deck = list(input_cards)
    deck_counter = Counter(new_main_deck)

    candidates = candidate_recs[:]  # list of (card, score)

    while len(new_main_deck) < target_max and candidates:
        aggregated_entities = get_aggregated_entities(new_main_deck, card_info, ner_pipe)
        aggregated_refs = extract_references_from_deck(new_main_deck, card_info, name_to_id)

        updated_candidates = []
        for card, score in candidates:
            if deck_counter[card] < avg_main.get(card, 1):
                new_score = boost_with_context(score, card, aggregated_entities, aggregated_refs, card_info,
                                               boost_factor=1.2, extra_boost=1.3)
                updated_candidates.append((card, new_score))
        if not updated_candidates:
            break
        updated_candidates.sort(key=lambda x: x[1], reverse=True)
        best_score = updated_candidates[0][1]
        filtered_candidates = [cand for cand in updated_candidates if cand[1] >= best_score * drop_threshold]
        # If we already have at least target_min cards, we can break if quality drops sharply
        if len(new_main_deck) >= target_min and len(updated_candidates) > 1 and updated_candidates[1][
            1] < best_score * drop_threshold:
            break
        for cand, score in filtered_candidates:
            copies = avg_main.get(cand, 1)
            while deck_counter[cand] < copies and len(new_main_deck) < target_max:
                new_main_deck.append(cand)
                deck_counter[cand] += 1
        candidates = [(card, score) for card, score in updated_candidates if deck_counter[card] < avg_main.get(card, 1)]
        # If no candidates remain, break
        if not candidates:
            break
    # If after the loop we have fewer than target_min cards, fill with fallback from candidate_recs (ignoring drop threshold)
    if len(new_main_deck) < target_min:
        remaining = [cand for cand, score in sorted(candidate_recs, key=lambda x: x[1], reverse=True) if
                     cand not in new_main_deck]
        for cand in remaining:
            copies = avg_main.get(cand, 1)
            while deck_counter[cand] < copies and len(new_main_deck) < target_min:
                new_main_deck.append(cand)
                deck_counter[cand] += 1
            if len(new_main_deck) >= target_min:
                break
    return new_main_deck


###################################
# Part 6: Write Deck to File
###################################

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


###################################
# Part 7: Helper for JSON Serialization
###################################

def convert_frozensets(obj):
    if isinstance(obj, frozenset):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_frozensets(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_frozensets(item) for item in obj]
    else:
        return obj


###################################
# Part 8: Main Function
###################################

def main():
    # ---- Settings ----
    ydk_folder = "ydk_download"  # UPDATE this path as needed
    min_support = 0.05
    min_confidence = 0.5
    max_itemset_length = 3
    target_min_main = 40  # Minimum main deck size
    target_max_main = 60  # Maximum main deck size
    target_extra_size = 15
    target_side_size = 15
    trained_model_file = "trained_model.json"
    new_deck_file = "generated_deck.ydk"

    # Load card info for descriptions and categorization
    card_info = load_card_info()

    # ---- Load Full Decks ----
    print("Loading full decks (main, extra, side)...")
    full_decks = load_full_decks(ydk_folder)
    print(f"Loaded {len(full_decks)} decks.")

    # Separate decks by section
    main_decks = [deck["main"] for deck in full_decks]
    extra_decks = [deck["extra"] for deck in full_decks if deck["extra"]]
    side_decks = [deck["side"] for deck in full_decks if deck["side"]]

    # ---- Global Main Deck Rule Mining ----
    print("Building global transaction DataFrame for main decks...")
    df = build_transaction_df(main_decks)
    print("Mining global association rules on main decks...")
    freq_itemsets, rules = mine_rules(df, min_support=min_support, min_confidence=min_confidence,
                                      max_len=max_itemset_length)
    print(f"Mined {len(rules)} global association rules for main deck.")
    model = {
        "frequent_itemsets": convert_frozensets(freq_itemsets.to_dict(orient="records")),
        "association_rules": convert_frozensets(rules.to_dict(orient="records"))
    }
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
    # Initially boost using input card descriptions
    initial_context = get_aggregated_entities(input_cards, card_info, ner_pipe)
    boosted_recs = []
    for card, score in combined_recs:
        candidate_name = card_info[card]["name"].lower() if card in card_info else ""
        if any(entity in candidate_name or candidate_name in entity for entity in initial_context):
            boosted_recs.append((card, score * 1.2))
        else:
            boosted_recs.append((card, score))
    boosted_recs.sort(key=lambda x: x[1], reverse=True)
    # Apply synergy filtering so that if a candidate depends on another card which isn't available, drop it.
    filtered_boosted_recs = filter_synergy_candidates(boosted_recs, card_info)
    print("Combined, boosted, and synergy-filtered contextual main deck recommendations:")
    for card, score in filtered_boosted_recs:
        print(f"Card {card} with final score {score:.2f}")

    # ---- Compute Average Copies ----
    avg_main = compute_average_copies_main(main_decks)
    avg_side = compute_average_copies_side(full_decks)

    # ---- Build Extended Main Deck (Between 40 and 60 Cards) ----
    new_main_deck = build_main_deck_extended(input_cards, filtered_boosted_recs, main_decks, avg_main, card_info,
                                             ner_pipe, target_min=target_min_main, target_max=target_max_main,
                                             drop_threshold=0.7)
    print(f"Main deck constructed with {len(new_main_deck)} cards.")

    # ---- Extra Deck Recommendations (Contextual) ----
    extra_assoc = recommend_extra_deck_contextual(input_cards, full_decks, top_n=10, min_decks=3)
    extra_freq = fallback_extra_deck(input_cards, full_decks, top_n=10)
    combined_extra = combine_recommendations(extra_assoc, extra_freq)
    candidate_extra = [card for card, _ in combined_extra]
    new_extra_deck = build_section_deck_simple(candidate_extra, target_extra_size)
    # If fewer than target_extra_size, fill with fallback extra deck candidates
    if len(new_extra_deck) < target_extra_size:
        fallback_extra = [card for card, _ in fallback_extra_deck(input_cards, full_decks, top_n=target_extra_size * 2)]
        new_extra_deck = fill_section_deck(new_extra_deck, fallback_extra, target_extra_size)

    # ---- Side Deck Recommendations (Contextual) ----
    side_assoc = recommend_side_deck_contextual(input_cards, full_decks, top_n=10, min_decks=3)
    side_freq = fallback_side_deck(input_cards, full_decks, top_n=10)
    combined_side = combine_recommendations(side_assoc, side_freq)
    candidate_side = [card for card, _ in combined_side]
    new_side_deck = build_section_deck_simple(candidate_side, target_side_size)
    if len(new_side_deck) < target_side_size:
        fallback_side = [card for card, _ in fallback_side_deck(input_cards, full_decks, top_n=target_side_size * 2)]
        new_side_deck = fill_section_deck(new_side_deck, fallback_side, target_side_size)

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
