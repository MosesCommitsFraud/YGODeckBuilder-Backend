import os
import json
import itertools
from collections import Counter, defaultdict
import pandas as pd
import requests
import warnings
import pickle

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Boost factors (adjust these as needed)
SYNERGY_BOOST_FACTOR = 1.5  # For boosting based on context/synergy
EXTRA_SYNERGY_BOOST = 1.5  # For extra boost if card is explicitly referenced
ARCHETYPE_BOOST_FACTOR = 1.5  # For boosting based on archetype matching

# Transformers for NER
from transformers import pipeline

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
    card_info = {str(card["id"]): card for card in card_data.get("data", [])}
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
# Dependency Extraction Helpers
###################################

def extract_referenced_card_ids(description, name_to_id):
    """
    Scans a description (lowercased) for substrings matching known card names.
    Returns a set of card IDs that are referenced.
    """
    referenced = set()
    for name, card_id in name_to_id.items():
        if name in description.lower():
            referenced.add(card_id)
    return referenced


def filter_synergy_candidates(boosted_recs, card_info):
    """
    Drops candidate cards whose descriptions indicate dependencies on other cards
    if none of those required cards appear among the candidate recommendations.
    """
    name_to_id = {card_info[c]["name"].lower(): c for c in card_info if "name" in card_info[c]}
    candidate_ids = {card for card, score in boosted_recs}
    filtered = []
    for card, score in boosted_recs:
        if card not in card_info:
            filtered.append((card, score))
            continue
        desc = card_info[card].get("desc", "")
        dependencies = extract_referenced_card_ids(desc, name_to_id)
        if dependencies and not dependencies.intersection(candidate_ids):
            continue
        filtered.append((card, score))
    return filtered


###################################
# YDK Parsing & Deck Loading
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
    Loads all .ydk files from a folder.
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
# Association Rule Mining Helpers
###################################

def build_transaction_df(deck_lists):
    """Converts a list of decks into a one‑hot encoded DataFrame."""
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
    """Generates recommendations from rules."""
    recs = defaultdict(list)
    input_set = set(input_cards)
    for idx, rule in rules.iterrows():
        if input_set.intersection(set(rule['antecedents'])):
            for card in set(rule['consequents']) - input_set:
                recs[card].append(rule['confidence'])
    ranked = [(card, sum(confs) / len(confs)) for card, confs in recs.items()]
    return sorted(ranked, key=lambda x: x[1], reverse=True)[:top_n]


def fallback_recommendations(input_cards, decks, top_n=10):
    """Fallback frequency-based recommendations."""
    input_set = set(input_cards)
    freq = Counter()
    for deck in decks:
        if input_set.intersection(set(deck)):
            for card in deck:
                if card not in input_set:
                    freq[card] += 1
    return freq.most_common(top_n)


def recommend_main_deck_by_input(input_cards, main_decks, top_n=10):
    """Frequency-based recommendations for main deck."""
    filtered = [deck for deck in main_decks if any(card in deck for card in input_cards)]
    freq = Counter(card for deck in filtered for card in deck if card not in input_cards)
    return freq.most_common(top_n)


def combine_recommendations(assoc, freq):
    """Combines association and frequency recommendations."""
    combined = {}
    max_freq = max(freq, key=lambda x: x[1])[1] if freq else 1
    for card, conf in assoc:
        combined[card] = conf
    for card, count in freq:
        norm = count / max_freq
        combined[card] = (combined.get(card, 0) + norm) / (2 if card in combined else 1)
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


def recommend_main_deck_contextual(input_cards, main_decks, top_n=10, min_decks=5):
    """Contextual recommendations from decks containing input cards."""
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
# Statistical Averages
###################################

def compute_average_copies_main(main_decks):
    """Computes average copies (capped at 3) for main deck cards."""
    total, cnt = defaultdict(int), defaultdict(int)
    for deck in main_decks:
        freq = Counter(deck)
        for card, copies in freq.items():
            total[card] += copies
            cnt[card] += 1
    return {card: min(3, round(total[card] / cnt[card])) for card in total}


def compute_average_copies_side(full_decks):
    """Computes average copies (capped at 3) for side deck cards."""
    total, cnt = defaultdict(int), defaultdict(int)
    for deck in full_decks:
        freq = Counter(deck["side"])
        for card, copies in freq.items():
            total[card] += copies
            cnt[card] += 1
    return {card: min(3, round(total[card] / cnt[card])) for card in total}


###################################
# Extra & Side Deck Filtering
###################################

def get_filtered_extra_decks(input_cards, full_decks):
    """Returns extra deck lists from decks whose main deck contains an input card."""
    return [deck["extra"] for deck in full_decks if deck["extra"] and any(card in deck["main"] for card in input_cards)]


def build_extra_deck_filtered(input_cards, full_decks, target_extra):
    """Builds an extra deck from the filtered extra decks using frequency counts."""
    filtered = get_filtered_extra_decks(input_cards, full_decks)
    if not filtered:
        return []
    all_extra = [card for deck in filtered for card in deck]
    freq = Counter(all_extra)
    sorted_candidates = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    candidates = list(dict.fromkeys([cand for cand, _ in sorted_candidates]))
    return candidates[:target_extra]


def get_filtered_side_decks(input_cards, full_decks):
    """Returns side deck lists from decks whose main deck contains an input card."""
    return [deck["side"] for deck in full_decks if deck["side"] and any(card in deck["main"] for card in input_cards)]


def build_side_deck_filtered(input_cards, full_decks, target_side):
    """Builds a side deck from the filtered side decks using frequency counts."""
    filtered = get_filtered_side_decks(input_cards, full_decks)
    if not filtered:
        return []
    all_side = [card for deck in filtered for card in deck]
    freq = Counter(all_side)
    sorted_candidates = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    candidates = list(dict.fromkeys([cand for cand, _ in sorted_candidates]))
    return candidates[:target_side]


###################################
# New: Contextual Extra Deck Builder
###################################

def build_extra_deck_contextual(input_extra, full_decks, target_extra, card_info, ner_pipe, archetypes,
                                fallback_context):
    """
    Constructs an extra deck that better fits the deck's synergy by scoring candidate extra deck cards
    based on frequency, context, and archetype matching. The candidate pool is drawn from the extra deck sections.
    If input_extra is empty, fallback_context (e.g. input_main) is used.
    """
    context_source = input_extra if input_extra else fallback_context
    candidate_lists = get_filtered_extra_decks(context_source, full_decks)
    candidates = []
    for deck in candidate_lists:
        candidates.extend(deck)
    candidates = list(dict.fromkeys(candidates))

    freq_counter = Counter()
    for deck in full_decks:
        if any(card in deck["main"] for card in context_source):
            for card in deck["extra"]:
                freq_counter[card] += 1

    agg_entities = set()
    for card in context_source:
        if card in card_info:
            name_lower = card_info[card]["name"].lower()
            desc_lower = card_info[card].get("desc", "").lower()
            agg_entities.update(name_lower.split())
            agg_entities.update(desc_lower.split())
    context_archetypes = set()
    for card in context_source:
        if card in card_info:
            name_lower = card_info[card]["name"].lower()
            desc_lower = card_info[card].get("desc", "").lower()
            for arch in archetypes:
                arch_lower = arch.lower().strip('"')
                if arch_lower in name_lower or arch_lower in desc_lower:
                    context_archetypes.add(arch_lower)
    candidate_scores = []
    for candidate in candidates:
        baseline = freq_counter[candidate]
        score = baseline
        if candidate in card_info:
            candidate_name = card_info[candidate]["name"].lower()
            candidate_desc = card_info[candidate].get("desc", "").lower()
            for entity in agg_entities:
                if entity in candidate_name or entity in candidate_desc:
                    score *= SYNERGY_BOOST_FACTOR
                    break
            for arch in context_archetypes:
                if arch in candidate_name or arch in candidate_desc:
                    score *= ARCHETYPE_BOOST_FACTOR
                    break
        candidate_scores.append((candidate, score))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    return [card for card, score in candidate_scores[:target_extra]]


###################################
# New: Contextual Side Deck Builder (Input-Based)
###################################

def build_side_deck_contextual(input_cards, full_decks, target_side, card_info, ner_pipe, archetypes, main_deck,
                               extra_deck):
    """
    Constructs a side deck that better fits the input cards by scoring candidate side deck cards
    based on synergy with decks where the input cards appear.
    Candidate pools are drawn from both the side deck and extra deck sections of training decks that contain at least one input card.
    Additionally, any candidate that already appears in the main deck at least 3 times or in the extra deck at least once is filtered out.
    """
    # Gather candidate lists from decks that contain an input card.
    side_candidate_lists = get_filtered_side_decks(input_cards, full_decks)
    extra_candidate_lists = get_filtered_extra_decks(input_cards, full_decks)
    candidates = []
    for deck in side_candidate_lists:
        candidates.extend(deck)
    for deck in extra_candidate_lists:
        candidates.extend(deck)
    candidates = list(dict.fromkeys(candidates))

    # Compute frequency for candidates from decks that contain an input card.
    freq_counter = Counter()
    for deck in full_decks:
        if any(card in deck["main"] for card in input_cards):
            for card in deck["side"]:
                freq_counter[card] += 1
            for card in deck["extra"]:
                freq_counter[card] += 1

    # Aggregate context from the input cards.
    agg_entities = set()
    for card in input_cards:
        if card in card_info:
            name_lower = card_info[card]["name"].lower()
            desc_lower = card_info[card].get("desc", "").lower()
            agg_entities.update(name_lower.split())
            agg_entities.update(desc_lower.split())

    # Determine archetypes present in the input cards.
    input_archetypes = set()
    for card in input_cards:
        if card in card_info:
            name_lower = card_info[card]["name"].lower()
            desc_lower = card_info[card].get("desc", "").lower()
            for arch in archetypes:
                arch_lower = arch.lower().strip('"')
                if arch_lower in name_lower or arch_lower in desc_lower:
                    input_archetypes.add(arch_lower)

    candidate_scores = []
    for candidate in candidates:
        baseline = freq_counter[candidate]
        score = baseline
        if candidate in card_info:
            candidate_name = card_info[candidate]["name"].lower()
            candidate_desc = card_info[candidate].get("desc", "").lower()
            for entity in agg_entities:
                if entity in candidate_name or entity in candidate_desc:
                    score *= SYNERGY_BOOST_FACTOR
                    break
            for arch in input_archetypes:
                if arch in candidate_name or arch in candidate_desc:
                    score *= ARCHETYPE_BOOST_FACTOR
                    break
        candidate_scores.append((candidate, score))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)

    # Filter out candidate if it already appears 3 or more times in main_deck or 1 or more times in extra_deck.
    filtered_candidates = []
    for candidate, score in candidate_scores:
        main_count = main_deck.count(candidate)
        extra_count = extra_deck.count(candidate)
        # Allowed if it appears less than 3 times in main and less than 1 time in extra.
        if main_count < 3 and extra_count < 1:
            filtered_candidates.append(candidate)

    return filtered_candidates[:target_side]


###################################
# Main Deck Distribution Analysis & Filling
###################################

def compute_desired_distribution_main(main_decks, card_info, target_size):
    """
    Computes the average distribution of Monster, Spell, and Trap cards from training main decks,
    scaled to target_size.
    """
    total_monster, total_spell, total_trap = 0, 0, 0
    count = len(main_decks)
    for deck in main_decks:
        m = sum(1 for card in deck if get_card_category(card, card_info) == "Monster")
        s = sum(1 for card in deck if get_card_category(card, card_info) == "Spell")
        t = sum(1 for card in deck if get_card_category(card, card_info) == "Trap")
        total_monster += m
        total_spell += s
        total_trap += t
    if count == 0:
        return {"Monster": target_size, "Spell": 0, "Trap": 0}
    avg_monster = total_monster / count
    avg_spell = total_spell / count
    avg_trap = total_trap / count
    total_avg = avg_monster + avg_spell + avg_trap
    desired_monster = round((avg_monster / total_avg) * target_size)
    desired_spell = round((avg_spell / total_avg) * target_size)
    desired_trap = target_size - desired_monster - desired_spell
    return {"Monster": desired_monster, "Spell": desired_spell, "Trap": desired_trap}


def fill_missing_main_types(new_main_deck, main_decks, card_info, desired_distribution, avg_main, input_cards):
    """
    Enforces the desired distribution (computed for a 60‑card deck) more strictly.
    For each category, if the current count is below the desired value, add fallback candidates
    from that category (using input_cards for fallback recommendations) until either
    the desired count is met or the deck reaches 60 cards.
    """
    current = {"Monster": 0, "Spell": 0, "Trap": 0}
    for card in new_main_deck:
        current[get_card_category(card, card_info)] += 1

    for category in desired_distribution:
        fallback_candidates = [card for card, _ in fallback_recommendations(input_cards, main_decks, top_n=100)
                               if get_card_category(card, card_info) == category and card not in new_main_deck]
        idx = 0
        while current[category] < desired_distribution[category] and len(new_main_deck) < 60:
            if idx >= len(fallback_candidates):
                break
            candidate = fallback_candidates[idx]
            new_main_deck.append(candidate)
            current[category] += 1
            idx += 1

    return new_main_deck


###################################
# Additional Helpers for Main Deck Construction
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
                results = ner_pipe(desc)
                for res in results:
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


def boost_with_context(candidate_score, candidate, aggregated_entities, aggregated_refs, card_info,
                       boost_factor=SYNERGY_BOOST_FACTOR, extra_boost=EXTRA_SYNERGY_BOOST):
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


def boost_with_archetypes(candidate_score, candidate, input_cards, card_info, archetypes,
                          boost_factor=ARCHETYPE_BOOST_FACTOR):
    """
    Boosts candidate score if any archetype (from the provided list) appears in any input card's name or description
    and also appears in the candidate card's name or description.
    """
    input_archetypes = set()
    for card in input_cards:
        if card in card_info:
            name_lower = card_info[card]["name"].lower()
            desc_lower = card_info[card].get("desc", "").lower()
            for archetype in archetypes:
                archetype_lower = archetype.lower().strip('"')
                if archetype_lower in name_lower or archetype_lower in desc_lower:
                    input_archetypes.add(archetype_lower)
    if candidate in card_info:
        candidate_name = card_info[candidate]["name"].lower()
        candidate_desc = card_info[candidate].get("desc", "").lower()
        for archetype in input_archetypes:
            if archetype in candidate_name or archetype in candidate_desc:
                candidate_score *= boost_factor
                break
    return candidate_score


def build_main_deck_extended(input_cards, candidate_recs, main_decks, avg_main, card_info, ner_pipe,
                             target_min=40, target_max=60, drop_threshold=0.5, archetypes=[]):
    """
    Iteratively builds the main deck between target_min and target_max cards.
    Uses boosted candidate recommendations; if synergy filtering removes too many cards,
    fills the deck from a combined fallback pool.
    Now incorporates both synergy- and archetype-based boosting.
    """
    name_to_id = {card_info[c]["name"].lower(): c for c in card_info if "name" in card_info[c]}
    new_main_deck = list(input_cards)
    deck_counter = Counter(new_main_deck)
    original_candidates = sorted(candidate_recs, key=lambda x: x[1], reverse=True)
    candidates = candidate_recs[:]  # Copy of candidate_recs
    while len(new_main_deck) < target_max and candidates:
        aggregated_entities = get_aggregated_entities(new_main_deck, card_info, ner_pipe)
        aggregated_refs = extract_references_from_deck(new_main_deck, card_info, name_to_id)
        updated_candidates = []
        for card, score in candidates:
            if deck_counter[card] < avg_main.get(card, 1):
                new_score = boost_with_context(score, card, aggregated_entities, aggregated_refs, card_info,
                                               boost_factor=SYNERGY_BOOST_FACTOR, extra_boost=EXTRA_SYNERGY_BOOST)
                new_score = boost_with_archetypes(new_score, card, input_cards, card_info, archetypes,
                                                  boost_factor=ARCHETYPE_BOOST_FACTOR)
                updated_candidates.append((card, new_score))
        if not updated_candidates:
            break
        updated_candidates.sort(key=lambda x: x[1], reverse=True)
        best_score = updated_candidates[0][1]
        filtered_candidates = [cand for cand in updated_candidates if cand[1] >= best_score * drop_threshold]
        for cand, score in filtered_candidates:
            copies = avg_main.get(cand, 1)
            while deck_counter[cand] < copies and len(new_main_deck) < target_max:
                new_main_deck.append(cand)
                deck_counter[cand] += 1
        candidates = [(card, score) for card, score in updated_candidates if deck_counter[card] < avg_main.get(card, 1)]
        if not candidates:
            break
    if len(new_main_deck) < target_min:
        remaining_candidates = [cand for cand, score in original_candidates if cand not in new_main_deck]
        fallback_list = [card for card, _ in fallback_recommendations(input_cards, main_decks, top_n=target_max)]
        combined_pool = list(dict.fromkeys(remaining_candidates + fallback_list))
        for cand in combined_pool:
            copies = avg_main.get(cand, 1)
            while deck_counter[cand] < copies and len(new_main_deck) < target_min:
                new_main_deck.append(cand)
                deck_counter[cand] += 1
            if len(new_main_deck) >= target_min:
                break
    while len(new_main_deck) < target_min:
        new_main_deck.append(input_cards[0])
    return new_main_deck


###################################
# Write Deck to File
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
# JSON Serialization Helper
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
# Main Function
###################################

def main():
    # Settings
    ydk_folder = "ydk_download"  # UPDATE this path as needed
    min_support = 0.05
    min_confidence = 0.5
    max_itemset_length = 3
    target_min_main = 40
    target_max_main = 60
    target_extra_size = 15
    target_side_size = 15
    trained_model_file = "trained_model.pkl"  # Save model as a pickle file
    new_deck_file = "generated_deck.ydk"

    # Load card info
    card_info = load_card_info()

    # Load archetypes from the JSON file
    with open("archetypes.json", "r", encoding="utf-8") as f:
        archetypes = json.load(f)

    # Load full decks
    print("Loading full decks...")
    full_decks = load_full_decks(ydk_folder)
    print(f"Loaded {len(full_decks)} decks.")

    # Separate decks by section
    main_decks = [deck["main"] for deck in full_decks]
    extra_decks = [deck["extra"] for deck in full_decks if deck["extra"]]
    side_decks = [deck["side"] for deck in full_decks if deck["side"]]

    # Global Main Deck Rule Mining
    print("Mining global rules for main decks...")
    df = build_transaction_df(main_decks)
    freq_itemsets, rules = mine_rules(df, min_support=min_support, min_confidence=min_confidence,
                                      max_len=max_itemset_length)
    print(f"Mined {len(rules)} global association rules for main deck.")
    model = {
        "frequent_itemsets": convert_frozensets(freq_itemsets.to_dict(orient="records")),
        "association_rules": convert_frozensets(rules.to_dict(orient="records"))
    }
    # Save model as a pickle file
    with open(trained_model_file, "wb") as f:
        pickle.dump(model, f)
    print(f"Trained model saved to {trained_model_file}")

    # --- Get user input for base cards for each deck section ---
    user_input_main = input("Enter base MAIN deck card IDs (separated by commas): ")
    input_main_cards = [card.strip() for card in user_input_main.split(",") if card.strip()]

    user_input_extra = input("Enter base EXTRA deck card IDs (separated by commas, if any): ")
    input_extra_cards = [card.strip() for card in user_input_extra.split(",") if card.strip()]

    user_input_side = input("Enter base SIDE deck card IDs (separated by commas, if any): ")
    input_side_cards = [card.strip() for card in user_input_side.split(",") if card.strip()]

    if not input_main_cards:
        print("No base MAIN deck cards provided. Exiting.")
        return

    # --- Main Deck Recommendations & Construction ---
    context_recs = recommend_main_deck_contextual(input_main_cards, main_decks, top_n=10, min_decks=5)
    freq_recs = recommend_main_deck_by_input(input_main_cards, main_decks, top_n=10)
    combined_recs = combine_recommendations(context_recs, freq_recs)
    initial_context = get_aggregated_entities(input_main_cards, card_info, ner_pipe)

    boosted_recs = []
    for card, score in combined_recs:
        new_score = score
        if card in card_info:
            if any(entity in card_info[card]["name"].lower() or card_info[card]["name"].lower() in entity for entity in
                   initial_context):
                new_score *= SYNERGY_BOOST_FACTOR
        new_score = boost_with_archetypes(new_score, card, input_main_cards, card_info, archetypes,
                                          boost_factor=ARCHETYPE_BOOST_FACTOR)
        boosted_recs.append((card, new_score))

    boosted_recs.sort(key=lambda x: x[1], reverse=True)
    filtered_boosted_recs = filter_synergy_candidates(boosted_recs, card_info)
    print("Main deck candidate recommendations:")
    for card, score in filtered_boosted_recs:
        print(f"Card {card} - Score: {score:.2f}")

    avg_main = compute_average_copies_main(main_decks)
    avg_side = compute_average_copies_side(full_decks)

    new_main_deck = build_main_deck_extended(input_main_cards, filtered_boosted_recs, main_decks, avg_main, card_info,
                                             ner_pipe, target_min=target_min_main, target_max=target_max_main,
                                             drop_threshold=0.5, archetypes=archetypes)
    print(f"Main deck constructed with {len(new_main_deck)} cards.")

    desired_distribution = compute_desired_distribution_main(main_decks, card_info, target_size=target_max_main)
    print("Desired main deck distribution (for 60 cards):", desired_distribution)
    new_main_deck = fill_missing_main_types(new_main_deck, main_decks, card_info, desired_distribution, avg_main,
                                            input_main_cards)
    print(f"Main deck after enforcing distribution: {len(new_main_deck)} cards.")

    # --- Extra Deck Construction (Contextual) ---
    extra_context = input_extra_cards if input_extra_cards else input_main_cards
    new_extra_deck = build_extra_deck_contextual(extra_context, full_decks, target_extra_size, card_info, ner_pipe,
                                                 archetypes, input_main_cards)
    if len(new_extra_deck) < target_extra_size:
        fallback_extra = [card for card, _ in
                          fallback_recommendations(input_main_cards, full_decks, top_n=target_extra_size * 2)]
        new_extra_deck = fill_section_deck(new_extra_deck, fallback_extra, target_extra_size)

    # --- Side Deck Construction (Contextual based on input cards) ---
    new_side_deck = build_side_deck_contextual(input_main_cards, full_decks, target_side_size, card_info, ner_pipe,
                                               archetypes, new_main_deck, new_extra_deck)
    if len(new_side_deck) < target_side_size:
        fallback_side = [card for card, _ in
                         fallback_recommendations(input_main_cards, full_decks, top_n=target_side_size * 2)]
        new_side_deck = fill_section_deck(new_side_deck, fallback_side, target_side_size)

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
