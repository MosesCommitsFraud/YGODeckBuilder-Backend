import os
import json
import itertools
from collections import Counter, defaultdict
import pandas as pd
import requests
import warnings
import pickle
import math

warnings.filterwarnings("ignore", category=RuntimeWarning)

###################################
# Configuration – IMPORTANT VARIABLES
###################################
# Boost factors (modify these to adjust how strongly synergy and archetype signals affect candidate scoring)
SYNERGY_BOOST_FACTOR = 1.5  # Multiplier when a candidate card's name/description matches synergy context tokens
EXTRA_SYNERGY_BOOST = 1.5  # Additional multiplier when a candidate is explicitly referenced
ARCHETYPE_BOOST_FACTOR = 1.5  # Multiplier when a candidate card matches an archetype found in input cards

# Enhanced boosting factors for stronger synergy prioritization
ENHANCED_SYNERGY_BOOST = 3.0  # Increased from 1.5
ENHANCED_ARCHETYPE_BOOST = 3.0  # Increased from 1.5
POPULARITY_PENALTY = 0.7  # New factor to reduce pure popularity-based picking

# Association rule mining thresholds (affect which frequent itemsets and rules are mined)
MIN_SUPPORT = 0.05  # Minimum support for frequent itemsets
MIN_CONFIDENCE = 0.5  # Minimum confidence for association rules
MAX_ITEMSET_LENGTH = 3  # Maximum length (number of items) in frequent itemsets

# Target deck sizes (these define the output deck size constraints)
TARGET_MIN_MAIN = 40  # Minimum number of cards in the Main Deck
TARGET_MAX_MAIN = 60  # Maximum number of cards in the Main Deck
TARGET_EXTRA_SIZE = 15  # Desired size of the Extra Deck
TARGET_SIDE_SIZE = 15  # Desired size of the Side Deck

# Candidate filtering threshold
DROP_THRESHOLD = 0.5  # Relative threshold to drop low-scoring candidate recommendations

# Duplicate filtering in side deck:
# A candidate is filtered out from the side deck if it appears 3+ times in main or 1+ time in extra.
MAIN_DUPLICATE_LIMIT = 3
EXTRA_DUPLICATE_LIMIT = 1

# File and folder settings
CARDINFO_FILE = "cardinfo.json"  # Local file for card information
ARCHETYPES_FILE = "archetypes.json"  # JSON file containing archetypes
YDK_FOLDER = "ydk_download"  # Folder containing .ydk deck files
TRAINED_MODEL_FILE = "trained_model.pkl"  # Output file for the trained model (pickle)
OUTPUT_DECK_FILE = "generated_deck.ydk"  # Output file for the generated deck

# Caching system for expensive operations
CACHE = {
    'archetypes': {},  # card_id -> detected archetypes
    'references': {},  # card_id -> referenced card_ids
    'synergy_score': {},  # (card_id, frozenset(context_cards)) -> score
    'name_to_id': None,  # Card name lookup (created only once)
}


def clear_cache():
    """Clears all cached values"""
    CACHE['archetypes'].clear()
    CACHE['references'].clear()
    CACHE['synergy_score'].clear()
    CACHE['name_to_id'] = None


###################################
# Card Info Loading & Helpers
###################################
def load_card_info(filename=CARDINFO_FILE):
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
    return {str(card["id"]): card for card in card_data.get("data", [])}


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
# Extra Deck Type Validation
###################################
def is_valid_extra_deck_card(card_id, card_info):
    """
    Validates if a card is a valid extra deck card type.
    Valid types: Fusion, Synchro, XYZ, Link monsters.
    """
    if card_id not in card_info:
        return False

    card_type = card_info[card_id].get("type", "").lower()
    return any(t in card_type for t in ["fusion", "synchro", "xyz", "link"])


def is_extra_deck_card(card_id, card_info):
    """
    Checks if a card is categorized as an extra deck card (not necessarily a valid one).
    """
    if card_id not in card_info:
        return False

    card_type = card_info[card_id].get("type", "").lower()
    return any(t in card_type for t in ["fusion", "synchro", "xyz", "link"])


def get_extra_deck_distribution(extra_decks, card_info):
    """
    Computes the average distribution of extra deck card types (Fusion, Synchro, XYZ, Link).
    Returns a dictionary with the proportion of each type.
    """
    type_counts = {"Fusion": 0, "Synchro": 0, "XYZ": 0, "Link": 0, "Other": 0}
    total_count = 0

    for deck in extra_decks:
        for card in deck:
            category = get_extra_category(card, card_info)
            type_counts[category] += 1
            total_count += 1

    # Convert counts to proportions
    if total_count > 0:
        type_distribution = {
            t: count / total_count for t, count in type_counts.items() if t != "Other"
        }
    else:
        # Fallback to equal distribution if no data
        type_distribution = {"Fusion": 0.25, "Synchro": 0.25, "XYZ": 0.25, "Link": 0.25}

    return type_distribution


###################################
# Improved Archetype Detection
###################################
def detect_archetypes(card_id, card_info, archetypes):
    """
    Directly detects archetypes in a card's name and description using string matching.
    Returns a set of archetypes found in the card.
    """
    if card_id not in card_info:
        return set()

    name = card_info[card_id].get("name", "").lower()
    desc = card_info[card_id].get("desc", "").lower()
    combined_text = name + " " + desc

    found_archetypes = set()
    for archetype in archetypes:
        # Strip quotes and lowercase for comparison
        clean_archetype = archetype.lower().strip('"')
        # Some archetypes need to be complete words, not just substrings
        # (e.g., "Red" shouldn't match "Red-Eyes" as an archetype)
        if (
                # Check for complete word match (surrounded by spaces or punctuation)
                f" {clean_archetype} " in f" {combined_text} " or
                # Check for archetype at start of text
                combined_text.startswith(clean_archetype + " ") or
                # Check for archetype at end of text
                combined_text.endswith(" " + clean_archetype) or
                # Check for specific archetype patterns like "X-Type"
                f"{clean_archetype}-" in combined_text or
                # Check for possessive form
                f"{clean_archetype}'s" in combined_text
        ):
            found_archetypes.add(clean_archetype)

    return found_archetypes


def detect_archetypes_cached(card_id, card_info, archetypes):
    """
    Cached version of archetype detection that only computes once per card.
    """
    if card_id in CACHE['archetypes']:
        return CACHE['archetypes'][card_id]

    result = detect_archetypes(card_id, card_info, archetypes)
    CACHE['archetypes'][card_id] = result
    return result


def get_deck_archetypes(card_list, card_info, archetypes):
    """
    Gets all archetypes present in a list of cards.
    Returns a set of archetypes.
    """
    deck_archetypes = set()
    for card in card_list:
        card_archetypes = detect_archetypes_cached(card, card_info, archetypes)
        deck_archetypes.update(card_archetypes)
    return deck_archetypes


###################################
# Improved Card Reference Detection
###################################
def build_card_name_lookup(card_info):
    """
    Builds an optimized lookup dictionary for card names.
    Maps lowercased card names to their IDs.
    """
    # Use cached version if available
    if CACHE['name_to_id'] is not None:
        return CACHE['name_to_id']

    name_to_id = {}
    # First pass - exact full names
    for card_id, card_data in card_info.items():
        if "name" in card_data:
            name_to_id[card_data["name"].lower()] = card_id

    # Store in cache
    CACHE['name_to_id'] = name_to_id
    return name_to_id


def extract_referenced_card_ids_improved(card_id, card_info, name_to_id):
    """
    Enhanced method to extract referenced card IDs from a card's description.
    Returns a set of card IDs that are referenced.
    """
    if card_id not in card_info:
        return set()

    desc = card_info[card_id].get("desc", "").lower()
    if not desc:
        return set()

    referenced_ids = set()

    # Look for full card names in the description
    for name, ref_id in name_to_id.items():
        # Skip self-references
        if ref_id == card_id:
            continue

        # Check if the card name appears as a complete word in the description
        if f" {name} " in f" {desc} " or desc.startswith(name + " ") or desc.endswith(" " + name):
            referenced_ids.add(ref_id)

    return referenced_ids


def extract_referenced_card_ids_improved_cached(card_id, card_info, name_to_id):
    """
    Cached version of reference detection that only computes once per card.
    """
    if card_id in CACHE['references']:
        return CACHE['references'][card_id]

    result = extract_referenced_card_ids_improved(card_id, card_info, name_to_id)
    CACHE['references'][card_id] = result
    return result


def get_deck_card_references(card_list, card_info):
    """
    Gets all card references within a list of cards.
    Returns a set of referenced card IDs.
    """
    name_to_id = build_card_name_lookup(card_info)
    all_references = set()

    for card in card_list:
        refs = extract_referenced_card_ids_improved_cached(card, card_info, name_to_id)
        all_references.update(refs)

    return all_references


###################################
# Improved Synergy Scoring
###################################
def score_card_synergy_enhanced(candidate_id, input_cards, card_info, archetypes,
                                synergy_boost=ENHANCED_SYNERGY_BOOST,
                                archetype_boost=ENHANCED_ARCHETYPE_BOOST):
    """
    Enhanced scoring function with stronger emphasis on synergy and archetype matching.
    Uses caching for performance and applies popularity penalty.
    """
    # Create a cache key using frozenset (order-independent)
    cache_key = (candidate_id, frozenset(input_cards))
    if cache_key in CACHE['synergy_score']:
        return CACHE['synergy_score'][cache_key]

    if candidate_id not in card_info:
        return 1.0

    # Get archetypes from input cards - using cached version
    input_archetypes = set()
    for card in input_cards:
        card_archetypes = detect_archetypes_cached(card, card_info, archetypes)
        input_archetypes.update(card_archetypes)

    # Initialize the score multipliers
    score_multiplier = 1.0
    synergy_applied = False

    # Build name lookup only once
    name_to_id = build_card_name_lookup(card_info)

    # Check for archetype matches - using cached version
    candidate_archetypes = detect_archetypes_cached(candidate_id, card_info, archetypes)
    if candidate_archetypes & input_archetypes:  # Intersection
        score_multiplier *= archetype_boost
        synergy_applied = True

    # Check for direct references in both directions
    # 1. This card is referenced by input cards
    for card in input_cards:
        refs = extract_referenced_card_ids_improved_cached(card, card_info, name_to_id)
        if candidate_id in refs:
            score_multiplier *= synergy_boost
            synergy_applied = True
            break

    # 2. This card references input cards
    refs = extract_referenced_card_ids_improved_cached(candidate_id, card_info, name_to_id)
    if any(ref in input_cards for ref in refs):
        score_multiplier *= synergy_boost
        synergy_applied = True

    # Apply popularity penalty if no synergy was found
    if not synergy_applied:
        score_multiplier *= POPULARITY_PENALTY

    # Cache and return the result
    CACHE['synergy_score'][cache_key] = score_multiplier
    return score_multiplier


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


def load_full_decks(ydk_folder=YDK_FOLDER):
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


def mine_rules(df, min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE, max_len=MAX_ITEMSET_LENGTH):
    """Mines frequent itemsets and association rules using Apriori."""
    from mlxtend.frequent_patterns import apriori, association_rules
    freq_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=max_len)
    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)
    return freq_itemsets, rules


def recommend_cards(input_cards, rules, top_n=10):
    """Generates recommendations from association rules."""
    recs = defaultdict(list)
    input_set = set(input_cards)
    for _, rule in rules.iterrows():
        if input_set.intersection(set(rule['antecedents'])):
            for card in set(rule['consequents']) - input_set:
                recs[card].append(rule['confidence'])
    ranked = [(card, sum(confs) / len(confs)) for card, confs in recs.items()]
    return sorted(ranked, key=lambda x: x[1], reverse=True)[:top_n]


def fallback_recommendations(input_cards, decks, top_n=10):
    """Provides fallback frequency-based recommendations."""
    freq = Counter()
    input_set = set(input_cards)
    for deck in decks:
        if input_set.intersection(set(deck)):
            for card in deck:
                if card not in input_set:
                    freq[card] += 1
    return freq.most_common(top_n)


def recommend_main_deck_by_input(input_cards, main_decks, top_n=10):
    """Provides frequency-based recommendations for the main deck."""
    filtered = [deck for deck in main_decks if any(card in deck for card in input_cards)]
    freq = Counter(card for deck in filtered for card in deck if card not in input_cards)
    return freq.most_common(top_n)


def combine_recommendations(assoc, freq):
    """Combines association-rule and frequency-based recommendations."""
    combined = {}
    max_freq = max(freq, key=lambda x: x[1])[1] if freq else 1
    for card, conf in assoc:
        combined[card] = conf
    for card, count in freq:
        norm = count / max_freq
        combined[card] = (combined.get(card, 0) + norm) / (2 if card in combined else 1)
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


def recommend_main_deck_contextual(input_cards, main_decks, top_n=10, min_decks=5):
    """Generates contextual recommendations from decks containing input cards."""
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


###################################
# Statistical Averages
###################################
def compute_average_copies_main(main_decks):
    """Computes average copies (capped at 3) for main deck cards."""
    total, cnt = defaultdict(int), defaultdict(int)
    for deck in main_decks:
        for card, copies in Counter(deck).items():
            total[card] += copies
            cnt[card] += 1
    return {card: min(3, round(total[card] / cnt[card])) for card in total}


def compute_average_copies_side(full_decks):
    """Computes average copies (capped at 3) for side deck cards."""
    total, cnt = defaultdict(int), defaultdict(int)
    for deck in full_decks:
        for card, copies in Counter(deck["side"]).items():
            total[card] += copies
            cnt[card] += 1
    return {card: min(3, round(total[card] / cnt[card])) for card in total}


###################################
# Extra & Side Deck Filtering
###################################
def get_filtered_extra_decks(input_cards, full_decks):
    """Returns extra deck lists from decks where the main deck contains an input card."""
    return [deck["extra"] for deck in full_decks if deck["extra"] and any(card in deck["main"] for card in input_cards)]


def get_filtered_side_decks(input_cards, full_decks):
    """Returns side deck lists from decks where the main deck contains an input card."""
    return [deck["side"] for deck in full_decks if deck["side"] and any(card in deck["main"] for card in input_cards)]


###################################
# Dynamic Deck Distribution
###################################
def analyze_archetype_distribution(input_cards, card_info, archetypes, main_decks, target_size):
    """
    Dynamically analyzes the monster/spell/trap distribution for detected archetypes.
    Returns a custom distribution that matches the archetype's strategy.
    """
    # Get the archetypes from input cards
    input_archetypes = set()
    for card in input_cards:
        archs = detect_archetypes_cached(card, card_info, archetypes)
        input_archetypes.update(archs)

    if not input_archetypes:
        # Default distribution if no archetypes detected
        return compute_desired_distribution_main(main_decks, card_info, target_size)

    # Find decks that match our archetypes
    archetype_decks = []
    for deck in main_decks:
        deck_archetypes = set()
        for card in deck:
            card_archs = detect_archetypes_cached(card, card_info, archetypes)
            deck_archetypes.update(card_archs)

        # If this deck shares at least one archetype with our input
        if deck_archetypes & input_archetypes:
            archetype_decks.append(deck)

    # If we found archetype-matching decks, compute distribution from them
    if archetype_decks:
        total_monster, total_spell, total_trap = 0, 0, 0
        for deck in archetype_decks:
            total_monster += sum(1 for card in deck if get_card_category(card, card_info) == "Monster")
            total_spell += sum(1 for card in deck if get_card_category(card, card_info) == "Spell")
            total_trap += sum(1 for card in deck if get_card_category(card, card_info) == "Trap")

        count = len(archetype_decks)
        if count == 0:
            # Fallback if no valid decks
            return compute_desired_distribution_main(main_decks, card_info, target_size)

        avg_monster = total_monster / count
        avg_spell = total_spell / count
        avg_trap = total_trap / count
        total_avg = avg_monster + avg_spell + avg_trap

        return {
            "Monster": round((avg_monster / total_avg) * target_size),
            "Spell": round((avg_spell / total_avg) * target_size),
            "Trap": target_size - round((avg_monster / total_avg) * target_size) - round(
                (avg_spell / total_avg) * target_size)
        }
    else:
        # Fallback to overall average if no matching archetype decks
        return compute_desired_distribution_main(main_decks, card_info, target_size)


def compute_desired_distribution_main(main_decks, card_info, target_size):
    """
    Computes the average distribution (Monster, Spell, Trap) from training decks scaled to target_size.
    """
    total_monster, total_spell, total_trap = 0, 0, 0
    for deck in main_decks:
        total_monster += sum(1 for card in deck if get_card_category(card, card_info) == "Monster")
        total_spell += sum(1 for card in deck if get_card_category(card, card_info) == "Spell")
        total_trap += sum(1 for card in deck if get_card_category(card, card_info) == "Trap")
    count = len(main_decks)
    if count == 0:
        return {"Monster": target_size, "Spell": 0, "Trap": 0}
    avg_monster = total_monster / count
    avg_spell = total_spell / count
    avg_trap = total_trap / count
    total_avg = avg_monster + avg_spell + avg_trap
    return {
        "Monster": round((avg_monster / total_avg) * target_size),
        "Spell": round((avg_spell / total_avg) * target_size),
        "Trap": target_size - round((avg_monster / total_avg) * target_size) - round(
            (avg_spell / total_avg) * target_size)
    }


def apply_flexible_distribution(new_main_deck, main_decks, card_info, desired_distribution,
                                avg_main, input_cards, archetypes, synergy_priority=True):
    """
    More flexible implementation for filling the deck that prioritizes synergy.
    Allows reasonable deviation from the desired distribution if it improves card quality.
    """
    # Count current types
    current = Counter(get_card_category(card, card_info) for card in new_main_deck)

    # Calculate the target distribution scaled to the desired total
    target_total = TARGET_MAX_MAIN
    target_distribution = {
        cat: math.ceil((desired / target_total) * TARGET_MAX_MAIN)
        for cat, desired in desired_distribution.items()
    }

    # Get recommendations with enhanced synergy scoring
    candidate_recs = []
    recommendations = recommend_main_deck_by_input(input_cards, main_decks, top_n=100)

    for card, score in recommendations:
        if card in new_main_deck:
            continue

        # Apply enhanced synergy scoring
        synergy_multiplier = score_card_synergy_enhanced(card, input_cards, card_info, archetypes)
        new_score = score * synergy_multiplier

        # Get card category and add to appropriate list
        category = get_card_category(card, card_info)
        candidate_recs.append((card, new_score, category))

    # Sort by score
    candidate_recs.sort(key=lambda x: x[1], reverse=True)

    # First pass: Add high-synergy cards regardless of distribution
    if synergy_priority:
        high_synergy_candidates = [c for c in candidate_recs[:30] if c[1] > 2.0]  # Cards with good synergy
        for card, score, category in high_synergy_candidates:
            if card not in new_main_deck and len(new_main_deck) < TARGET_MAX_MAIN:
                # Add even if we're a bit over on this category
                if current[category] < target_distribution[category] * 1.3:  # Allow 30% over target
                    copies = avg_main.get(card, 1)
                    for _ in range(copies):
                        if len(new_main_deck) < TARGET_MAX_MAIN:
                            new_main_deck.append(card)
                            current[category] += 1

    # Second pass: Fill remaining slots with attention to distribution
    for category, target in target_distribution.items():
        # Filter candidates by category
        cat_candidates = [c for c in candidate_recs if c[2] == category]

        # Add cards until we reach the target or run out of candidates
        for card, _, _ in cat_candidates:
            if current[category] >= target:
                break

            if card not in new_main_deck and len(new_main_deck) < TARGET_MAX_MAIN:
                copies = avg_main.get(card, 1)
                for _ in range(copies):
                    if current[category] < target and len(new_main_deck) < TARGET_MAX_MAIN:
                        new_main_deck.append(card)
                        current[category] += 1

    # Ensure minimum deck size
    if len(new_main_deck) < TARGET_MIN_MAIN:
        remaining_candidates = [c[0] for c in candidate_recs if c[0] not in new_main_deck]

        while len(new_main_deck) < TARGET_MIN_MAIN and remaining_candidates:
            new_main_deck.append(remaining_candidates.pop(0))

    return new_main_deck


###################################
# Legacy Filtering Functions
###################################
def filter_synergy_candidates(boosted_recs, card_info):
    """
    Legacy method: Filters out candidate cards whose descriptions indicate dependencies on other cards.
    """
    name_to_id = build_card_name_lookup(card_info)
    candidate_ids = {card for card, _ in boosted_recs}
    filtered = []
    for card, score in boosted_recs:
        if card not in card_info:
            filtered.append((card, score))
            continue
        dependencies = extract_referenced_card_ids_improved_cached(card, card_info, name_to_id)
        if dependencies and not dependencies.intersection(candidate_ids):
            continue
        filtered.append((card, score))
    return filtered


###################################
# Optimized Extra Deck Builder
###################################
def build_extra_deck_optimized(input_cards, full_decks, target_extra, card_info, archetypes):
    """
    Optimized version of the extra deck builder with enhanced synergy scoring and better performance.
    """
    # Get filtered candidate pool
    candidate_lists = get_filtered_extra_decks(input_cards, full_decks)
    candidates = list({card for deck in candidate_lists for card in deck
                       if is_valid_extra_deck_card(card, card_info)})

    # Base frequency counts
    freq_counter = Counter()
    for deck in full_decks:
        if any(card in deck["main"] for card in input_cards):
            for card in deck["extra"]:
                if is_valid_extra_deck_card(card, card_info):
                    freq_counter[card] += 1

    # Score candidates with enhanced synergy scoring
    candidate_scores = []
    for candidate in candidates:
        base_score = freq_counter[candidate] if candidate in freq_counter else 0
        synergy_multiplier = score_card_synergy_enhanced(candidate, input_cards, card_info, archetypes)
        final_score = base_score * synergy_multiplier
        candidate_scores.append((candidate, final_score))

    # Sort and take top cards
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    selected_cards = [card for card, _ in candidate_scores[:target_extra]]

    # Fill with type-appropriate fallbacks if needed
    if len(selected_cards) < target_extra:
        return fill_extra_deck_with_fallback(selected_cards, input_cards, full_decks, card_info, target_extra)

    return selected_cards


def fill_extra_deck_with_fallback(extra_deck, input_cards, full_decks, card_info, target_size):
    """
    Fills the extra deck to target size using a fallback mechanism appropriate for
    extra deck cards (using only Fusion, Synchro, XYZ, Link monsters).
    """
    if len(extra_deck) >= target_size:
        return extra_deck[:target_size]

    # Get the distribution of extra deck types
    extra_decks = [deck["extra"] for deck in full_decks if deck["extra"]]
    type_distribution = get_extra_deck_distribution(extra_decks, card_info)

    # Calculate how many cards of each type we want
    remaining = target_size - len(extra_deck)
    desired_counts = {
        t: math.ceil(remaining * prop) for t, prop in type_distribution.items()
    }

    # Count current types in the extra deck
    current_counts = Counter(get_extra_category(card, card_info) for card in extra_deck)

    # Get missing counts
    missing_counts = {
        t: max(0, desired_counts.get(t, 0) - current_counts.get(t, 0))
        for t in type_distribution.keys()
    }

    # Create a fallback pool of cards by type
    fallback_pool = {}
    for card_type in missing_counts.keys():
        type_candidates = []
        for deck in full_decks:
            for card in deck["extra"]:
                if get_extra_category(card, card_info) == card_type and card not in extra_deck:
                    type_candidates.append(card)
        fallback_pool[card_type] = Counter(type_candidates).most_common()

    # Fill the extra deck by type
    for card_type, count in missing_counts.items():
        candidates = fallback_pool.get(card_type, [])
        added = 0
        for candidate, _ in candidates:
            if candidate not in extra_deck and is_valid_extra_deck_card(candidate, card_info):
                extra_deck.append(candidate)
                added += 1
                if added >= count or len(extra_deck) >= target_size:
                    break

    # If still not enough, add any valid extra deck cards
    if len(extra_deck) < target_size:
        all_candidates = []
        for deck in full_decks:
            for card in deck["extra"]:
                if card not in extra_deck and is_valid_extra_deck_card(card, card_info):
                    all_candidates.append(card)

        fallback_counter = Counter(all_candidates)
        for card, _ in fallback_counter.most_common():
            if card not in extra_deck:
                extra_deck.append(card)
                if len(extra_deck) >= target_size:
                    break

    return extra_deck


###################################
# Optimized Side Deck Builder
###################################
def build_side_deck_optimized(input_cards, full_decks, target_side, card_info, archetypes, main_deck, extra_deck):
    """
    Optimized version of the side deck builder with enhanced synergy scoring and better performance.
    """
    # Get filtered candidate pool
    side_candidate_lists = get_filtered_side_decks(input_cards, full_decks)

    # Create pool of valid candidates
    candidates = []
    for deck in side_candidate_lists:
        for card in deck:
            if is_extra_deck_card(card, card_info) and not is_valid_extra_deck_card(card, card_info):
                continue
            if main_deck.count(card) >= MAIN_DUPLICATE_LIMIT or extra_deck.count(card) >= EXTRA_DUPLICATE_LIMIT:
                continue
            candidates.append(card)

    # Unique candidates
    candidates = list(set(candidates))

    # Base frequency counts
    freq_counter = Counter()
    for deck in full_decks:
        if any(card in deck["main"] for card in input_cards):
            for card in deck["side"]:
                if card in candidates:
                    freq_counter[card] += 1

    # Score with enhanced synergy
    candidate_scores = []
    for candidate in candidates:
        base_score = freq_counter[candidate] if candidate in freq_counter else 0
        synergy_multiplier = score_card_synergy_enhanced(candidate, input_cards, card_info, archetypes)
        final_score = base_score * synergy_multiplier
        candidate_scores.append((candidate, final_score))

    # Sort and select top cards
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    selected_cards = [card for card, _ in candidate_scores[:target_side]]

    # Fill with fallback if needed
    if len(selected_cards) < target_side:
        return fill_side_deck_with_fallback(selected_cards, input_cards, full_decks,
                                            card_info, main_deck, extra_deck, target_side)

    return selected_cards


def fill_side_deck_with_fallback(side_deck, input_cards, full_decks, card_info, new_main_deck, new_extra_deck,
                                 target_size):
    """
    Fills the side deck to target size using appropriate fallback mechanism.
    Ensures valid cards and proper representation of counters to common strategies.
    """
    if len(side_deck) >= target_size:
        return side_deck[:target_size]

    # Create a fallback pool from common side deck cards
    fallback_candidates = []
    side_freq = Counter()

    for deck in full_decks:
        for card in deck["side"]:
            # Ensure extra deck cards are valid
            if is_extra_deck_card(card, card_info) and not is_valid_extra_deck_card(card, card_info):
                continue
            side_freq[card] += 1

    # Get most common side deck cards that aren't in our main or extra deck
    for card, _ in side_freq.most_common(target_size * 3):
        # Skip if already at duplicate limit in main or extra deck
        if new_main_deck.count(card) >= MAIN_DUPLICATE_LIMIT or new_extra_deck.count(card) >= EXTRA_DUPLICATE_LIMIT:
            continue
        # Skip if already in side deck
        if card in side_deck:
            continue
        fallback_candidates.append(card)

    # Fill up to target size
    while len(side_deck) < target_size and fallback_candidates:
        side_deck.append(fallback_candidates.pop(0))

    return side_deck


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
    # Load card info and archetypes
    print("Loading card data...")
    card_info = load_card_info()
    with open(ARCHETYPES_FILE, "r", encoding="utf-8") as f:
        archetypes = json.load(f)

    # Clear any previous cache
    clear_cache()

    # Load training decks
    print("Loading deck data...")
    full_decks = load_full_decks(YDK_FOLDER)
    print(f"Loaded {len(full_decks)} decks.")

    # Separate decks by section
    main_decks = [deck["main"] for deck in full_decks]
    extra_decks = [deck["extra"] for deck in full_decks if deck["extra"]]
    side_decks = [deck["side"] for deck in full_decks if deck["side"]]

    # Mine association rules - reusing existing code
    print("Mining association rules...")
    df = build_transaction_df(main_decks)
    freq_itemsets, rules = mine_rules(df)
    print(f"Mined {len(rules)} association rules for main deck.")
    model = {
        "frequent_itemsets": convert_frozensets(freq_itemsets.to_dict(orient="records")),
        "association_rules": convert_frozensets(rules.to_dict(orient="records"))
    }
    with open(TRAINED_MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"Trained model saved to {TRAINED_MODEL_FILE}")

    # Get user input
    user_input_main = input("Enter base MAIN deck card IDs (separated by commas): ")
    input_main_cards = [card.strip() for card in user_input_main.split(",") if card.strip()]

    user_input_extra = input("Enter base EXTRA deck card IDs (separated by commas, if any): ")
    input_extra_cards = [card.strip() for card in user_input_extra.split(",") if card.strip()]

    user_input_side = input("Enter base SIDE deck card IDs (separated by commas, if any): ")
    input_side_cards = [card.strip() for card in user_input_side.split(",") if card.strip()]

    if not input_main_cards:
        print("No base MAIN deck cards provided. Exiting.")
        return

    # Detect archetypes using cached function
    input_archetypes = set()
    for card in input_main_cards:
        archs = detect_archetypes_cached(card, card_info, archetypes)
        input_archetypes.update(archs)

    print(f"Detected archetypes in input cards: {input_archetypes}")

    # Calculate recommendation scores with enhanced synergy
    print("Generating recommendations...")
    # Get baseline recommendations
    context_recs = recommend_main_deck_contextual(input_main_cards, main_decks, top_n=30, min_decks=5)
    freq_recs = recommend_main_deck_by_input(input_main_cards, main_decks, top_n=30)
    combined_recs = combine_recommendations(context_recs, freq_recs)

    # Apply enhanced synergy scoring
    boosted_recs = []
    for card, score in combined_recs:
        synergy_multiplier = score_card_synergy_enhanced(card, input_main_cards, card_info, archetypes)
        new_score = score * synergy_multiplier
        boosted_recs.append((card, new_score))

    # Sort and filter
    boosted_recs.sort(key=lambda x: x[1], reverse=True)
    filtered_boosted_recs = filter_synergy_candidates(boosted_recs, card_info)

    # Show top recommendations
    print("Top main deck recommendations:")
    for card, score in filtered_boosted_recs[:10]:
        card_name = card_info[card]['name'] if card in card_info else f"Unknown ({card})"
        print(f"Card {card_name} (ID: {card}) - Score: {score:.2f}")

    # Get average copies
    avg_main = compute_average_copies_main(main_decks)

    # Start with input cards
    new_main_deck = list(input_main_cards)

    # Calculate dynamic distribution based on archetypes
    print("Analyzing deck archetype distribution...")
    desired_distribution = analyze_archetype_distribution(
        input_main_cards, card_info, archetypes, main_decks, target_size=TARGET_MAX_MAIN
    )

    print(f"Determined optimal distribution for this archetype: {desired_distribution}")

    # Build main deck with flexible distribution
    new_main_deck = apply_flexible_distribution(
        new_main_deck, main_decks, card_info, desired_distribution,
        avg_main, input_main_cards, archetypes
    )

    print(f"Main deck constructed with {len(new_main_deck)} cards.")

    # Print main deck category distribution
    main_categories = Counter(get_card_category(card, card_info) for card in new_main_deck)
    print(f"Main deck distribution: {dict(main_categories)}")

    # Extra deck with enhanced synergy
    valid_input_extra = [card for card in input_extra_cards if is_valid_extra_deck_card(card, card_info)]
    if len(valid_input_extra) != len(input_extra_cards):
        invalid_count = len(input_extra_cards) - len(valid_input_extra)
        print(f"Warning: {invalid_count} invalid extra deck cards were removed.")

    # Build optimized extra deck
    all_input_cards = input_main_cards + valid_input_extra
    new_extra_deck = build_extra_deck_optimized(all_input_cards, full_decks, TARGET_EXTRA_SIZE, card_info, archetypes)

    print(f"Extra deck constructed with {len(new_extra_deck)} cards.")

    # Show extra deck distribution
    extra_types = Counter(get_extra_category(card, card_info) for card in new_extra_deck)
    print(f"Extra deck type distribution: {dict(extra_types)}")

    # Print extra deck cards
    print("Extra deck cards:")
    for card in new_extra_deck:
        card_name = card_info[card]['name'] if card in card_info else f"Unknown ({card})"
        card_type = get_extra_category(card, card_info)
        print(f"- {card_name} (Type: {card_type})")

    # Side deck with enhanced synergy
    valid_input_side = [card for card in input_side_cards
                        if not is_extra_deck_card(card, card_info) or is_valid_extra_deck_card(card, card_info)]

    if len(valid_input_side) != len(input_side_cards):
        invalid_count = len(input_side_cards) - len(valid_input_side)
        print(f"Warning: {invalid_count} invalid side deck cards were removed.")

    # Start with valid input side cards
    new_side_deck = valid_input_side[:]

    # Build optimized side deck
    if len(new_side_deck) < TARGET_SIDE_SIZE:
        side_cards = build_side_deck_optimized(
            all_input_cards, full_decks, TARGET_SIDE_SIZE - len(new_side_deck),
            card_info, archetypes, new_main_deck, new_extra_deck
        )
        new_side_deck.extend([card for card in side_cards if card not in new_side_deck])
        new_side_deck = new_side_deck[:TARGET_SIDE_SIZE]

    print(f"Side deck constructed with {len(new_side_deck)} cards.")

    # Show side deck cards
    side_categories = Counter()
    print("Side deck cards:")
    for card in new_side_deck:
        if is_extra_deck_card(card, card_info):
            category = get_extra_category(card, card_info)
        else:
            category = get_card_category(card, card_info)
        side_categories[category] += 1

        card_name = card_info[card]['name'] if card in card_info else f"Unknown ({card})"
        print(f"- {card_name} (Type: {category})")

    print(f"Side deck category distribution: {dict(side_categories)}")

    # Save the final deck
    new_deck = {"main": new_main_deck, "extra": new_extra_deck, "side": new_side_deck}
    print("New deck built!")
    write_ydk(new_deck, OUTPUT_DECK_FILE)

    # Final archetype analysis
    final_archetypes = set()
    for card in new_main_deck + new_extra_deck:
        archs = detect_archetypes_cached(card, card_info, archetypes)
        final_archetypes.update(archs)

    print(f"Archetypes in the final deck: {final_archetypes}")
    print(f"Deck saved to {OUTPUT_DECK_FILE}")


if __name__ == "__main__":
    main()