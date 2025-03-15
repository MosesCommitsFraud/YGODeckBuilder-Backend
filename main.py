import os
import json
import re
from collections import Counter, defaultdict
import math
import requests

###################################
# Configuration and Constants
###################################
# File and folder settings
CARDINFO_FILE = "cardinfo.json"  # Local file for card information
ARCHETYPES_FILE = "archetypes.json"  # JSON file containing archetypes
YDK_FOLDER = "ydk_download"  # Folder containing .ydk deck files
OUTPUT_DECK_FILE = "generated_deck.ydk"  # Output file for the generated deck

# Target deck sizes
TARGET_MIN_MAIN = 40  # Minimum number of cards in the Main Deck
TARGET_MAX_MAIN = 60  # Maximum number of cards in the Main Deck
TARGET_EXTRA_SIZE = 15  # Desired size of the Extra Deck
TARGET_SIDE_SIZE = 15  # Desired size of the Side Deck

# Synergy scoring factors
SYNERGY_BOOST_FACTOR = 2.0  # Base multiplier for synergy
ARCHETYPE_BOOST = 2.5  # Boost for archetype matches
REFERENCE_BOOST = 3.0  # Boost for cards that reference each other
REQUIRED_CARD_BOOST = 5.0  # Boost for cards required by Extra Deck monsters

# Duplicate filtering in side deck
MAIN_DUPLICATE_LIMIT = 3  # Filter out if it appears 3+ times in main deck
EXTRA_DUPLICATE_LIMIT = 1  # Filter out if it appears in extra deck

# Caching system for expensive operations
CACHE = {
    'archetypes': {},  # card_id -> detected archetypes
    'references': {},  # card_id -> referenced card_ids
    'synergy_score': {},  # (card_id, frozenset(context_cards)) -> score
    'name_to_id': None,  # Card name lookup (created only once)
    'required_cards': {}  # card_id -> required card IDs for Extra Deck monsters
}


def clear_cache():
    """Clears all cached values"""
    CACHE['archetypes'].clear()
    CACHE['references'].clear()
    CACHE['synergy_score'].clear()
    CACHE['name_to_id'] = None
    CACHE['required_cards'].clear()


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


def is_tuner(card_id, card_info):
    """Check if a card is a Tuner monster."""
    if card_id not in card_info:
        return False

    card_type = card_info[card_id].get("type", "").lower()
    return "tuner" in card_type


###################################
# Archetype Detection
###################################
def detect_archetypes(card_id, card_info, archetypes):
    """
    Detects archetypes in a card's name and description using string matching.
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

        if (
                # Check for complete word match
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


def extract_archetype_mentions(card_desc, archetypes):
    """
    Extract archetypes mentioned in a card description.
    """
    mentioned_archetypes = set()
    desc = card_desc.lower()

    # Look for archetype patterns in quotes
    quote_pattern = re.compile(r'["\']([^"\']+)["\']')
    quoted_terms = quote_pattern.findall(desc)

    # Check each quoted term against archetypes
    for term in quoted_terms:
        term_lower = term.lower()
        for archetype in archetypes:
            archetype_lower = archetype.lower().strip('"')
            if archetype_lower == term_lower or f"{archetype_lower}-" in term_lower:
                mentioned_archetypes.add(archetype_lower)

    # Look for common patterns that indicate archetype mentions
    archetype_patterns = [
        r'(["\'][^"\']+["\'])\s+monster',
        r'(["\'][^"\']+["\'])\s+card',
        r'(["\'][^"\']+["\'])\s+spell',
        r'(["\'][^"\']+["\'])\s+trap'
    ]

    for pattern in archetype_patterns:
        matches = re.findall(pattern, desc)
        for match in matches:
            clean_match = match.strip('"\'')
            for archetype in archetypes:
                archetype_lower = archetype.lower().strip('"')
                if archetype_lower == clean_match.lower() or archetype_lower in clean_match.lower():
                    mentioned_archetypes.add(archetype_lower)

    return mentioned_archetypes


###################################
# Card Reference Detection
###################################
def build_card_name_lookup(card_info):
    """
    Builds a lookup dictionary for card names.
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


def extract_referenced_card_ids_improved(card_id, card_info, name_to_id=None):
    """
    Extract referenced card IDs from a card's description.
    Returns a set of card IDs that are referenced.
    """
    if card_id in CACHE['references']:
        return CACHE['references'][card_id]

    if card_id not in card_info:
        return set()

    if name_to_id is None:
        name_to_id = build_card_name_lookup(card_info)

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

    # Cache and return the result
    CACHE['references'][card_id] = referenced_ids
    return referenced_ids


def extract_referenced_card_ids_advanced(card_id, card_info, name_to_id=None):
    """
    Enhanced version of extract_referenced_card_ids_improved that catches more references.
    Handles quoted names and specific patterns in card descriptions.
    """
    # Build name lookup if needed
    if name_to_id is None:
        name_to_id = build_card_name_lookup(card_info)

    # Start with the original function's results
    references = extract_referenced_card_ids_improved(card_id, card_info, name_to_id)

    # If the card isn't in card_info, just return the base references
    if card_id not in card_info:
        return references

    # Enhanced pattern matching for card text
    desc = card_info[card_id].get('desc', '').lower()
    if not desc:
        return references

    # Additional references we find
    additional_references = set()

    # Look for quoted card names that might be missed by the base function
    quoted_names = re.findall(r'"([^"]+)"', desc)
    quoted_names.extend(re.findall(r"'([^']+)'", desc))

    for name in quoted_names:
        name_lower = name.lower().strip()

        # Check for exact matches in the name lookup
        for card_name, cid in name_to_id.items():
            if name_lower == card_name.lower():
                additional_references.add(cid)
                break

    # Combine with base references
    all_references = references.union(additional_references)
    return all_references


def extract_extra_deck_requirements(card_id, card_info, name_to_id=None):
    """
    Extract cards that are explicitly required by an Extra Deck monster.
    Focuses on finding quoted card names and key patterns based on monster type.
    """
    if card_id in CACHE['required_cards']:
        return CACHE['required_cards'][card_id]

    if card_id not in card_info or not is_extra_deck_card(card_id, card_info):
        CACHE['required_cards'][card_id] = []
        return []

    # Build name lookup if needed
    if name_to_id is None:
        name_to_id = build_card_name_lookup(card_info)

    card_data = card_info[card_id]
    card_desc = card_data.get('desc', '').lower()
    card_type = card_data.get('type', '').lower()

    required_cards = []

    # Look for quoted card names which are often specific required materials
    quoted_names = re.findall(r'"([^"]+)"', card_desc)
    quoted_names.extend(re.findall(r"'([^']+)'", card_desc))

    for name in quoted_names:
        name_lower = name.lower().strip()

        # Check for exact matches in the name lookup
        for card_name, cid in name_to_id.items():
            if name_lower == card_name.lower():
                if cid not in required_cards:
                    required_cards.append(cid)
                break

    # For Fusion monsters, also check for "+" pattern
    if "fusion" in card_type and "+" in card_desc:
        # First try to find the fusion materials section
        materials_section = card_desc
        for marker in ["fusion material", "materials:", "material:"]:
            if marker in card_desc:
                parts = card_desc.split(marker, 1)
                if len(parts) > 1:
                    materials_section = parts[1].split(".", 1)[0]
                    break

        # Look for the "+" pattern which is common in fusion requirements
        if "+" in materials_section:
            parts = materials_section.split("+")
            for part in parts:
                clean_part = part.strip().lower()
                for card_name, cid in name_to_id.items():
                    card_name_lower = card_name.lower()
                    if clean_part == card_name_lower or (len(clean_part) > 4 and clean_part in card_name_lower):
                        if cid not in required_cards:
                            required_cards.append(cid)

    # For Synchro monsters, check for tuners
    elif "synchro" in card_type and "tuner" in card_desc:
        # Look for tuner levels
        tuner_mentioned = "tuner" in card_desc
        level_patterns = re.findall(r'(\d+)\s+level', card_desc)

        # If we found a tuner reference but no specific tuner, add some generic tuners
        if tuner_mentioned and not required_cards:
            # Find some appropriate tuners (up to 2)
            tuner_count = 0
            for cid, cdata in card_info.items():
                if tuner_count >= 2:
                    break

                if is_extra_deck_card(cid, card_info):
                    continue  # Skip extra deck cards as materials

                if "tuner" in cdata.get("type", "").lower():
                    if cid not in required_cards:
                        required_cards.append(cid)
                        tuner_count += 1

    # For XYZ monsters, check for level requirements
    elif "xyz" in card_type:
        level_patterns = re.findall(r'(\d+)\s+level', card_desc)

        # If we found level requirements, get some matching monsters
        if level_patterns and not required_cards:
            try:
                level = int(level_patterns[0])
                # Find a few monsters with the required level (up to 2)
                level_count = 0
                for cid, cdata in card_info.items():
                    if level_count >= 2:
                        break

                    if is_extra_deck_card(cid, card_info):
                        continue  # Skip extra deck cards as materials

                    card_level = cdata.get("level", 0)
                    if card_level == level:
                        if cid not in required_cards:
                            required_cards.append(cid)
                            level_count += 1
            except (ValueError, IndexError):
                pass

    # Cache results
    CACHE['required_cards'][card_id] = required_cards
    return required_cards


def ensure_required_cards_in_main(main_deck, extra_deck, card_info, max_additions=5):
    """
    Ensure main deck contains cards required by Extra Deck monsters.
    Limits additions to avoid bloating the deck.
    """
    if not extra_deck:  # No extra deck to analyze
        return main_deck

    name_to_id = build_card_name_lookup(card_info)
    all_required = []

    # Get required cards for each Extra Deck monster
    for card_id in extra_deck:
        required = extract_extra_deck_requirements(card_id, card_info, name_to_id)
        all_required.extend(required)

    # Filter to cards not already in main deck and that aren't Extra Deck cards themselves
    missing_required = [
        card for card in all_required
        if card not in main_deck and not is_extra_deck_card(card, card_info)
    ]

    # Limit to a reasonable number of additions
    if missing_required:
        # Add up to max_additions required cards
        for card in missing_required[:max_additions]:
            if len(main_deck) < TARGET_MAX_MAIN:
                main_deck.append(card)
                if card in card_info:
                    print(f"Added required card: {card_info[card]['name']} for Extra Deck support")

    return main_deck


###################################
# Synergy Scoring
###################################
def score_card_synergy_improved(candidate_id, input_cards, card_info, archetypes):
    """
    Enhanced scoring function that considers if a card is required by Extra Deck monsters.
    """
    if candidate_id not in card_info:
        return 1.0

    # Start with base score multiplier
    score_multiplier = 1.0

    # Check for archetype matches
    input_archetypes = set()
    for card in input_cards:
        if card in card_info:
            card_archetypes = detect_archetypes_cached(card, card_info, archetypes)
            input_archetypes.update(card_archetypes)

    candidate_archetypes = detect_archetypes_cached(candidate_id, card_info, archetypes)
    if candidate_archetypes and input_archetypes and candidate_archetypes & input_archetypes:
        score_multiplier *= ARCHETYPE_BOOST

    # Build name lookup for reference detection
    name_to_id = build_card_name_lookup(card_info)

    # Check if this card is referenced by any input card
    for card in input_cards:
        refs = extract_referenced_card_ids_advanced(card, card_info, name_to_id)
        if candidate_id in refs:
            score_multiplier *= REFERENCE_BOOST
            break

    # Check if this card references any input card
    refs = extract_referenced_card_ids_advanced(candidate_id, card_info, name_to_id)
    if any(ref in input_cards for ref in refs):
        score_multiplier *= REFERENCE_BOOST

    # Check if this card is required by any Extra Deck monster in the input
    extra_deck_cards = [card for card in input_cards if is_extra_deck_card(card, card_info)]
    for card in extra_deck_cards:
        required_cards = extract_extra_deck_requirements(card, card_info, name_to_id)
        if candidate_id in required_cards:
            score_multiplier *= REQUIRED_CARD_BOOST
            break

    return score_multiplier


def filter_and_sort_candidates(candidates, input_cards, card_info, archetypes):
    """
    Filter and sort candidate cards based on improved synergy scoring.
    Returns a list of (card_id, score) tuples sorted by score.
    """
    scored_candidates = []
    for card in candidates:
        if card in input_cards:
            continue

        score = score_card_synergy_improved(card, input_cards, card_info, archetypes)
        scored_candidates.append((card, score))

    # Sort by score (highest first)
    return sorted(scored_candidates, key=lambda x: x[1], reverse=True)


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


def get_filtered_extra_decks(input_cards, full_decks):
    """Returns extra deck lists from decks where the main deck contains an input card."""
    return [deck["extra"] for deck in full_decks if deck["extra"] and any(card in deck["main"] for card in input_cards)]


def get_filtered_side_decks(input_cards, full_decks):
    """Returns side deck lists from decks where the main deck contains an input card."""
    return [deck["side"] for deck in full_decks if deck["side"] and any(card in deck["main"] for card in input_cards)]


###################################
# Deck Distribution Analysis
###################################
def compute_average_copies_main(main_decks):
    """Computes average copies (capped at 3) for main deck cards."""
    total, cnt = defaultdict(int), defaultdict(int)
    for deck in main_decks:
        for card, copies in Counter(deck).items():
            total[card] += copies
            cnt[card] += 1
    return {card: min(3, round(total[card] / cnt[card])) for card in total}


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

    # Get all main deck cards as candidates
    all_candidates = set()
    for deck in main_decks:
        all_candidates.update(deck)

    # Score candidates and filter out extra deck cards
    valid_candidates = [c for c in all_candidates if not is_extra_deck_card(c, card_info)]
    scored_candidates = filter_and_sort_candidates(valid_candidates, input_cards, card_info, archetypes)

    # Group candidates by category
    candidates_by_category = defaultdict(list)
    for card, score in scored_candidates:
        if card in new_main_deck:
            continue
        category = get_card_category(card, card_info)
        candidates_by_category[category].append((card, score))

    # First pass: Add high-synergy cards regardless of distribution
    if synergy_priority:
        high_synergy_candidates = [(c, s) for c, s in scored_candidates if s > 2.0]
        for card, score in high_synergy_candidates:
            if card not in new_main_deck and len(new_main_deck) < TARGET_MAX_MAIN:
                category = get_card_category(card, card_info)
                # Add even if we're a bit over on this category
                if current[category] < target_distribution[category] * 1.3:  # Allow 30% over target
                    copies = avg_main.get(card, 1)
                    for _ in range(copies):
                        if len(new_main_deck) < TARGET_MAX_MAIN:
                            new_main_deck.append(card)
                            current[category] += 1

    # Second pass: Fill remaining slots with attention to distribution
    for category, target in target_distribution.items():
        # Add cards until we reach the target or run out of candidates
        cat_candidates = candidates_by_category[category]

        for card, _ in cat_candidates:
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
        remaining_candidates = [c for c, _ in scored_candidates if c not in new_main_deck]

        while len(new_main_deck) < TARGET_MIN_MAIN and remaining_candidates:
            new_main_deck.append(remaining_candidates.pop(0))

    return new_main_deck


###################################
# Deck Building Functions
###################################
def preserve_input_cards(input_cards, target_deck, card_info, is_extra=False):
    """
    Ensure all valid input cards are preserved in the target deck.
    """
    for card in input_cards:
        # Skip invalid cards based on deck type
        if is_extra and not is_valid_extra_deck_card(card, card_info):
            continue

        if not is_extra and is_extra_deck_card(card, card_info):
            continue

        # Add if not already in the deck
        if card not in target_deck:
            target_deck.append(card)

    return target_deck


def build_main_deck_improved(input_cards, main_decks, card_info, archetypes, avg_main=None):
    """
    Build an improved main deck that ensures input cards are preserved
    and focuses on synergy rather than popularity.
    """
    # Start with input cards that belong in main deck
    valid_input_main = [card for card in input_cards if not is_extra_deck_card(card, card_info)]
    new_main_deck = list(valid_input_main)

    # Compute average copies if not provided
    if avg_main is None:
        avg_main = compute_average_copies_main(main_decks)

    # Determine desired distribution based on archetypes
    desired_distribution = analyze_archetype_distribution(
        input_cards, card_info, archetypes, main_decks, target_size=TARGET_MAX_MAIN
    )

    # Process candidates for inclusion
    if valid_input_main:
        # Get all cards from main decks as candidates
        all_candidates = set()
        for deck in main_decks:
            all_candidates.update(deck)

        # Score candidates based on synergy with input cards
        scored_candidates = filter_and_sort_candidates(
            all_candidates, valid_input_main, card_info, archetypes
        )

        # Add high-synergy cards first
        for card, score in scored_candidates:
            if card in new_main_deck:
                continue

            if is_extra_deck_card(card, card_info):
                continue

            # Determine number of copies based on score and average
            if card in avg_main:
                base_copies = avg_main[card]
            else:
                base_copies = 1

            # Adjust copies based on synergy score
            if score > 3.0:
                copies = min(3, base_copies + 1)  # Very high synergy - add an extra copy
            elif score > 1.5:
                copies = base_copies  # Good synergy - use average copies
            else:
                copies = max(1, base_copies - 1)  # Lower synergy - reduce copies

            # Add copies up to the limit
            for _ in range(copies):
                if len(new_main_deck) < TARGET_MAX_MAIN:
                    new_main_deck.append(card)

    # If we have enough cards, we're done
    if len(new_main_deck) >= TARGET_MIN_MAIN:
        return new_main_deck

    # Otherwise, use flexible distribution to complete the deck
    completed_deck = apply_flexible_distribution(
        new_main_deck, main_decks, card_info, desired_distribution,
        avg_main, input_cards, archetypes, synergy_priority=True
    )

    # Final pass to ensure input cards are included
    completed_deck = preserve_input_cards(valid_input_main, completed_deck, card_info)

    return completed_deck


###################################
# Optimized Fallback Functions
###################################
def find_generic_extra_deck_candidates_optimized(main_deck, card_info, max_candidates=50):
    """
    Optimized version to find generic extra deck monsters compatible with the main deck.
    Limits search scope to prevent timeouts.
    """
    candidates = set()

    # Quick analysis of main deck properties
    main_deck_levels = {}
    main_deck_types = {}
    has_tuners = False

    # Limit analysis to first 20 cards to save time
    for card in main_deck[:20]:
        if card in card_info:
            card_data = card_info[card]

            # Check if this is a tuner
            if "tuner" in card_data.get("type", "").lower():
                has_tuners = True

            # Track levels for xyz/synchro matching
            level = card_data.get("level", 0)
            if level > 0:
                main_deck_levels[level] = main_deck_levels.get(level, 0) + 1

            # Track monster types
            card_type = card_data.get("race", "")
            if card_type:
                main_deck_types[card_type] = main_deck_types.get(card_type, 0) + 1

    # Get most common levels and types
    common_levels = sorted(main_deck_levels.items(), key=lambda x: x[1], reverse=True)[:3]
    common_types = sorted(main_deck_types.items(), key=lambda x: x[1], reverse=True)[:2]

    # Create a smaller subset of cards to search through
    # This dramatically reduces processing time
    sample_size = 1000  # Limit to checking 1000 cards max
    card_sample = list(card_info.items())[:sample_size]

    # Find compatible extra deck monsters
    candidate_count = 0
    for card_id, card_data in card_sample:
        # Stop if we have enough candidates
        if candidate_count >= max_candidates:
            break

        if not is_valid_extra_deck_card(card_id, card_info):
            continue

        card_type = card_data.get("type", "").lower()
        added = False

        # Check for generic Extra Deck monsters
        desc = card_data.get("desc", "").lower()
        if ("fusion" in card_type and "fusion material" in desc and "monster" in desc) or \
                ("synchro" in card_type and has_tuners and "1 tuner" in desc) or \
                ("xyz" in card_type and any(level[0] == card_data.get("level", 0) for level in common_levels)) or \
                ("link" in card_type and "2+ monster" in desc):
            candidates.add(card_id)
            candidate_count += 1
            added = True

        # If not added yet, check type matches
        if not added:
            card_race = card_data.get("race", "")
            if card_race and any(card_race == common_type[0] for common_type in common_types):
                candidates.add(card_id)
                candidate_count += 1

    return list(candidates)


def find_generic_side_deck_candidates_optimized(main_deck, card_info, max_candidates=50):
    """
    Optimized version to find generic side deck cards.
    Limits search scope to prevent timeouts.
    """
    candidates = set()
    staple_candidates = set()

    # Quick analysis of main deck
    monsters = 0
    spells = 0
    traps = 0

    # Limit analysis to first 20 cards to save time
    for card in main_deck[:20]:
        if card in card_info:
            category = get_card_category(card, card_info)
            if category == "Monster":
                monsters += 1
            elif category == "Spell":
                spells += 1
            else:
                traps += 1

    # Determine if deck is monster-heavy or spell/trap-heavy
    total = monsters + spells + traps
    monster_ratio = monsters / total if total > 0 else 0.5

    # Create a smaller subset of cards to search through
    sample_size = 1000  # Limit to checking 1000 cards max
    card_sample = list(card_info.items())[:sample_size]

    # Include some general staples based on deck type
    candidate_count = 0
    for card_id, card_data in card_sample:
        # Stop if we have enough candidates
        if candidate_count >= max_candidates:
            break

        if is_extra_deck_card(card_id, card_info):
            continue

        category = get_card_category(card_id, card_info)
        desc = card_data.get("desc", "").lower()

        # Generic staples for any deck (high priority)
        if any(term in desc for term in ["negate", "destroy", "banish"]):
            if category == "Monster" and "hand" in desc and "discard" in desc:
                staple_candidates.add(card_id)  # Hand traps
                candidate_count += 1
            elif category == "Spell" and "quick" in desc:
                staple_candidates.add(card_id)  # Quick-play spells
                candidate_count += 1
            elif category == "Trap" and "counter" in desc:
                staple_candidates.add(card_id)  # Counter traps
                candidate_count += 1

        # For monster-heavy decks, add board wipes
        if monster_ratio > 0.6 and category == "Spell" and "destroy all" in desc:
            candidates.add(card_id)
            candidate_count += 1

        # For spell/trap-heavy decks, add protection
        if monster_ratio < 0.4 and "cannot be target" in desc:
            candidates.add(card_id)
            candidate_count += 1

    # Combine and return results
    all_candidates = list(staple_candidates) + list(candidates)
    return all_candidates[:max_candidates]


###################################
# Optimized Extra & Side Deck Builders
###################################
def build_extra_deck_improved_optimized(input_cards, full_decks, card_info, archetypes, main_deck=None):
    """
    Optimized extra deck builder with streamlined fallback mechanism.
    """
    # Start with valid input extra deck cards
    valid_input_extra = [card for card in input_cards if is_valid_extra_deck_card(card, card_info)]
    new_extra_deck = list(valid_input_extra)

    # If we already have enough cards, we're done
    if len(new_extra_deck) >= TARGET_EXTRA_SIZE:
        return new_extra_deck[:TARGET_EXTRA_SIZE]

    # Use original method first
    filtered_extra = get_filtered_extra_decks(input_cards, full_decks)

    # Check if we can use the normal method
    if filtered_extra:
        # Build the deck using the original method
        candidates = set()
        for deck in filtered_extra:
            for card in deck:
                if is_valid_extra_deck_card(card, card_info) and card not in new_extra_deck:
                    candidates.add(card)

        # Score and add candidates
        synergy_context = input_cards + (main_deck if main_deck else [])
        scored = filter_and_sort_candidates(list(candidates), synergy_context, card_info, archetypes)

        for card, _ in scored:
            if len(new_extra_deck) < TARGET_EXTRA_SIZE:
                new_extra_deck.append(card)

    # If we still need more cards, use the fallback
    if len(new_extra_deck) < TARGET_EXTRA_SIZE and main_deck:
        print("Using optimized fallback for extra deck")
        generic_cards = find_generic_extra_deck_candidates_optimized(main_deck, card_info)

        # Filter out cards already in the deck
        generic_cards = [c for c in generic_cards if c not in new_extra_deck]

        # Add generic cards
        for card in generic_cards:
            if len(new_extra_deck) < TARGET_EXTRA_SIZE:
                new_extra_deck.append(card)
                if card in card_info:
                    print(f"Added fallback extra deck card: {card_info[card].get('name', card)}")

    # Final verification of input card preservation
    new_extra_deck = preserve_input_cards(valid_input_extra, new_extra_deck, card_info, is_extra=True)

    return new_extra_deck[:TARGET_EXTRA_SIZE]


def build_side_deck_improved_optimized(input_cards, full_decks, card_info, archetypes,
                                       main_deck=None, extra_deck=None):
    """
    Optimized side deck builder with streamlined fallback mechanism.
    """
    # Start with valid input side deck cards
    valid_input_side = [card for card in input_cards
                        if not is_extra_deck_card(card, card_info) or
                        is_valid_extra_deck_card(card, card_info)]
    new_side_deck = list(valid_input_side)

    # If we already have enough cards, we're done
    if len(new_side_deck) >= TARGET_SIDE_SIZE:
        return new_side_deck[:TARGET_SIDE_SIZE]

    # Use original method first
    filtered_side = get_filtered_side_decks(input_cards, full_decks)

    # Check if we can use the normal method
    if filtered_side:
        # Build the deck using the original method
        candidates = set()
        for deck in filtered_side:
            for card in deck:
                # Skip cards that are already in main or extra
                if (main_deck and card in main_deck) or (extra_deck and card in extra_deck):
                    continue

                if card not in new_side_deck:
                    candidates.add(card)

        # Score and add candidates
        synergy_context = input_cards + (main_deck if main_deck else []) + (extra_deck if extra_deck else [])
        scored = filter_and_sort_candidates(list(candidates), synergy_context, card_info, archetypes)

        for card, _ in scored:
            if len(new_side_deck) < TARGET_SIDE_SIZE:
                new_side_deck.append(card)

    # If we still need more cards, use the fallback
    if len(new_side_deck) < TARGET_SIDE_SIZE and main_deck:
        print("Using optimized fallback for side deck")
        generic_cards = find_generic_side_deck_candidates_optimized(main_deck, card_info)

        # Filter out cards already in any deck part
        filtered_cards = []
        for card in generic_cards:
            if card in new_side_deck:
                continue
            if main_deck and card in main_deck:
                continue
            if extra_deck and card in extra_deck:
                continue
            filtered_cards.append(card)

        # Add generic cards
        for card in filtered_cards:
            if len(new_side_deck) < TARGET_SIDE_SIZE:
                new_side_deck.append(card)
                if card in card_info:
                    print(f"Added fallback side deck card: {card_info[card].get('name', card)}")

    # Final verification of input card preservation
    new_side_deck = preserve_input_cards(valid_input_side, new_side_deck, card_info)

    return new_side_deck[:TARGET_SIDE_SIZE]


def build_complete_deck_improved(input_cards, full_decks, card_info, archetypes):
    """
    Build a complete deck with improved synergy detection and input preservation.
    Uses optimized fallback mechanisms to prevent timeouts with uncommon cards.
    """
    print(f"Building complete deck with {len(input_cards)} input cards...")

    # Extract main decks for average copy calculation
    main_decks = [deck["main"] for deck in full_decks]

    # Get average copies for main deck cards
    avg_main = compute_average_copies_main(main_decks)

    # Build main deck with improved synergy
    print("Building main deck...")
    new_main_deck = build_main_deck_improved(
        input_cards, main_decks, card_info, archetypes, avg_main
    )
    print(f"Main deck built with {len(new_main_deck)} cards")

    # Build extra deck with optimized fallback
    print("Building extra deck...")
    new_extra_deck = build_extra_deck_improved_optimized(
        input_cards, full_decks, card_info, archetypes, new_main_deck
    )
    print(f"Extra deck built with {len(new_extra_deck)} cards")

    # Ensure main deck contains cards required by Extra Deck monsters
    print("Ensuring main deck contains required support cards...")
    new_main_deck = ensure_required_cards_in_main(
        new_main_deck, new_extra_deck, card_info
    )
    print(f"Main deck adjusted to {len(new_main_deck)} cards")

    # Build side deck with optimized fallback
    print("Building side deck...")
    new_side_deck = build_side_deck_improved_optimized(
        input_cards, full_decks, card_info, archetypes,
        new_main_deck, new_extra_deck
    )
    print(f"Side deck built with {len(new_side_deck)} cards")

    # Report status
    if len(new_extra_deck) < TARGET_EXTRA_SIZE:
        print(f"Note: Extra deck contains {len(new_extra_deck)}/{TARGET_EXTRA_SIZE} cards")
    if len(new_side_deck) < TARGET_SIDE_SIZE:
        print(f"Note: Side deck contains {len(new_side_deck)}/{TARGET_SIDE_SIZE} cards")

    print("Deck building completed!")
    return {
        "main": new_main_deck,
        "extra": new_extra_deck,
        "side": new_side_deck
    }


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

    # Get user input
    user_input_main = input("Enter base MAIN deck card IDs (separated by commas): ")
    input_main_cards = [card.strip() for card in user_input_main.split(",") if card.strip()]

    user_input_extra = input("Enter base EXTRA deck card IDs (separated by commas, if any): ")
    input_extra_cards = [card.strip() for card in user_input_extra.split(",") if card.strip()]

    user_input_side = input("Enter base SIDE deck card IDs (separated by commas, if any): ")
    input_side_cards = [card.strip() for card in user_input_side.split(",") if card.strip()]

    if not input_main_cards and not input_extra_cards and not input_side_cards:
        print("No base cards provided. Exiting.")
        return

    # Combine all input cards
    all_input_cards = input_main_cards + input_extra_cards + input_side_cards

    # Build complete deck with improved synergy
    new_deck = build_complete_deck_improved(all_input_cards, full_decks, card_info, archetypes)

    # Save the generated deck
    write_ydk(new_deck, OUTPUT_DECK_FILE)
    print(
        f"Generated deck with {len(new_deck['main'])} main, {len(new_deck['extra'])} extra, and {len(new_deck['side'])} side cards.")

    # Print some information about the main deck
    main_categories = Counter(get_card_category(card, card_info) for card in new_deck["main"])
    print(f"Main deck distribution: {dict(main_categories)}")

    # Print extra deck types
    extra_types = Counter(get_extra_category(card, card_info) for card in new_deck["extra"])
    print(f"Extra deck types: {dict(extra_types)}")


if __name__ == "__main__":
    main()