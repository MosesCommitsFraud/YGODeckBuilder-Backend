from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
import traceback
import re
import numpy as np
from collections import Counter, defaultdict

# Import the functions from main.py - make sure main.py is in the same directory
# or adjust the import path as needed
from main import (
    load_card_info as original_load_card_info,
    detect_archetypes_cached,
    load_full_decks,
    compute_average_copies_main,
    analyze_archetype_distribution,
    apply_flexible_distribution,
    build_extra_deck_optimized,
    build_side_deck_optimized,
    is_valid_extra_deck_card,
    is_extra_deck_card,
    clear_cache,
    detect_archetypes,
    score_card_synergy_enhanced,
    build_card_name_lookup,
    extract_referenced_card_ids_improved_cached,
    extract_referenced_card_ids_improved,
    ARCHETYPES_FILE,
    YDK_FOLDER,
    TARGET_MIN_MAIN,
    TARGET_MAX_MAIN,
    TARGET_EXTRA_SIZE,
    TARGET_SIDE_SIZE,
    SYNERGY_BOOST_FACTOR,
    ENHANCED_SYNERGY_BOOST,
    CARDINFO_FILE
)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Constants for relationship scoring
REQUIRED_RELATIONSHIP_SCORE = 5.0  # High score for required cards
SYNERGY_RELATIONSHIP_SCORE = 2.5  # Medium score for synergy cards
ARCHETYPE_RELATIONSHIP_SCORE = 3.0  # Score for archetype relationships


# Define request and response models
class Card(BaseModel):
    id: int
    name: Optional[str] = None
    type: Optional[str] = None
    desc: Optional[str] = None
    atk: Optional[int] = None
    def_: Optional[int] = None
    level: Optional[int] = None
    race: Optional[str] = None
    attribute: Optional[str] = None


class DeckEntry(BaseModel):
    deckEntryId: str
    card: Card


class DeckRequest(BaseModel):
    mainCards: List[Dict[str, Any]]
    extraCards: List[Dict[str, Any]]
    sideCards: List[Dict[str, Any]]


class DeckResponse(BaseModel):
    mainCards: List[Dict[str, Any]]
    extraCards: List[Dict[str, Any]]
    sideCards: List[Dict[str, Any]]


# Modified card info loader to handle duplicate IDs
def load_card_info(filename=CARDINFO_FILE):
    """
    Enhanced version of load_card_info that handles duplicate card IDs properly.

    Returns a dictionary where keys are card IDs and values are card data.
    Also returns a mapping of duplicate IDs for special handling.
    """
    # Get original card info
    original_cards = original_load_card_info(filename)

    # Track duplicates
    card_names = {}
    duplicate_ids = set()
    id_variants = defaultdict(list)

    # First pass: identify duplicates
    for card_id, card_data in original_cards.items():
        card_name = card_data.get('name', '').lower()

        if card_name in card_names:
            # Found a duplicate name
            existing_id = card_names[card_name]
            duplicate_ids.add(card_id)
            duplicate_ids.add(existing_id)

            # Record both IDs as variants of each other
            id_variants[card_id].append(existing_id)
            id_variants[existing_id].append(card_id)
        else:
            card_names[card_name] = card_id

    # Log duplicate information
    if duplicate_ids:
        print(f"Found {len(duplicate_ids)} card IDs with duplicates")
        for dup_id in duplicate_ids:
            if dup_id in original_cards:
                print(f"Duplicate ID {dup_id}: {original_cards[dup_id].get('name', 'Unknown')}")
                print(f"  Variants: {id_variants[dup_id]}")

    # Create a new dictionary with unique IDs
    unique_cards = {}
    for card_id, card_data in original_cards.items():
        # Store original ID in the card data for reference
        if card_id in duplicate_ids:
            # For duplicates, store the variant IDs
            card_data['duplicate_variants'] = id_variants[card_id]

        # Add to the unique cards dictionary
        unique_cards[card_id] = card_data

    return unique_cards


def handle_duplicate_ids(card_ids, card_info):
    """
    Handle duplicate card IDs by ensuring each required card is included
    even if it has duplicate variants.

    Args:
        card_ids: List of card IDs
        card_info: Card information dictionary

    Returns:
        Modified list of card IDs with duplicates properly handled
    """
    # Track which cards we've already included
    included_names = set()
    result_ids = []

    for card_id in card_ids:
        if card_id not in card_info:
            result_ids.append(card_id)
            continue

        card_name = card_info[card_id].get('name', '').lower()

        # Check if this is a duplicate
        if 'duplicate_variants' in card_info[card_id]:
            # If we haven't included this card name yet, include it
            if card_name not in included_names:
                included_names.add(card_name)
                result_ids.append(card_id)
                # Also check for any variant IDs that might be better (for multi-art cards)
                for variant_id in card_info[card_id]['duplicate_variants']:
                    if variant_id in card_info and 'card_images' in card_info[variant_id]:
                        # If variant has images and current doesn't, prefer the variant
                        if (not card_info[card_id].get('card_images') or
                                len(card_info[variant_id]['card_images']) > len(
                                    card_info[card_id].get('card_images', []))):
                            # Replace with better variant
                            result_ids[-1] = variant_id
        else:
            # Regular card, just add it
            result_ids.append(card_id)
            included_names.add(card_name)

    return result_ids


def extract_archetype_mentions(card_desc, archetypes):
    """
    Extract archetypes mentioned in a card description.

    Args:
        card_desc (str): The card description text
        archetypes (list): List of known archetypes

    Returns:
        list: List of mentioned archetypes
    """
    mentioned_archetypes = []

    # Lowercase the description for case-insensitive matching
    desc = card_desc.lower()

    # Look for archetype patterns in quotes or apostrophes
    quote_pattern = re.compile(r'["\']([^"\']+)["\']')
    quoted_terms = quote_pattern.findall(desc)

    # Check each quoted term against archetypes
    for term in quoted_terms:
        term_lower = term.lower()
        # Check if the term is an archetype or part of an archetype
        for archetype in archetypes:
            archetype_lower = archetype.lower().strip('"')
            if archetype_lower == term_lower or f"{archetype_lower}-" in term_lower:
                mentioned_archetypes.append(archetype_lower)

    # Also look for common patterns that indicate archetype mentions
    archetype_patterns = [
        r'(["\'][^"\']+["\'])\s+monster',
        r'(["\'][^"\']+["\'])\s+card',
        r'(["\'][^"\']+["\'])\s+spell',
        r'(["\'][^"\']+["\'])\s+trap'
    ]

    for pattern in archetype_patterns:
        matches = re.findall(pattern, desc)
        for match in matches:
            # Clean up the match (remove quotes)
            clean_match = match.strip('"\'')
            # Check if it's an archetype
            for archetype in archetypes:
                archetype_lower = archetype.lower().strip('"')
                if archetype_lower == clean_match.lower() or archetype_lower in clean_match.lower():
                    mentioned_archetypes.append(archetype_lower)

    return list(set(mentioned_archetypes))  # Remove duplicates


def get_archetype_cards(archetype, card_info):
    """
    Get cards belonging to a specific archetype.

    Args:
        archetype (str): The archetype to find cards for
        card_info (dict): Card information database

    Returns:
        list: List of card IDs belonging to the archetype
    """
    archetype_cards = []

    # Clean up archetype name
    clean_archetype = archetype.lower().strip('"')

    for card_id, card_data in card_info.items():
        # Check explicit archetype field
        if card_data.get('archetype', '').lower() == clean_archetype:
            archetype_cards.append(card_id)
            continue

        # Check card name for archetype
        card_name = card_data.get('name', '').lower()
        if clean_archetype in card_name:
            archetype_cards.append(card_id)
            continue

    return archetype_cards


def extract_card_relationships(card_id, card_info, archetypes):
    """
    Analyze card text to extract relationships between cards.

    Returns a dictionary mapping:
    - 'required': List of card IDs that are required for this card
    - 'synergy': List of card IDs that have synergy with this card
    - 'archetype': List of card IDs that belong to archetypes mentioned in this card
    """
    if card_id not in card_info:
        return {'required': [], 'synergy': [], 'archetype': []}

    card_data = card_info[card_id]
    card_desc = card_data.get('desc', '').lower()
    card_name = card_data.get('name', '').lower()
    card_type = card_data.get('type', '').lower()

    # Build name lookup dictionary for matching
    name_to_id = {}
    for cid, cdata in card_info.items():
        name = cdata.get('name', '').lower()
        if name:
            name_to_id[name] = cid

    # Initialize relationship dictionaries
    required_cards = []
    synergy_cards = []
    archetype_cards = []

    # Check for card requirements based on card type
    if 'fusion' in card_type:
        # Extract fusion materials from description
        required_cards.extend(extract_fusion_requirements(card_desc, name_to_id))

    elif 'synchro' in card_type:
        # Extract synchro materials (tuners and non-tuners)
        required_cards.extend(extract_synchro_requirements(card_desc, name_to_id))

    elif 'xyz' in card_type:
        # Extract xyz materials (level requirements)
        required_cards.extend(extract_xyz_requirements(card_desc, name_to_id, card_info))

    elif 'ritual' in card_type:
        # Extract ritual requirements (ritual spell and tributes)
        required_cards.extend(extract_ritual_requirements(card_desc, name_to_id, card_info))

    elif 'link' in card_type:
        # Extract link materials
        required_cards.extend(extract_link_requirements(card_desc, name_to_id))

    # Find other related cards (synergy)
    # This will look for cards mentioned in the description that aren't already required
    referenced_ids = extract_referenced_card_ids_improved(card_id, card_info, name_to_id)
    for ref_id in referenced_ids:
        if ref_id not in required_cards:
            synergy_cards.append(ref_id)

    # Extract archetype relationships from card description
    mentioned_archetypes = extract_archetype_mentions(card_desc, archetypes)
    for arch in mentioned_archetypes:
        arch_cards = get_archetype_cards(arch, card_info)

        # Skip extra deck cards if this is a main deck card
        if not is_extra_deck_card(card_id, card_info):
            arch_cards = [c for c in arch_cards if not is_extra_deck_card(c, card_info)]

        # Skip cards already in required or synergy lists
        for arch_card in arch_cards:
            if arch_card not in required_cards and arch_card not in synergy_cards:
                archetype_cards.append(arch_card)

    # Remove duplicates
    required_cards = list(set(required_cards))
    synergy_cards = list(set(synergy_cards))
    archetype_cards = list(set(archetype_cards))

    return {
        'required': required_cards,
        'synergy': synergy_cards,
        'archetype': archetype_cards
    }


def extract_fusion_requirements(card_desc, name_to_id):
    """Extract fusion material requirements from card description"""
    required_cards = []

    # Convert description to lowercase for easier matching
    desc = card_desc.lower()

    # Check for summon requirements section
    for section_marker in ['materials:', 'material:', 'fusion material']:
        if section_marker in desc:
            sections = desc.split(section_marker)
            if len(sections) > 1:
                material_section = sections[1].split('.')[0]  # Get text until next period
                break
        else:
            material_section = desc

    # Look for card names in quotes
    quoted_names = re.findall(r'"([^"]+)"', desc)
    quoted_names.extend(re.findall(r"'([^']+)'", desc))

    for name in quoted_names:
        name_lower = name.lower()
        if name_lower in name_to_id:
            required_cards.append(name_to_id[name_lower])

    # If no materials found using quotes, look for "+" pattern which is common in fusion requirements
    if not required_cards and '+' in desc:
        plus_sections = desc.split('+')
        for section in plus_sections:
            # Clean up the section
            cleaned = section.strip()
            # Look for full matches in name_to_id
            for name, cid in name_to_id.items():
                if name in cleaned:
                    required_cards.append(cid)

    # Check for specific patterns ("3 Blue-Eyes White Dragon")
    quantity_pattern = re.findall(r'(\d+)\s+([^,.]+)', desc)
    for quantity, item in quantity_pattern:
        try:
            quantity = int(quantity)
            # Check if item matches a card name
            for name, cid in name_to_id.items():
                if name in item.lower():
                    for _ in range(quantity):
                        required_cards.append(cid)
        except ValueError:
            pass

    return required_cards


def extract_synchro_requirements(card_desc, name_to_id):
    """Extract synchro material requirements from card description"""
    required_cards = []

    # Check for tuner requirements
    tuner_pattern = re.search(r'(\d+)\s+tuners?', card_desc.lower())
    non_tuner_pattern = re.search(r'(\d+)\s+non-tuners?', card_desc.lower())

    # Look for specifically named cards
    quoted_names = re.findall(r'"([^"]+)"', card_desc)
    quoted_names.extend(re.findall(r"'([^']+)'", card_desc))

    for name in quoted_names:
        name_lower = name.lower()
        if name_lower in name_to_id:
            required_cards.append(name_to_id[name_lower])

    return required_cards


def extract_xyz_requirements(card_desc, name_to_id, card_info):
    """Extract xyz material requirements from card description"""
    required_cards = []

    # Check for level requirements
    level_pattern = re.search(r'(\d+)\s+level\s+(\d+)', card_desc.lower())

    # Look for specifically named cards
    quoted_names = re.findall(r'"([^"]+)"', card_desc)
    quoted_names.extend(re.findall(r"'([^']+)'", card_desc))

    for name in quoted_names:
        name_lower = name.lower()
        if name_lower in name_to_id:
            required_cards.append(name_to_id[name_lower])

    return required_cards


def extract_ritual_requirements(card_desc, name_to_id, card_info):
    """Extract ritual requirements from card description"""
    required_cards = []

    # Look for specifically named cards
    quoted_names = re.findall(r'"([^"]+)"', card_desc)
    quoted_names.extend(re.findall(r"'([^']+)'", card_desc))

    for name in quoted_names:
        name_lower = name.lower()
        if name_lower in name_to_id:
            required_cards.append(name_to_id[name_lower])

    # Look for ritual spell cards
    for card_id, data in card_info.items():
        card_type = data.get('type', '').lower()
        if 'ritual' in card_type and 'spell' in card_type:
            card_desc = data.get('desc', '').lower()

            # If this ritual spell mentions our ritual monster, it's likely the matching spell
            for name in quoted_names:
                if name.lower() in card_desc:
                    required_cards.append(card_id)
                    break

    return required_cards


def extract_link_requirements(card_desc, name_to_id):
    """Extract link material requirements from card description"""
    required_cards = []

    # Look for specifically named cards
    quoted_names = re.findall(r'"([^"]+)"', card_desc)
    quoted_names.extend(re.findall(r"'([^']+)'", card_desc))

    for name in quoted_names:
        name_lower = name.lower()
        if name_lower in name_to_id:
            required_cards.append(name_to_id[name_lower])

    return required_cards


def analyze_card_relationships(input_cards, card_info, archetypes):
    """
    Analyze relationships between input cards and all cards in the database.
    Returns scoring modifiers for all relevant cards.
    """
    # Initialize scoring dictionary with default score of 1.0 (no modification)
    card_scores = defaultdict(lambda: 1.0)

    # Process each input card
    for card_id in input_cards:
        if card_id not in card_info:
            continue

        # Get relationships for this card
        relationships = extract_card_relationships(card_id, card_info, archetypes)

        # Apply scores for required cards
        for req_id in relationships['required']:
            card_scores[req_id] *= REQUIRED_RELATIONSHIP_SCORE

        # Apply scores for synergy cards
        for syn_id in relationships['synergy']:
            card_scores[syn_id] *= SYNERGY_RELATIONSHIP_SCORE

        # Apply scores for archetype cards
        for arch_id in relationships['archetype']:
            card_scores[arch_id] *= ARCHETYPE_RELATIONSHIP_SCORE

    return card_scores


def analyze_card_relationships_balanced(input_cards, card_info, archetypes):
    """
    Analyze relationships between input cards and all cards in the database,
    ensuring a balance between potentially competing strategies.

    Returns a dictionary with:
    - 'scores': Dictionary mapping card_id to relationship score
    - 'card_groups': Dictionary mapping each input card to its support cards
    """
    # Track overall scores and per-input-card scores
    overall_scores = defaultdict(lambda: 1.0)
    card_groups = {}

    # Process each input card individually to identify its support group
    for input_card in input_cards:
        if input_card not in card_info:
            continue

        # Create a separate score dictionary for this input card
        card_scores = defaultdict(lambda: 1.0)

        # Get relationships for this card
        relationships = extract_card_relationships(input_card, card_info, archetypes)

        # Apply scores for required cards
        for req_id in relationships['required']:
            card_scores[req_id] *= REQUIRED_RELATIONSHIP_SCORE

        # Apply scores for synergy cards
        for syn_id in relationships['synergy']:
            card_scores[syn_id] *= SYNERGY_RELATIONSHIP_SCORE

        # Apply scores for archetype cards
        for arch_id in relationships['archetype']:
            card_scores[arch_id] *= ARCHETYPE_RELATIONSHIP_SCORE

        # Get archetype for this input card
        card_archetypes = set()
        arch = card_info[input_card].get("archetype")
        if arch:
            card_archetypes.add(arch.lower())
        detected = detect_archetypes_cached(input_card, card_info, archetypes)
        card_archetypes.update(detected)

        # Find other cards of the same archetype
        for card_id, card_data in card_info.items():
            if card_id == input_card:
                continue

            # Check card's archetypes
            arch_matches = False
            arch = card_data.get("archetype")
            if arch and arch.lower() in card_archetypes:
                arch_matches = True
            else:
                card_archs = detect_archetypes_cached(card_id, card_info, archetypes)
                if card_archs & card_archetypes:  # If there's an intersection
                    arch_matches = True

            if arch_matches:
                card_scores[card_id] *= 1.5  # Boost for same archetype

        # Store the top support cards for this input card
        # Sort by score and select top cards
        top_support = sorted(
            [(card_id, score) for card_id, score in card_scores.items() if score > 1.0],
            key=lambda x: x[1],
            reverse=True
        )[:20]  # Store top 20 support cards

        card_groups[input_card] = top_support

        # Add these scores to the overall score map
        for card_id, score in card_scores.items():
            overall_scores[card_id] = max(overall_scores[card_id], score)

    return {
        'scores': overall_scores,
        'card_groups': card_groups
    }


def build_balanced_main_deck(input_cards, relationship_data, card_info, archetypes,
                             main_decks, desired_distribution, avg_main, target_size=40):
    """
    Build a main deck that balances support for multiple input strategies.

    This ensures that each input card's strategy is properly represented.
    """
    # Start with an empty deck
    new_main_deck = []

    # First add all input cards that belong in the main deck
    for card in input_cards:
        if card in card_info and not is_extra_deck_card(card, card_info) and card not in new_main_deck:
            new_main_deck.append(card)

    # Get required supporting cards for each input card
    card_groups = relationship_data['card_groups']
    remaining_slots = target_size - len(new_main_deck)

    # Calculate how many slots to allocate per input card's strategy
    if not input_cards:
        return []

    slots_per_strategy = max(3, remaining_slots // (len(input_cards) * 2))  # Ensure at least 3 slots per strategy

    # For each input card, add its most important support cards
    for input_card, support_cards in card_groups.items():
        if len(new_main_deck) >= target_size:
            break

        # Skip extra deck cards when building main deck
        if is_extra_deck_card(input_card, card_info):
            continue

        # Add top support cards for this input card
        cards_added = 0
        for card_id, score in support_cards:
            # Skip if already in deck
            if card_id in new_main_deck:
                continue

            # Skip extra deck cards
            if is_extra_deck_card(card_id, card_info):
                continue

            # Add the card
            new_main_deck.append(card_id)
            cards_added += 1

            # Check if we've added enough for this strategy
            if cards_added >= slots_per_strategy or len(new_main_deck) >= target_size:
                break

    # If we still need more cards, add high-scoring cards regardless of strategy
    if len(new_main_deck) < target_size:
        # Sort all cards by score
        overall_scores = relationship_data['scores']
        all_cards_scored = sorted(
            [(card_id, score) for card_id, score in overall_scores.items() if score > 1.0],
            key=lambda x: x[1],
            reverse=True
        )

        # Add remaining high-score cards
        for card_id, score in all_cards_scored:
            if card_id in new_main_deck:
                continue

            if not is_extra_deck_card(card_id, card_info):
                # For high-scoring cards, consider adding multiple copies
                copies = min(3, int(np.ceil(score / 2.0)))
                for _ in range(copies):
                    if len(new_main_deck) < target_size:
                        new_main_deck.append(card_id)

    # If we still don't have enough cards, use the original flexible distribution
    if len(new_main_deck) < TARGET_MIN_MAIN:
        completed_deck = apply_flexible_distribution(
            new_main_deck, main_decks, card_info, desired_distribution,
            avg_main, input_cards, archetypes
        )
        new_main_deck = completed_deck

    return new_main_deck


def enhanced_score_card_synergy(candidate_id, input_cards, card_info, archetypes, relationship_scores):
    """
    Enhanced version of score_card_synergy that takes into account relationship scores.
    """
    # Get base score from the original function
    base_score = score_card_synergy_enhanced(candidate_id, input_cards, card_info, archetypes)

    # Apply relationship score modifier
    relationship_modifier = relationship_scores.get(candidate_id, 1.0)

    # Combine scores
    final_score = base_score * relationship_modifier

    return final_score


@app.post("/api/generate-deck", response_model=DeckResponse)
async def generate_deck(request: DeckRequest):
    try:
        print("Received deck generation request")

        # Extract card IDs from the request
        main_card_ids = [str(entry.get("card", {}).get("id")) for entry in request.mainCards]
        extra_card_ids = [str(entry.get("card", {}).get("id")) for entry in request.extraCards]
        side_card_ids = [str(entry.get("card", {}).get("id")) for entry in request.sideCards]

        # Filter out None or empty values
        main_card_ids = [card_id for card_id in main_card_ids if card_id]
        extra_card_ids = [card_id for card_id in extra_card_ids if card_id]
        side_card_ids = [card_id for card_id in side_card_ids if card_id]

        print(
            f"Processing deck with {len(main_card_ids)} main, {len(extra_card_ids)} extra, {len(side_card_ids)} side cards")

        # Load necessary data with enhanced card loading to handle duplicates
        card_info = load_card_info()
        with open(ARCHETYPES_FILE, "r", encoding="utf-8") as f:
            archetypes = json.load(f)

        # Handle duplicate IDs in input cards
        main_card_ids = handle_duplicate_ids(main_card_ids, card_info)
        extra_card_ids = handle_duplicate_ids(extra_card_ids, card_info)
        side_card_ids = handle_duplicate_ids(side_card_ids, card_info)

        # Clear cache and load decks
        clear_cache()
        full_decks = load_full_decks(YDK_FOLDER)
        main_decks = [deck["main"] for deck in full_decks]

        # Process valid input cards
        valid_input_extra = [card for card in extra_card_ids if is_valid_extra_deck_card(card, card_info)]
        valid_input_side = [card for card in side_card_ids
                            if not is_extra_deck_card(card, card_info) or is_valid_extra_deck_card(card, card_info)]

        # Combine all input cards
        all_input_cards = main_card_ids + valid_input_extra + valid_input_side

        # Track distinct archetypes in input
        distinct_archetypes = set()
        for card in all_input_cards:
            if card in card_info:
                arch = card_info[card].get("archetype")
                if arch:
                    distinct_archetypes.add(arch.lower())

                detected = detect_archetypes_cached(card, card_info, archetypes)
                distinct_archetypes.update(detected)

        if len(distinct_archetypes) > 1:
            print(f"Multiple archetypes detected in input: {distinct_archetypes}")
            print("Using balanced strategy algorithm for deck generation")

        # Analyze card relationships with balanced approach
        print("Analyzing card relationships...")
        relationship_data = analyze_card_relationships_balanced(all_input_cards, card_info, archetypes)
        relationship_scores = relationship_data['scores']

        # Print top relationship scores for debugging
        print("Top relationship scores:")
        top_relationships = sorted(relationship_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        for card_id, score in top_relationships:
            card_name = card_info[card_id]["name"] if card_id in card_info else f"Unknown ({card_id})"
            print(f"  {card_name}: {score}")

        # Find required cards for the main deck
        required_main_cards = []

        # If extra deck cards but no main deck cards, check what's needed
        if valid_input_extra and not main_card_ids:
            print("Finding required main deck cards for extra deck summons...")

            for card_id in valid_input_extra:
                relationships = extract_card_relationships(card_id, card_info, archetypes)

                # Add required cards
                for req_id in relationships['required']:
                    if req_id not in required_main_cards and not is_extra_deck_card(req_id, card_info):
                        required_main_cards.append(req_id)
                        card_name = card_info[req_id]["name"] if req_id in card_info else f"Unknown ({req_id})"
                        print(f"  Required: {card_name}")

                # Add archetype cards with high scores
                for arch_id in relationships['archetype']:
                    if arch_id not in required_main_cards and not is_extra_deck_card(arch_id, card_info):
                        if relationship_scores[arch_id] >= ARCHETYPE_RELATIONSHIP_SCORE:
                            required_main_cards.append(arch_id)
                            card_name = card_info[arch_id]["name"] if arch_id in card_info else f"Unknown ({arch_id})"
                            print(f"  Archetype required: {card_name}")

        # CRITICAL FIX: Create synthetic main cards when there are no main cards
        # This works around limitations in the underlying code that only filters based on main deck
        synthetic_main_cards = []

        # Add required cards to synthetic main
        if required_main_cards:
            print(f"Adding {len(required_main_cards)} required cards to synthetic main deck")
            synthetic_main_cards.extend(required_main_cards)

        if not main_card_ids and (valid_input_extra or valid_input_side):
            print("No main deck cards provided. Creating synthetic main cards to guide generation.")

            # First use required cards from relationships, then add more based on archetypes

            # Get archetypes from extra and side cards
            input_archetypes = set()
            for card in valid_input_extra + valid_input_side:
                if card in card_info:
                    # Try to get from explicit archetype field
                    arch = card_info[card].get("archetype")
                    if arch:
                        input_archetypes.add(arch.lower())

                    # Also get from name/desc detection
                    detected_archetypes = detect_archetypes_cached(card, card_info, archetypes)
                    input_archetypes.update(detected_archetypes)

                    # Add archetypes mentioned in card description
                    mentioned_archetypes = extract_archetype_mentions(card_info[card].get('desc', ''), archetypes)
                    input_archetypes.update(mentioned_archetypes)

            print(f"Detected archetypes from input cards: {input_archetypes}")

            # Find popular main deck cards that match these archetypes
            if input_archetypes:
                archetype_cards = []
                for deck in main_decks:
                    for card in deck:
                        if card in card_info:
                            # Check card's archetypes
                            arch = card_info[card].get("archetype")
                            arch_matches = False

                            if arch and arch.lower() in input_archetypes:
                                arch_matches = True
                            else:
                                card_archetypes = detect_archetypes_cached(card, card_info, archetypes)
                                if card_archetypes & input_archetypes:  # If there's an intersection
                                    arch_matches = True

                            # Only use non-extra deck cards for the synthetic main
                            if arch_matches and not is_extra_deck_card(card, card_info):
                                archetype_cards.append(card)

                # Count frequencies and use top cards as synthetic main
                card_counter = Counter(archetype_cards)
                # Apply relationship scores to the counter values
                for card_id, count in card_counter.items():
                    card_counter[card_id] = count * relationship_scores.get(card_id, 1.0)

                synthetic_archetype_cards = [card for card, _ in card_counter.most_common(10)]
                synthetic_main_cards.extend(
                    [card for card in synthetic_archetype_cards if card not in synthetic_main_cards])
                print(f"Created {len(synthetic_archetype_cards)} synthetic main cards from archetypes")

        # Combine all input cards - now including synthetic main cards
        all_input_cards = main_card_ids + synthetic_main_cards + valid_input_extra + valid_input_side

        # If we still have no input cards at all, we'll have to start from scratch
        if not all_input_cards:
            print("No input cards at all, generating a generic deck")
            # Just pick some generic popular cards to start with
            card_counter = Counter(card for deck in main_decks for card in deck)
            synthetic_main_cards = [card for card, _ in card_counter.most_common(5)]
            all_input_cards = synthetic_main_cards
            print(f"Created {len(synthetic_main_cards)} generic starter cards")

        # Get average copies for main deck
        avg_main = compute_average_copies_main(main_decks)

        # For main deck building, use both real main cards and synthetic ones
        starter_main_deck = main_card_ids + synthetic_main_cards

        # Always use all input cards to determine archetype distribution
        desired_distribution = analyze_archetype_distribution(
            all_input_cards, card_info, archetypes, main_decks, target_size=TARGET_MAX_MAIN
        )

        print(f"Determined distribution: {desired_distribution}")

        # Use balanced deck building approach when we have multiple input cards or archetypes
        if len(all_input_cards) > 1 and len(distinct_archetypes) > 1:
            new_main_deck = build_balanced_main_deck(
                all_input_cards,
                relationship_data,
                card_info,
                archetypes,
                main_decks,
                desired_distribution,
                avg_main
            )
        else:
            # Custom deck building with relationship scores (single strategy)
            new_main_deck = []

            # First, add all cards with high relationship scores
            for card_id, score in sorted(relationship_scores.items(), key=lambda x: x[1], reverse=True):
                if score > 2.0 and not is_extra_deck_card(card_id, card_info) and card_id not in new_main_deck:
                    # More copies for higher scores
                    copies = min(3, int(np.ceil(score / 2.0)))
                    for _ in range(copies):
                        if len(new_main_deck) < TARGET_MAX_MAIN:
                            new_main_deck.append(card_id)
                            if score > 3.0:  # For very high scores, log these additions
                                card_name = card_info[card_id][
                                    "name"] if card_id in card_info else f"Unknown ({card_id})"
                                print(f"Added high-scoring card: {card_name} (score: {score:.1f})")

            # If we already have starter cards, include them
            for card in starter_main_deck:
                if card not in new_main_deck and len(new_main_deck) < TARGET_MAX_MAIN:
                    new_main_deck.append(card)

            # Now use the original function to complete the deck
            if len(new_main_deck) < TARGET_MIN_MAIN:
                completed_deck = apply_flexible_distribution(
                    new_main_deck, main_decks, card_info, desired_distribution,
                    avg_main, all_input_cards, archetypes
                )
                new_main_deck = completed_deck

        # Ensure all required main cards are included
        for card_id in required_main_cards:
            if card_id not in new_main_deck and len(new_main_deck) < TARGET_MAX_MAIN:
                new_main_deck.append(card_id)
                card_name = card_info[card_id]["name"] if card_id in card_info else f"Unknown ({card_id})"
                print(f"Added required card: {card_name}")

        # Remove synthetic cards if they made it through and they're not required cards
        if len(new_main_deck) > TARGET_MIN_MAIN:
            # Keep required cards and high-scoring cards
            new_main_deck = [card for card in new_main_deck if
                             card in main_card_ids or
                             card in required_main_cards or
                             relationship_scores.get(card, 1.0) > 2.0 or
                             card not in synthetic_main_cards]

            # Ensure we still have enough cards
            if len(new_main_deck) < TARGET_MIN_MAIN:
                print(f"After removing synthetic cards, need {TARGET_MIN_MAIN - len(new_main_deck)} more cards")

                # Add more cards to reach minimum, incorporating relationship scores
                candidate_scores = []
                for deck in main_decks:
                    for card in deck:
                        if card not in new_main_deck and not is_extra_deck_card(card, card_info):
                            base_score = score_card_synergy_enhanced(card, all_input_cards, card_info, archetypes)
                            final_score = base_score * relationship_scores.get(card, 1.0)
                            candidate_scores.append((card, final_score))

                # Sort by score and add top cards
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                for card, _ in candidate_scores:
                    if len(new_main_deck) >= TARGET_MIN_MAIN:
                        break
                    if card not in new_main_deck:
                        new_main_deck.append(card)

        # IMPROVED EXTRA DECK HANDLING
        print(f"Building extra deck with {len(valid_input_extra)} provided cards")
        if valid_input_extra:
            # Only get recommendations for the remaining slots
            remaining_extra_slots = TARGET_EXTRA_SIZE - len(valid_input_extra)

            if remaining_extra_slots > 0:
                # For extra deck synergy, now use ALL cards (main + extra + side)
                all_synergy_cards = new_main_deck + valid_input_extra + valid_input_side

                # We need to use the existing function, but make sure it has card IDs to work with
                recommended_extra = build_extra_deck_optimized(
                    all_synergy_cards, full_decks, TARGET_EXTRA_SIZE, card_info, archetypes
                )
                # Filter out duplicates of input cards
                recommended_extra = [card for card in recommended_extra if card not in valid_input_extra]
                # Combine, prioritizing input cards
                new_extra_deck = valid_input_extra + recommended_extra[:remaining_extra_slots]
            else:
                # If already at or exceeding TARGET_EXTRA_SIZE, just use input cards
                new_extra_deck = valid_input_extra[:TARGET_EXTRA_SIZE]
        else:
            # No input extra cards, generate based on all provided cards and the new main deck
            all_synergy_cards = new_main_deck + valid_input_side
            new_extra_deck = build_extra_deck_optimized(
                all_synergy_cards, full_decks, TARGET_EXTRA_SIZE, card_info, archetypes
            )

        # IMPROVED SIDE DECK HANDLING
        print(f"Building side deck with {len(valid_input_side)} provided cards")
        if valid_input_side:
            remaining_side_slots = TARGET_SIDE_SIZE - len(valid_input_side)

            if remaining_side_slots > 0:
                # Use all cards for better synergy
                all_synergy_cards = new_main_deck + new_extra_deck + valid_input_side

                recommended_side = build_side_deck_optimized(
                    all_synergy_cards, full_decks, TARGET_SIDE_SIZE, card_info,
                    archetypes, new_main_deck, new_extra_deck
                )
                # Filter out duplicates of input cards
                recommended_side = [card for card in recommended_side if card not in valid_input_side]
                # Combine, prioritizing input cards
                new_side_deck = valid_input_side + recommended_side[:remaining_side_slots]
            else:
                # If already at or exceeding TARGET_SIDE_SIZE, just use input cards
                new_side_deck = valid_input_side[:TARGET_SIDE_SIZE]
        else:
            # No input side cards, generate based on all provided cards
            all_synergy_cards = new_main_deck + new_extra_deck
            new_side_deck = build_side_deck_optimized(
                all_synergy_cards, full_decks, TARGET_SIDE_SIZE, card_info,
                archetypes, new_main_deck, new_extra_deck
            )

        # Handle duplicate IDs in final lists to ensure variety
        new_main_deck = handle_duplicate_ids(new_main_deck, card_info)
        new_extra_deck = handle_duplicate_ids(new_extra_deck, card_info)
        new_side_deck = handle_duplicate_ids(new_side_deck, card_info)

        # Format the response to match the expected React structure
        formatted_main = [
            {"deckEntryId": str(uuid.uuid4()), "card": {"id": int(card_id)}}
            for card_id in new_main_deck
        ]

        formatted_extra = [
            {"deckEntryId": str(uuid.uuid4()), "card": {"id": int(card_id)}}
            for card_id in new_extra_deck
        ]

        formatted_side = [
            {"deckEntryId": str(uuid.uuid4()), "card": {"id": int(card_id)}}
            for card_id in new_side_deck
        ]

        print(
            f"Deck generation complete: {len(new_main_deck)} main, {len(new_extra_deck)} extra, {len(new_side_deck)} side")

        return {
            "mainCards": formatted_main,
            "extraCards": formatted_extra,
            "sideCards": formatted_side
        }

    except Exception as e:
        print(f"Error generating deck: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run("deck_ai_api:app", host="0.0.0.0", port=8000, reload=True)