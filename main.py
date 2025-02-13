import os
import json
import requests
from collections import Counter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ---------------------------
# Existing Deck Builder Code
# ---------------------------

# Global cache for card data
CARD_CACHE = {}

def load_card_cache(cache_filename="cards.json"):
    """
    Loads the entire card database from a local cache.
    If the cache file does not exist, download the data from the API and save it.
    Returns a dictionary mapping card IDs (as strings) to their card info.
    """
    global CARD_CACHE
    if os.path.exists(cache_filename):
        with open(cache_filename, "r") as f:
            data = json.load(f)
    else:
        print("Downloading card data from API...")
        url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
        response = requests.get(url)
        data = response.json()
        with open(cache_filename, "w") as f:
            json.dump(data, f)
    CARD_CACHE = {str(card["id"]): card for card in data["data"]}
    return CARD_CACHE

def get_card_info(card_id):
    """
    Retrieve card information from the local cache.
    """
    if not CARD_CACHE:
        load_card_cache()
    return CARD_CACHE.get(card_id)

# ---------------------------
# Parsing .ydk files
# ---------------------------
def parse_ydk_file(file_path):
    """
    Parse a .ydk deck file into main, extra, and side sections.
    """
    main_deck, extra_deck, side_deck = [], [], []
    current_section = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "#main":
                current_section = "main"
                continue
            elif line == "#extra":
                current_section = "extra"
                continue
            elif line == "!side":
                current_section = "side"
                continue
            if current_section == "main":
                main_deck.append(line)
            elif current_section == "extra":
                extra_deck.append(line)
            elif current_section == "side":
                side_deck.append(line)
    return main_deck, extra_deck, side_deck

# ---------------------------
# Historical Composition Analysis
# ---------------------------
def compute_average_deck_composition(deck_files):
    """
    Computes the global average counts of monsters, spells, and traps in the main decks.
    """
    total_monsters = total_spells = total_traps = 0
    deck_count = 0
    for deck_file in deck_files:
        main_deck, _, _ = parse_ydk_file(deck_file)
        for card_id in main_deck:
            info = get_card_info(card_id)
            if not info:
                continue
            ctype = info.get("type", "")
            if "Monster" in ctype:
                total_monsters += 1
            elif "Spell" in ctype:
                total_spells += 1
            elif "Trap" in ctype:
                total_traps += 1
        deck_count += 1
    if deck_count == 0:
        return 0, 0, 0
    return total_monsters / deck_count, total_spells / deck_count, total_traps / deck_count

def compute_card_deck_composition(card_id, deck_files):
    """
    For decks that include card_id in the main deck, compute the average composition.
    Returns a tuple: (avg_monsters, avg_spells, avg_traps)
    """
    total_monsters = total_spells = total_traps = 0
    count = 0
    for deck_file in deck_files:
        main_deck, _, _ = parse_ydk_file(deck_file)
        if card_id in main_deck:
            for cid in main_deck:
                info = get_card_info(cid)
                if not info:
                    continue
                ctype = info.get("type", "")
                if "Monster" in ctype:
                    total_monsters += 1
                elif "Spell" in ctype:
                    total_spells += 1
                elif "Trap" in ctype:
                    total_traps += 1
            count += 1
    if count == 0:
        return (0, 0, 0)
    return (total_monsters / count, total_spells / count, total_traps / count)

def compute_average_deck_for_input_cards(input_cards, deck_files):
    """
    Averages the deck composition for each input card over historical decks.
    """
    totals = [0, 0, 0]
    count = 0
    for card in input_cards:
        comp = compute_card_deck_composition(card, deck_files)
        if sum(comp) > 0:
            totals[0] += comp[0]
            totals[1] += comp[1]
            totals[2] += comp[2]
            count += 1
    if count == 0:
        return compute_average_deck_composition(deck_files)
    return (totals[0] / count, totals[1] / count, totals[2] / count)

# ---------------------------
# Recommendation functions
# ---------------------------
def get_card_neighbors(target_card, deck_files):
    """
    Returns a Counter of cards that appear in the same decks as target_card.
    """
    neighbor_counter = Counter()
    for deck_file in deck_files:
        main_deck, extra_deck, side_deck = parse_ydk_file(deck_file)
        all_cards = main_deck + extra_deck + side_deck
        if target_card in all_cards:
            for card in all_cards:
                if card != target_card:
                    neighbor_counter[card] += 1
    return neighbor_counter

def get_section_frequency(deck_files, section):
    """
    Returns a frequency counter for all cards appearing in the specified section ('main', 'extra', or 'side')
    across all historical decks.
    """
    freq = Counter()
    for deck_file in deck_files:
        main_deck, extra_deck, side_deck = parse_ydk_file(deck_file)
        if section == "main":
            freq.update(main_deck)
        elif section == "extra":
            freq.update(extra_deck)
        elif section == "side":
            freq.update(side_deck)
    return freq

# ---------------------------
# Dropoff check helper:
# ---------------------------
def fill_from_sorted_recommendations(sorted_recs, desired_size, dropoff_ratio=0.3):
    """
    Given sorted recommendations as a list of tuples (card_id, frequency),
    add cards until desired_size is reached.
    If the frequency of the next card is below dropoff_ratio * (max frequency), then stop.
    """
    deck_section = []
    if not sorted_recs:
        return deck_section
    max_freq = sorted_recs[0][1]
    for card, freq in sorted_recs:
        if len(deck_section) >= desired_size:
            break
        if freq < dropoff_ratio * max_freq:
            break
        if card not in deck_section:
            deck_section.append(card)
    return deck_section

# ---------------------------
# Copy Count Analysis
# ---------------------------
def compute_average_copy(card_id, deck_files):
    """
    Computes the average number of copies of a given card in historical main decks.
    Returns at least 1 (and rounds to nearest integer, capped at 3).
    """
    total_copies = 0
    deck_count = 0
    for deck_file in deck_files:
        main_deck, _, _ = parse_ydk_file(deck_file)
        if card_id in main_deck:
            total_copies += main_deck.count(card_id)
            deck_count += 1
    if deck_count == 0:
        return 1
    avg = total_copies / deck_count
    return max(1, min(3, round(avg)))

def adjust_main_deck_copy_counts(unique_deck, deck_files, desired_size):
    """
    Given a unique list of card IDs, replicate each card based on its average copy count.
    If adding the recommended copies would exceed desired_size, add as many as possible.
    Returns a new main deck list (which may have duplicates).
    """
    new_deck = []
    for card in unique_deck:
        rec_copies = compute_average_copy(card, deck_files)
        for _ in range(rec_copies):
            if len(new_deck) < desired_size:
                new_deck.append(card)
            else:
                break
        if len(new_deck) >= desired_size:
            break
    idx = 0
    while len(new_deck) < desired_size and unique_deck:
        card = unique_deck[idx % len(unique_deck)]
        new_deck.append(card)
        idx += 1
    return new_deck

# ---------------------------
# Build Main Deck
# ---------------------------
def build_main_deck(input_cards, deck_files, desired_main_deck_size=40, dropoff_ratio=0.3):
    """
    Build a main deck based on input cards and recommendations.
    Uses the average deck composition from decks that contain the input cards,
    falling back to the global historical average if needed.
    Then adjusts copy counts based on historical averages.
    """
    target_monsters, target_spells, target_traps = compute_average_deck_for_input_cards(input_cards, deck_files)
    print(f"\nTarget main deck composition based on input cards:")
    print(f"  Monsters: {target_monsters:.1f}, Spells: {target_spells:.1f}, Traps: {target_traps:.1f}\n")

    unique_main = list(input_cards)

    recommendations = Counter()
    for card in input_cards:
        recs = get_card_neighbors(card, deck_files)
        recommendations.update(recs)
    sorted_recs = recommendations.most_common()

    current_counts = {"Monster": 0, "Spell": 0, "Trap": 0}
    for card in unique_main:
        info = get_card_info(card)
        if info:
            ctype = info.get("type", "")
            if "Monster" in ctype:
                current_counts["Monster"] += 1
            elif "Spell" in ctype:
                current_counts["Spell"] += 1
            elif "Trap" in ctype:
                current_counts["Trap"] += 1

    for rec_card, freq in sorted_recs:
        if len(unique_main) >= desired_main_deck_size:
            break
        if freq < dropoff_ratio * sorted_recs[0][1]:
            break
        if rec_card in unique_main:
            continue
        info = get_card_info(rec_card)
        if not info:
            continue
        ctype = info.get("type", "")
        if "Monster" in ctype and current_counts["Monster"] < target_monsters:
            unique_main.append(rec_card)
            current_counts["Monster"] += 1
        elif "Spell" in ctype and current_counts["Spell"] < target_spells:
            unique_main.append(rec_card)
            current_counts["Spell"] += 1
        elif "Trap" in ctype and current_counts["Trap"] < target_traps:
            unique_main.append(rec_card)
            current_counts["Trap"] += 1

    for rec_card, _ in sorted_recs:
        if len(unique_main) >= desired_main_deck_size:
            break
        if rec_card not in unique_main:
            unique_main.append(rec_card)

    final_main = adjust_main_deck_copy_counts(unique_main, deck_files, desired_main_deck_size)
    return final_main

# ---------------------------
# Build Extra Deck
# ---------------------------
def build_extra_deck(deck_files, desired_extra_deck_size=15, dropoff_ratio=0.3):
    """
    Build an extra deck by analyzing historical extra deck cards.
    Only include cards appropriate for the extra deck (Fusion, Synchro, Xyz, or Link monsters).
    """
    freq_extra = get_section_frequency(deck_files, "extra")
    valid_extra = {}
    for card_id, freq in freq_extra.items():
        info = get_card_info(card_id)
        if not info:
            continue
        ctype = info.get("type", "").lower()
        if any(keyword in ctype for keyword in ["fusion", "synchro", "xyz", "link"]):
            valid_extra[card_id] = freq
    sorted_recs = sorted(valid_extra.items(), key=lambda x: x[1], reverse=True)
    extra_deck = fill_from_sorted_recommendations(sorted_recs, desired_extra_deck_size, dropoff_ratio)
    return extra_deck

# ---------------------------
# Build Side Deck
# ---------------------------
def build_side_deck(deck_files, desired_side_deck_size=15, dropoff_ratio=0.3):
    """
    Build a side deck by analyzing historical side deck cards.
    """
    freq_side = get_section_frequency(deck_files, "side")
    sorted_recs = sorted(freq_side.items(), key=lambda x: x[1], reverse=True)
    side_deck = fill_from_sorted_recommendations(sorted_recs, desired_side_deck_size, dropoff_ratio)
    return side_deck

# ---------------------------
# Banlist & Deck Validation (Ruleset)
# ---------------------------
def fetch_banlist():
    """
    Fetches the official banlist from YGOProDeck.
    """
    url = "https://db.ygoprodeck.com/api/v7/cardinfo.php?banlist=tcg"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to retrieve banlist.")
        return None

def validate_deck(main_deck, side_deck, extra_deck):
    """
    Validates the complete deck against official rules:
      - Main Deck must have 40 to 60 cards.
      - Extra Deck must have no more than 15 cards and only include Fusion, Synchro, Xyz, or Link monsters.
      - Side Deck must have no more than 15 cards.
      - A card’s total copies (across all sections) must not exceed 3 (unless limited).
      - Forbidden cards are not allowed.
    """
    errors = []

    # Rule: Main Deck size 40-60
    if not (40 <= len(main_deck) <= 60):
        errors.append(f"Main Deck must have 40 to 60 cards (currently {len(main_deck)}).")

    # Rule: Extra Deck size <=15
    if len(extra_deck) > 15:
        errors.append(f"Extra Deck must have 15 or fewer cards (currently {len(extra_deck)}).")

    # Rule: Side Deck size <=15
    if len(side_deck) > 15:
        errors.append(f"Side Deck must have 15 or fewer cards (currently {len(side_deck)}).")

    banlist_data = fetch_banlist()
    forbidden_cards = []
    limited_cards = []
    semi_limited_cards = []
    if banlist_data:
        for card in banlist_data['data']:
            ban_status = card.get('banlist_info', {})
            if ban_status.get('ban_tcg') == "Forbidden":
                forbidden_cards.append(card['name'])
            elif ban_status.get('ban_tcg') == "Limited":
                limited_cards.append(card['name'])
            elif ban_status.get('ban_tcg') == "Semi-Limited":
                semi_limited_cards.append(card['name'])

    all_cards = main_deck + extra_deck + side_deck
    card_names = [get_card_info(card)["name"] for card in all_cards if get_card_info(card)]
    counts = Counter(card_names)

    # Rule: Forbidden cards
    for name in card_names:
        if name in forbidden_cards:
            errors.append(f"Forbidden card '{name}' is in the deck.")

    # Rule: Copy limits (3 copies normally; 1 for Limited; 2 for Semi-Limited)
    for name, count in counts.items():
        if name in limited_cards and count > 1:
            errors.append(f"Limited card '{name}' appears {count} times (maximum 1 allowed).")
        elif name in semi_limited_cards and count > 2:
            errors.append(f"Semi-Limited card '{name}' appears {count} times (maximum 2 allowed).")
        elif count > 3:
            errors.append(f"Card '{name}' appears {count} times (maximum 3 allowed).")

    # Rule: Extra Deck must only include valid Extra Deck monsters.
    for card in extra_deck:
        info = get_card_info(card)
        if info:
            ctype = info.get("type", "").lower()
            if not any(x in ctype for x in ["fusion", "synchro", "xyz", "link"]):
                errors.append(f"Extra Deck card '{info.get('name')}' is not a Fusion, Synchro, Xyz, or Link monster.")

    if errors:
        print("\n--- Deck Validation Errors ---")
        for error in errors:
            print(error)
    else:
        print("\nDeck validation complete: The deck complies with official rules.")

    return errors

# ---------------------------
# Export Deck to .ydk File
# ---------------------------
def export_deck_to_ydk(deck, deck_name):
    """
    Exports the deck as a .ydk file.
    The file format is:
      !name <deck name>
      #main
      <sorted main deck card IDs>
      #extra
      <sorted extra deck card IDs>
      !side
      <sorted side deck card IDs>
    Sorting is by card type for readability.
    """
    filename = f"{deck_name}.ydk"
    sorted_main = sorted(deck["main"], key=lambda cid: get_card_info(cid).get("type", ""))
    sorted_extra = sorted(deck["extra"], key=lambda cid: get_card_info(cid).get("type", ""))
    sorted_side = sorted(deck["side"], key=lambda cid: get_card_info(cid).get("type", ""))

    with open(filename, "w") as f:
        f.write(f"!name {deck_name}\n")
        f.write("#main\n")
        for card in sorted_main:
            f.write(f"{card}\n")
        f.write("#extra\n")
        for card in sorted_extra:
            f.write(f"{card}\n")
        f.write("!side\n")
        for card in sorted_side:
            f.write(f"{card}\n")
    print(f"\nDeck exported to {filename}")

# ---------------------------
# FastAPI Integration
# ---------------------------
app = FastAPI()

class DeckRequest(BaseModel):
    input_cards: List[str]

@app.post("/build-deck")
def build_deck_endpoint(request: DeckRequest):
    # Define desired deck sizes
    MAIN_DECK_SIZE = 40
    EXTRA_DECK_SIZE = 15
    SIDE_DECK_SIZE = 15

    # Use the "ydk_download" folder in the current directory for historical deck files
    deck_directory = os.path.join(os.getcwd(), "ydk_download")
    if not os.path.isdir(deck_directory):
        raise HTTPException(status_code=500, detail=f"Directory '{deck_directory}' not found.")
    deck_files = [os.path.join(deck_directory, f) for f in os.listdir(deck_directory) if f.endswith('.ydk')]
    if not deck_files:
        raise HTTPException(status_code=500, detail="No .ydk files found in the historical decks directory.")

    # Build the decks
    new_main = build_main_deck(request.input_cards, deck_files, desired_main_deck_size=MAIN_DECK_SIZE, dropoff_ratio=0.3)
    new_extra = build_extra_deck(deck_files, desired_extra_deck_size=EXTRA_DECK_SIZE, dropoff_ratio=0.3)
    new_side = build_side_deck(deck_files, desired_side_deck_size=SIDE_DECK_SIZE, dropoff_ratio=0.3)

    errors = validate_deck(new_main, new_side, new_extra)

    return {
        "main": new_main,
        "extra": new_extra,
        "side": new_side,
        "errors": errors
    }

# ---------------------------
# Main Execution: Run API if executed directly
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
