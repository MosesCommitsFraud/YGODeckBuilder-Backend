from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
import traceback
from collections import Counter, defaultdict

# Import the essential functions from main.py
from main import (
    load_card_info as original_load_card_info,
    clear_cache,
    is_valid_extra_deck_card,
    is_extra_deck_card,
    build_complete_deck_improved,
    load_full_decks,
    ARCHETYPES_FILE,
    YDK_FOLDER,
    TARGET_MIN_MAIN,
    TARGET_MAX_MAIN,
    TARGET_EXTRA_SIZE,
    TARGET_SIDE_SIZE,
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

        # Combine all input cards for analysis
        all_input_cards = main_card_ids + extra_card_ids + side_card_ids

        # Check if we have any input cards at all
        if not all_input_cards:
            # No input cards, we'll need to create a generic deck
            raise HTTPException(status_code=400, detail="No valid cards provided for deck generation.")

        # Build the complete deck using our improved algorithm
        new_deck = build_complete_deck_improved(all_input_cards, full_decks, card_info, archetypes)

        # Extract the deck components
        new_main_deck = new_deck["main"]
        new_extra_deck = new_deck["extra"]
        new_side_deck = new_deck["side"]

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