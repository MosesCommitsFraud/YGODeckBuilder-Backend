from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
import traceback

# Import the functions from main.py - make sure main.py is in the same directory
# or adjust the import path as needed
from main import (
    load_card_info,
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
    ARCHETYPES_FILE,
    YDK_FOLDER,
    TARGET_MAX_MAIN,
    TARGET_EXTRA_SIZE,
    TARGET_SIDE_SIZE
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

        # Load necessary data
        card_info = load_card_info()
        with open(ARCHETYPES_FILE, "r", encoding="utf-8") as f:
            archetypes = json.load(f)

        # Clear cache and load decks
        clear_cache()
        full_decks = load_full_decks(YDK_FOLDER)
        main_decks = [deck["main"] for deck in full_decks]

        # Process valid input cards
        valid_input_extra = [card for card in extra_card_ids if is_valid_extra_deck_card(card, card_info)]
        valid_input_side = [card for card in side_card_ids
                            if not is_extra_deck_card(card, card_info) or is_valid_extra_deck_card(card, card_info)]

        # Start with input cards for main deck
        new_main_deck = list(main_card_ids)

        print("Analyzing deck archetype distribution...")
        # Calculate deck distribution
        desired_distribution = analyze_archetype_distribution(
            main_card_ids, card_info, archetypes, main_decks, target_size=TARGET_MAX_MAIN
        )

        # Get average copies
        avg_main = compute_average_copies_main(main_decks)

        print("Building main deck...")
        # Build main deck
        new_main_deck = apply_flexible_distribution(
            new_main_deck, main_decks, card_info, desired_distribution,
            avg_main, main_card_ids, archetypes
        )

        print("Building extra deck...")
        # Build extra deck
        all_input_cards = main_card_ids + valid_input_extra
        new_extra_deck = build_extra_deck_optimized(all_input_cards, full_decks, TARGET_EXTRA_SIZE, card_info,
                                                    archetypes)

        print("Building side deck...")
        # Build side deck
        new_side_deck = valid_input_side[:]
        if len(new_side_deck) < TARGET_SIDE_SIZE:
            side_cards = build_side_deck_optimized(
                all_input_cards, full_decks, TARGET_SIDE_SIZE - len(new_side_deck),
                card_info, archetypes, new_main_deck, new_extra_deck
            )
            new_side_deck.extend([card for card in side_cards if card not in new_side_deck])
            new_side_deck = new_side_deck[:TARGET_SIDE_SIZE]

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

        print("Deck generation complete")

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