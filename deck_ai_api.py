from fastapi import FastAPI, HTTPException, Body, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
import traceback
from collections import Counter, defaultdict
import pickle
import numpy as np
import os
from sqlalchemy.orm import Session
from sqlalchemy import text

# Importiere Engine und den Kontextmanager get_session aus deinem db-Paket:
from db.db import engine, get_session
from db.models import Card, CardPrice, CardImage, CardSet

# Import the essential functions from main.py
from main import (
    load_card_info as original_load_card_info,
    clear_cache,
    build_complete_deck_improved,
    load_full_decks,
    ARCHETYPES_FILE,
    YDK_FOLDER,
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

# --- Global variables ---
EMBEDDINGS = {}


# --- Similarity Functions ---
def load_embeddings(filename="graph_embeddings.pkl"):
    """
    Loads graph embeddings from the pickle file.
    """
    global EMBEDDINGS
    if EMBEDDINGS:
        return EMBEDDINGS

    if os.path.exists(filename):
        with open(filename, "rb") as f:
            EMBEDDINGS = pickle.load(f)
        return EMBEDDINGS
    else:
        raise FileNotFoundError(f"Embedding file {filename} not found")


def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors.
    """
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def get_similar_cards(selected_card_id, top_n=5):
    """
    Calculates similarity between the selected card and all other cards.
    Returns a list of the top-n cards sorted by descending similarity.
    """
    if selected_card_id not in EMBEDDINGS:
        return []

    selected_embedding = EMBEDDINGS[selected_card_id]
    similarities = {}
    for card_id, emb in EMBEDDINGS.items():
        if card_id == selected_card_id:
            continue
        sim = cosine_similarity(selected_embedding, emb)
        similarities[card_id] = sim
    sorted_cards = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_cards[:top_n]


# --- Models for Similarity Endpoint ---
class SimilarCard(BaseModel):
    card_id: str
    name: Optional[str] = None
    type: Optional[str] = None
    archetype: Optional[str] = None
    similarity: float
    image_url: Optional[str] = None


class SimilarCardsResponse(BaseModel):
    selected_card: Dict[str, Any]
    similar_cards: List[SimilarCard]


# Define request and response models for the existing API
class CardSchema(BaseModel):
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
    card: CardSchema


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


# --- NEW API ENDPOINT for similar cards ---
@app.get("/api/similar/{card_id}", response_model=SimilarCardsResponse)
async def get_similar_cards_endpoint(card_id: str, top_n: int = 10):
    """
    Returns similar cards based on graph embeddings.
    """
    try:
        # Load embeddings if not already loaded
        if not EMBEDDINGS:
            load_embeddings()

        # Check if card exists in embeddings
        if card_id not in EMBEDDINGS:
            raise HTTPException(
                status_code=404,
                detail=f"Card with ID {card_id} not found in embeddings"
            )

        # Load card info for names and other details
        card_info = load_card_info()

        # Get similar cards
        similar = get_similar_cards(card_id, top_n=top_n)

        # Get info about the selected card
        selected_card_info = card_info.get(card_id, {"id": card_id, "name": "Unknown Card"})

        # Format the similar cards
        similar_cards = []
        for similar_id, similarity in similar:
            card_data = card_info.get(similar_id, {})
            similar_cards.append(SimilarCard(
                card_id=similar_id,
                name=card_data.get("name", "Unknown Card"),
                type=card_data.get("type", ""),
                archetype=card_data.get("archetype", ""),
                similarity=float(similarity),
                image_url=card_data.get("card_images", [{}])[0].get("image_url") if card_data.get(
                    "card_images") else None
            ))

        return SimilarCardsResponse(
            selected_card=selected_card_info,
            similar_cards=similar_cards
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Error getting similar cards: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {
        "message": "Yu-Gi-Oh Deck AI API",
        "endpoints": [
            "/api/generate-deck",
            "/api/similar/{card_id}",
            "/api/health"
            "/card/{card_id}",
            "/graph"
        ]
    }


# --- App startup event ---
@app.on_event("startup")
async def startup_event():
    """
    Load embeddings on startup to avoid delay on first request.
    """
    try:
        print("Loading card embeddings...")
        load_embeddings()
        print(f"Loaded embeddings for {len(EMBEDDINGS)} cards")
    except FileNotFoundError:
        print("Warning: Embeddings file not found. /api/similar endpoint will not work until file is available.")
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        print("The /api/similar endpoint may not work correctly.")


def serialize_card_set(card_set: CardSet) -> Dict[str, Any]:
    return {
        "id": card_set.id,
        "set_name": card_set.set_name,
        "set_code": card_set.set_code,
        "set_rarity": card_set.set_rarity,
        "set_rarity_code": card_set.set_rarity_code,
        "set_price": card_set.set_price,
    }

def serialize_card_image(card_image: CardImage) -> Dict[str, Any]:
    return {
        "id": card_image.id,
        "image_url": card_image.image_url,
        "image_url_small": card_image.image_url_small,
        "image_url_cropped": card_image.image_url_cropped,
    }

def serialize_card_price(card_price: CardPrice) -> Dict[str, Any]:
    return {
        "id": card_price.id,
        "cardmarket_price": card_price.cardmarket_price,
        "tcgplayer_price": card_price.tcgplayer_price,
        "ebay_price": card_price.ebay_price,
        "amazon_price": card_price.amazon_price,
        "coolstuffinc_price": card_price.coolstuffinc_price,
    }
def serialize_card(card: Card) -> Dict[str, Any]:
    return {
        "id": card.id,
        "name": card.name,
        "type": card.type,
        "human_readable_card_type": card.human_readable_card_type,
        "frame_type": card.frame_type,
        "desc": card.desc,
        "race": card.race,
        "archetype": card.archetype,
        "ygoprodeck_url": card.ygoprodeck_url,
        "is_staple": card.is_staple,
        "atk": card.atk,
        "defense": card.defense,
        "level": card.level,
        "attribute": card.attribute,
        "ban_tcg": card.ban_tcg,
        "ban_ocg": card.ban_ocg,
        "ban_goat": card.ban_goat,
        "referenced_cards": card.referenced_cards,
        "referenced_archetypes": card.referenced_archetypes,
        "referenced_races": card.referenced_races,
        "effect_search": card.effect_search,
        "effect_destroy": card.effect_destroy,
        "effect_negate": card.effect_negate,
        "effect_draw": card.effect_draw,
        "effect_special_summon": card.effect_special_summon,
        "effect_banish": card.effect_banish,
        "effect_send_gy": card.effect_send_gy,
        "effect_recover_lp": card.effect_recover_lp,
        "effect_inflict_damage": card.effect_inflict_damage,
        "effect_equip": card.effect_equip,
        "effect_modify_stats": card.effect_modify_stats,
        "effect_protect": card.effect_protect,
        "effect_discard": card.effect_discard,
        "effect_change_position": card.effect_change_position,
        "effect_return": card.effect_return,
        "effect_shuffle": card.effect_shuffle,
        "effect_copy": card.effect_copy,
        "effect_counter": card.effect_counter,
        "effect_token_summon": card.effect_token_summon,
        "effect_deck_manipulation": card.effect_deck_manipulation,
        "sets": [serialize_card_set(s) for s in card.sets],
        "images": [serialize_card_image(i) for i in card.images],
        "prices": [serialize_card_price(p) for p in card.prices],
    }

# Dependency: Pro Request eine DB-Session über den Kontextmanager öffnen
def get_db():
    with get_session() as session:
        yield session

@app.get("/card/{card_id}")

def get_card(card_id: int, db: Session = Depends(get_db)):
    """
    Gibt die vollständigen Card-Daten zurück, basierend auf der Card-ID.
    Es wird geprüft, ob die Karte existiert. Falls gefunden, wird das
    Card-Objekt (serialisiert mittels serialize_card) zurückgegeben.
    """
    card = db.query(Card).filter(Card.id == card_id).first()
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    return serialize_card(card)

@app.post("/graph")
def get_graph_data(
    data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Erwartet im Body:
    {
      "decks": [
        {
          "name": "Mermail Deck",
          "main": [ { "41546": 2 }, { "32864": 1 } ],
          "extra": [],
          "side": []
        }
      ]
    }

    Gibt zurück:
    {
      "nodes": [...],
      "links": [...]
    }
    """
    decks = data.get("decks", [])
    nodes = []
    links = []
    node_ids_set = set()  # zum Vermeiden doppelter Nodes

    def add_node(node_id: str, label: str, extra_data: dict = None):
        """Fügt einen Node nur hinzu, wenn noch nicht existierend."""
        if node_id in node_ids_set:
            return
        node_ids_set.add(node_id)
        node_obj = {
            "id": node_id,
            "label": label,
            "x": None,
            "y": None,
            "z": None,
        }
        if extra_data:
            node_obj.update(extra_data)
        nodes.append(node_obj)

    def add_link(source: str, target: str, relation: str, reference_type: str = None):
        """
        Fügt einen Link hinzu. Dabei wird ein extra Feld 'referenceType' hinzugefügt,
        das angibt, aus welcher Referenz diese Relation stammt. Zudem ist der Link directed.
        """
        link_obj = {
            "source": source,
            "target": target,
            "relation": relation,
            "directed": True
        }
        if reference_type:
            link_obj["referenceType"] = reference_type
        links.append(link_obj)

    def parse_json_field(field_value: str) -> List[str]:
        """
        Versucht, das Feld als JSON zu parsen.
        Falls dies fehlschlägt, wird eine manuelle Bereinigung vorgenommen.
        """
        try:
            parsed = json.loads(field_value)
            if isinstance(parsed, list):
                return parsed
            return [str(parsed)]
        except Exception:
            # Entferne führende und abschließende Zeichen [ ] und "
            cleaned = field_value.strip('[]"')
            # Falls mehrere Werte per Komma getrennt sind
            return [v.strip() for v in cleaned.split(",") if v.strip()]

    # Hilfsfunktion: Legt eine DB-Karte als Node an, unter Nutzung einer SQL-Abfrage
    def add_card_node(card_id_str: str):
        node_id = f"Card_{card_id_str}"
        if node_id in node_ids_set:
            return

        try:
            card_id_int = int(card_id_str)
        except ValueError:
            card_id_int = None

        if card_id_int is not None:
            sql = text("""
                SELECT cards.*, card_images.image_url as image_url
                FROM cards
                LEFT JOIN card_images ON cards.id = card_images.card_id
                WHERE cards.id = :card_id
                LIMIT 1
            """)
            result = db.execute(sql, {"card_id": card_id_int}).mappings().fetchone()
        else:
            result = None

        if result:
            effects = []
            if result["effect_destroy"]:
                effects.append("effect_destroy")
            if result["effect_negate"]:
                effects.append("effect_negate")
            if result["effect_special_summon"]:
                effects.append("effect_special_summon")
            extra_data = {
                "archetype": result["archetype"],
                "cardAttribute": result["attribute"],
                "atk": result["atk"],
                "defense": result["defense"],
                "race": result["race"],
                "level": result["level"],
                "cardDesc": result["desc"],
                "effects": effects,
                "banTcg": result["ban_tcg"],
                "banOcg": result["ban_ocg"],
                "banGoat": result["ban_goat"],
                "imageUrl": result["image_url"],
            }
            # Verarbeite das Feld referenced_card – es können mehrere Werte vorhanden sein:
            if result.get("referenced_cards"):
                referenced_field = result["referenced_cards"]
                extra_data["referencedCard"] = parse_json_field(referenced_field)
            # Ebenso für referenced_archetype:
            if result.get("referenced_archetypes"):
                referenced_field = result["referenced_archetypes"]
                extra_data["referencedArchetype"] = parse_json_field(referenced_field)
            # Und falls es ein Feld für referenzierte Races gibt:
            if result.get("referenced_races"):
                referenced_field = result["referenced_races"]
                extra_data["referencedRace"] = parse_json_field(referenced_field)

            add_node(node_id, result["name"], extra_data)
        else:
            add_node(node_id, f"Card {card_id_str}")

    # Durchlaufe alle Decks
    for deck_info in decks:
        deck_name = deck_info.get("name", "Unnamed Deck")
        deck_node_id = f"deck:{deck_name}"
        add_node(deck_node_id, deck_name, {"deckName": deck_name})

        # Set zum Speichern der im Deck verwendeten Karten-Node-IDs
        deck_card_ids = set()

        def handle_section(section_name: str, card_list: List[Dict[str, int]]):
            for card_count_map in card_list:  # z.B. { "41546": 2 }
                for card_id_str, count in card_count_map.items():
                    add_card_node(card_id_str)
                    card_node_id = f"Card_{card_id_str}"
                    deck_card_ids.add(card_node_id)
                    # Hier wird zusätzlich der section_name als reference_type mitgegeben.
                    add_link(
                        deck_node_id,
                        card_node_id,
                        f"in_{section_name}_x{count}",
                        reference_type=section_name
                    )

        handle_section("main", deck_info.get("main", []))
        handle_section("extra", deck_info.get("extra", []))
        handle_section("side", deck_info.get("side", []))

        # Erzeuge Verbindungen basierend auf Referenzen innerhalb des Decks:
        deck_card_nodes = [node for node in nodes if node["id"] in deck_card_ids]

        # Mapping: Kartenname -> Node-ID (wird für referencedCard benötigt)
        card_name_to_id = {node["label"]: node["id"] for node in deck_card_nodes}

        for card in deck_card_nodes:
            # Verknüpfung für referencedCard:
            referenced_cards = card.get("referencedCard", [])
            if isinstance(referenced_cards, str):
                referenced_cards = [referenced_cards]
            for ref_card in referenced_cards:
                target_id = card_name_to_id.get(ref_card)
                if target_id:
                    add_link(card["id"], target_id, "ref_card", reference_type="referencedCard")
            # Verknüpfung für referencedArchetype:
            referenced_archetypes = card.get("referencedArchetype", [])
            if isinstance(referenced_archetypes, str):
                referenced_archetypes = [referenced_archetypes]
            for ref_arch in referenced_archetypes:
                for other in deck_card_nodes:
                    if other["id"] != card["id"] and other.get("archetype") == ref_arch:
                        add_link(card["id"], other["id"], "ref_archetype", reference_type="referencedArchetype")
            # Verknüpfung für referencedRace:
            referenced_races = card.get("referencedRace", [])
            if isinstance(referenced_races, str):
                referenced_races = [referenced_races]
            for ref_race in referenced_races:
                for other in deck_card_nodes:
                    if other["id"] != card["id"] and other.get("race") == ref_race:
                        add_link(card["id"], other["id"], "ref_race", reference_type="referencedRace")

    print({"nodes": nodes, "links": links})
    return {"nodes": nodes, "links": links}


if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run("deck_ai_api:app", host="0.0.0.0", port=8000, reload=True)