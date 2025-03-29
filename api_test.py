import json
from fastapi import FastAPI, Body, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text

# Importiere Engine und den Kontextmanager get_session aus deinem db-Paket:
from db.db import engine, get_session
from db.models import Card, CardPrice, CardImage, CardSet

app = FastAPI()

# Wenn dein Frontend z.B. auf Port 3000 läuft:
origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
