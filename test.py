import spacy
import re
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Card  # Wir nutzen hier nur das Card-Modell

# spaCy-Modell laden (englisch; passe an, falls du deutsch bevorzugst)
nlp = spacy.load("en_core_web_sm")


def load_all_card_attributes(db_url="sqlite:///data.sqlite"):
    """
    Lädt alle Karten aus der DB und gibt drei Sets zurück:
      - card_names: Alle Kartennamen (in lowercase)
      - archetypes: Alle Archetypen (Originalschreibweise)
      - races: Alle Races (Originalschreibweise)
    """
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    card_names = set()
    archetypes = set()
    races = set()

    cards = session.query(Card).all()
    for card in cards:
        if card.name:
            card_names.add(card.name.lower())
        if card.archetype:
            archetypes.add(card.archetype)
        if card.race:
            races.add(card.race)
    session.close()
    return card_names, archetypes, races


def filter_substrings(strings):
    """
    Filtert eine Liste von Strings so, dass kürzere Strings, die Teil eines längeren Strings sind, entfernt werden.
    Beispiel: aus ["Starry Knight", "Star", "Knight"] bleibt nur ["Starry Knight"].
    """
    sorted_strings = sorted(strings, key=lambda s: len(s), reverse=True)
    final = []
    for s in sorted_strings:
        if not any(s.lower() in t.lower() and s.lower() != t.lower() for t in final):
            final.append(s)
    return final


def analyze_card_effect(desc, card_names, archetypes, races):
    """
    Analysiert den Effekttext (desc) einer Karte und gibt ein Dictionary zurück,
    das Effekt-Kategorien sowie referenzierte Karten, Archetypen und Races enthält.

    Referenzierte Karten werden aus Textteilen in doppelten Anführungszeichen extrahiert,
    und nur übernommen, wenn sie in card_names (lowercase) vorkommen.

    Für Archetypen und Races wird mittels Regex mit Wortgrenzen geprüft.
    Zusätzlich werden irrelevante Kontexte (z.B. "damage phase", "damage step") ausgeschlossen.
    """
    if not desc:
        return {}

    text = desc  # Originaltext
    text_lower = desc.lower()

    # Effekt-Kategorien anhand einfacher Schlüsselwortsuche
    categories = {
        "effect_search": any(kw in text_lower for kw in ["search your deck", "add to your hand", "fetch"]),
        "effect_destroy": "destroy" in text_lower,
        "effect_negate": "negate" in text_lower,
        "effect_draw": "draw" in text_lower,
        "effect_special_summon": "special summon" in text_lower,
        "effect_banish": "banish" in text_lower,
        "effect_send_gy": (
                "send to the graveyard" in text_lower or
                "send this card to the graveyard" in text_lower or
                ("send" in text_lower and ("gy" in text_lower or "g y" in text_lower))
        ),
        "effect_recover_lp": ("recover lp" in text_lower or "gain lp" in text_lower or "recover life points" in text_lower),
        "effect_inflict_damage": ("damage" in text_lower and "damage phase" not in text_lower and "damage step" not in text_lower),
        "effect_equip": "equip" in text_lower,
        "effect_modify_stats": ("atk" in text_lower or "def" in text_lower),
        "effect_protect": ("cannot be destroyed" in text_lower or "unaffected" in text_lower),
        "effect_discard": "discard" in text_lower,
        "effect_change_position": (("change" in text_lower or "switch" in text_lower) and "position" in text_lower),
        "effect_return": "return" in text_lower,
        "effect_shuffle": "shuffle" in text_lower,
        "effect_copy": "copy" in text_lower,
        "effect_counter": "counter" in text_lower,
        "effect_token_summon": "token" in text_lower,
        "effect_deck_manipulation": ("deck" in text_lower and ("arrange" in text_lower or "order" in text_lower))
    }

    # Referenzierte Karten: Suche nach Text in doppelten Anführungszeichen
    extracted_cards = re.findall(r'"([^"]+)"', text)
    referenced_cards = [ref.strip() for ref in extracted_cards if ref.strip().lower() in card_names]
    referenced_cards = filter_substrings(referenced_cards)
    categories["referenced_cards"] = list(set(referenced_cards))

    # Referenzierte Archetypen: Suche mittels Regex mit Wortgrenzen (case-sensitive)
    matched_archetypes = []
    for arch in archetypes:
        pattern = r'\b' + re.escape(arch) + r'\b'
        if re.search(pattern, text):
            matched_archetypes.append(arch)
    matched_archetypes = filter_substrings(matched_archetypes)
    categories["referenced_archetypes"] = list(set(matched_archetypes))

    # Referenzierte Races: Analog
    matched_races = []
    for race in races:
        pattern = r'\b' + re.escape(race) + r'\b'
        if re.search(pattern, text):
            matched_races.append(race)
    matched_races = filter_substrings(matched_races)
    categories["referenced_races"] = list(set(matched_races))

    return categories


def update_card_effects(db_url="sqlite:///data.sqlite"):
    """
    Lädt alle Karten aus der Datenbank, analysiert den Effekttext (desc)
    und aktualisiert die neuen Spalten:
      - Referenced Felder (als JSON-Text)
      - One-Hot Effekt-Kategorien (Boolean)
    """
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Lade alle bekannten Attribute (für Referenzvergleiche)
    card_names, archetypes, races = load_all_card_attributes(db_url)

    cards = session.query(Card).all()
    for card in cards:
        analysis = analyze_card_effect(card.desc, card_names, archetypes, races)

        # Update One-Hot Effekt-Kategorien
        card.effect_search = analysis.get("effect_search", False)
        card.effect_destroy = analysis.get("effect_destroy", False)
        card.effect_negate = analysis.get("effect_negate", False)
        card.effect_draw = analysis.get("effect_draw", False)
        card.effect_special_summon = analysis.get("effect_special_summon", False)
        card.effect_banish = analysis.get("effect_banish", False)
        card.effect_send_gy = analysis.get("effect_send_gy", False)
        card.effect_recover_lp = analysis.get("effect_recover_lp", False)
        card.effect_inflict_damage = analysis.get("effect_inflict_damage", False)
        card.effect_equip = analysis.get("effect_equip", False)
        card.effect_modify_stats = analysis.get("effect_modify_stats", False)
        card.effect_protect = analysis.get("effect_protect", False)
        card.effect_discard = analysis.get("effect_discard", False)
        card.effect_change_position = analysis.get("effect_change_position", False)
        card.effect_return = analysis.get("effect_return", False)
        card.effect_shuffle = analysis.get("effect_shuffle", False)
        card.effect_copy = analysis.get("effect_copy", False)
        card.effect_counter = analysis.get("effect_counter", False)
        card.effect_token_summon = analysis.get("effect_token_summon", False)
        card.effect_deck_manipulation = analysis.get("effect_deck_manipulation", False)

        # Speichere referenzierte Informationen als JSON-Strings
        card.referenced_cards = json.dumps(analysis.get("referenced_cards", []))
        card.referenced_archetypes = json.dumps(analysis.get("referenced_archetypes", []))
        card.referenced_races = json.dumps(analysis.get("referenced_races", []))

        print(f"Card ID {card.id} - {card.name} aktualisiert.")

    session.commit()
    session.close()
    print("Alle Karten wurden aktualisiert.")


if __name__ == "__main__":
    update_card_effects()
