import os
import urllib.parse
from typing import Tuple

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, OWL, XSD
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Card


def add_card_to_graph(g: Graph, card_obj, deck_uri: URIRef, section: str = "main", count: int = 1,
                      base_iri: str = "http://example.org/ygo#") -> Graph:
    """
    Fügt einen einzelnen Karten-Eintrag in den bestehenden RDF-Graphen g hinzu.

    - card_obj: Ein Card-Objekt (z. B. aus der Datenbank) mit den Attributen id, name, atk, defense, level, attribute und den Effekt-Boolean-Feldern.
    - deck_uri: Der URI des Deck-Individuals (z. B. URIRef("http://example.org/ygo#Deck_Deck123"))
    - section: Die Deck-Sektion ("main", "extra" oder "side")
    - count: Wie oft diese Karte hinzugefügt wird (Standard: 1)

    Die Funktion erzeugt einen neuen CardSlot, verknüpft diesen mit dem Deck und der Karte.
    """
    YGO = Namespace(base_iri)
    g.bind("ygo", YGO)

    # Erzeuge den Card-Individual URI
    card_uri = URIRef(f"{base_iri}Card_{card_obj}")
    # Füge den Card-Knoten hinzu (falls noch nicht vorhanden)
    g.add((card_uri, RDF.type, YGO.Card))
    if getattr(card_obj, "name", None):
        g.add((card_uri, YGO.cardName, Literal(card_obj.name, datatype=XSD.string)))
    if getattr(card_obj, "atk", None) is not None:
        g.add((card_uri, YGO.atk, Literal(card_obj.atk, datatype=XSD.integer)))
    if getattr(card_obj, "defense", None) is not None:
        g.add((card_uri, YGO.defense, Literal(card_obj.defense, datatype=XSD.integer)))
    if getattr(card_obj, "level", None) is not None:
        g.add((card_uri, YGO.level, Literal(card_obj.level, datatype=XSD.integer)))
    if getattr(card_obj, "attribute", None):
        g.add((card_uri, YGO.cardAttribute, Literal(card_obj.attribute, datatype=XSD.string)))

    # Füge alle Effekt-Boolean-Felder hinzu (nur wenn True)
    effect_fields = [
        "effect_search", "effect_destroy", "effect_negate", "effect_draw",
        "effect_special_summon", "effect_banish", "effect_send_gy", "effect_recover_lp",
        "effect_inflict_damage", "effect_equip", "effect_modify_stats", "effect_protect",
        "effect_discard", "effect_change_position", "effect_return", "effect_shuffle",
        "effect_copy", "effect_counter", "effect_token_summon", "effect_deck_manipulation"
    ]
    for eff in effect_fields:
        if getattr(card_obj, eff, False) is True:
            eff_prop = URIRef(f"{base_iri}effect_{eff}")
            g.add((card_uri, eff_prop, Literal(True, datatype=XSD.boolean)))

    # Erzeuge einen neuen CardSlot für diese Karte in dem angegebenen Deck und Abschnitt.
    # Der Slot-URI wird aus Deck-URI, Section und Card-ID zusammengesetzt.
    deck_str = str(deck_uri)
    deck_local = deck_str.split("#")[-1] if "#" in deck_str else deck_str
    slot_uri = URIRef(f"{base_iri}{deck_local}_Slot_{section}_{card_obj.id}")
    g.add((slot_uri, RDF.type, YGO.CardSlot))
    g.add((slot_uri, YGO["count"], Literal(count, datatype=XSD.integer)))
    g.add((slot_uri, YGO.section, Literal(section, datatype=XSD.string)))

    # Verknüpfe den Slot mit dem Deck
    g.add((deck_uri, YGO.hasCardSlot, slot_uri))
    # Verknüpfe den Slot mit der Karte
    g.add((slot_uri, YGO.forCard, card_uri))

    return g


def init_deck_graph(session, deck_id, seed_cards, base_iri="http://example.org/ygo#",
                    tbox_file="ygo_ontology_base.ttl") -> tuple[Graph, URIRef]:
    """
    Erzeugt einen initialen RDF-Graphen für ein Deck, indem das Deck-Individual angelegt
    und alle Seed-Karten (a
    ls Card-Objekte) mit jeweils einem CardSlot (Section "main" und count=1)
    hinzugefügt werden.

    Parameter:
      - session: SQLAlchemy-Session (z. B. für DB-Abfragen, falls benötigt)
      - deck_id: Eine Kennung für das Deck (z. B. "Deck123")
      - seed_cards: Eine Liste von Card-Objekten, die als Seed-Karten dienen
      - base_iri: Basis-IRI, z. B. "http://example.org/ygo#"
      - tbox_file: Pfad zur lokalen T-Box-Datei (im Turtle-Format)

    Rückgabe:
      - Den initialen, erweiterten RDF-Graphen.
    """
    # Erstelle einen neuen RDF-Graphen
    g = Graph()
    YGO = Namespace(base_iri)
    g.bind("ygo", YGO)

    # Lade die T-Box aus der lokalen Datei und integriere sie in den Graphen.
    import os, urllib.parse
    tbox_path = os.path.join(os.path.dirname(__file__), tbox_file)
    if os.path.exists(tbox_path):
        g.parse(tbox_path, format="turtle")
        print(f"T-Box aus '{tbox_path}' erfolgreich geladen.")
    else:
        print(f"T-Box-Datei '{tbox_path}' nicht gefunden.")

    # Erzeuge einen Ontology-Knoten und füge owl:imports hinzu.
    abs_tbox_path = os.path.abspath(tbox_path)
    tbox_uri_str = "file:///" + urllib.parse.quote(abs_tbox_path.replace("\\", "/"))
    tbox_uri = URIRef(tbox_uri_str)
    ont = URIRef(f"{base_iri}Deck_{deck_id}_Ontology")
    g.add((ont, RDF.type, OWL.Ontology))
    g.add((ont, OWL.imports, tbox_uri))

    # Erzeuge das Deck-Individual (vorausgesetzt, ygo:Deck ist in der T-Box definiert)
    deck_uri = URIRef(f"{base_iri}Deck_{deck_id}")
    g.add((deck_uri, RDF.type, YGO.Deck))
    g.add((deck_uri, YGO.cardName, Literal(deck_id, datatype=XSD.string)))

    # Füge für jede Seed-Karte einen CardSlot hinzu (Section "main", count=1)
    for card_obj in seed_cards:
        g = add_card_to_graph(g, card_obj, deck_uri, section="main", count=1, base_iri=base_iri)

    return g, deck_uri  # Hier wird ein Tupel zurückgegeben


# Beispiel-Hauptprogramm zum Initialisieren eines Deck-Graphen
if __name__ == "__main__":
    # SQLAlchemy-Engine und Session einrichten (DB-Pfad ggf. anpassen)
    engine = create_engine("sqlite:///data.sqlite")
    Session = sessionmaker(bind=engine)
    session = Session()

    # Beispiel: Liste von Seed-Karten aus der DB abrufen (z. B. per IDs)
    # Hier nehmen wir an, dass du eine Liste von Card-Objekten hast.
    seed_card_ids = [89631139, 6983839]  # Beispiel-IDs
    seed_cards = []
    for cid in seed_card_ids:
        card = session.query(Card).filter(Card.id == cid).one_or_none()
        if card:
            seed_cards.append(card)
        else:
            print(f"Karte mit ID {cid} nicht gefunden.")

    # Initialen Graph für das Deck erzeugen
    deck_id = "Deck123"
    g = init_deck_graph(session, deck_id, seed_cards, base_iri="http://example.org/ygo#",
                        tbox_file="ygo_ontology_base.ttl")

    # Graph in Turtle-Format serialisieren
    output_file = "../../../initial_deck_abox.ttl"
    g.serialize(destination=output_file, format="turtle")
    print(f"Initialer Deck-Graph wurde in '{output_file}' geschrieben.")
