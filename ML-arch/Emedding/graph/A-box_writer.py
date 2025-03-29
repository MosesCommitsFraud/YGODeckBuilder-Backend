import os
import json
import urllib.parse
from collections import Counter
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, OWL, XSD
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Card  # Passe diesen Import an, falls dein Modell woanders liegt


def parse_ydk_file(file_path):
    """
    Liest eine .ydk-Datei und gibt ein Dict zurück mit den Keys:
    'main', 'extra', 'side' – jeweils Listen von Card-IDs (als Strings).
    """
    main_cards = []
    extra_cards = []
    side_cards = []
    current_section = None

    with open(file_path, "r", encoding="utf-8") as f:
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
            else:
                if current_section == "main":
                    main_cards.append(line)
                elif current_section == "extra":
                    extra_cards.append(line)
                elif current_section == "side":
                    side_cards.append(line)
    return {"main": main_cards, "extra": extra_cards, "side": side_cards}


def build_abox_from_ydk(
        session,
        ydk_file,
        deck_id="MyDeck",
        base_iri="http://example.org/ygo#",
        tbox_file="ygo_ontology_base.ttl",
        output_file="my_deck_abox.ttl"
):
    """
    Erzeugt eine RDF A-Box (im Turtle-Format) für ein Deck.

    1) Lädt T-Box aus lokaler Datei (z.B. "ygo_ontology_base.ttl")
    2) Legt ein Deck-Individual an
    3) Erzeugt pro Karte in main/extra/side ein CardSlot-Individual + Tripel
       (Deck -> hasCardSlot -> Slot, Slot -> count, Slot -> section, Slot -> forCard -> Card_X).
    4) Erzeugt Card_Instanzen mit Daten (Name, ATK, DEF, usw.) aus der DB
    5) Referenzen:
       - IDs in card_obj.referenced_cards (z.B. "[12345678]")
       - Namen in card_obj.referenced_cards_by_name (z.B. "[\"Blue-Eyes White Dragon\", ...]")
         => Hier wird in der DB eine Karte per Name gesucht. Falls nicht existiert, wird sie neu angelegt.
         => Dann wird ein Slot für diese Karte ins selbe Deck gepackt (falls noch nicht vorhanden)
         => ygo:refersTo-Kante von der ursprünglichen Karte zu diesem referenced-Objekt
    """
    g = Graph()
    YGO = Namespace(base_iri)
    g.bind("ygo", YGO)

    # 1) T-Box laden
    tbox_path = os.path.join(os.path.dirname(__file__), tbox_file)
    if os.path.exists(tbox_path):
        g.parse(tbox_path, format="turtle")
        print(f"T-Box aus '{tbox_path}' erfolgreich geladen und in den Graphen integriert.")
    else:
        print(f"T-Box-Datei '{tbox_path}' nicht gefunden. Fortfahren ohne T-Box.")

    # 2) Ontology-Knoten + owl:imports
    abs_tbox_path = os.path.abspath(tbox_path)
    tbox_uri_str = "file:///" + urllib.parse.quote(abs_tbox_path.replace("\\", "/"))
    tbox_uri = URIRef(tbox_uri_str)
    ont = URIRef(f"{base_iri}Deck_{deck_id}_Ontology")
    g.add((ont, RDF.type, OWL.Ontology))
    g.add((ont, OWL.imports, tbox_uri))

    # 3) YDK parsen
    data = parse_ydk_file(ydk_file)
    main_list = data["main"]
    extra_list = data["extra"]
    side_list = data["side"]

    # 4) Häufigkeiten
    main_counts = Counter(main_list)
    extra_counts = Counter(extra_list)
    side_counts = Counter(side_list)
    all_card_ids = set(main_counts.keys()) | set(extra_counts.keys()) | set(side_counts.keys())

    # 5) Deck-Individual
    deck_uri = URIRef(f"{base_iri}Deck_{deck_id}")
    g.add((deck_uri, RDF.type, YGO.Deck))
    g.add((deck_uri, YGO.cardName, Literal(deck_id, datatype=XSD.string)))

    # Effekt-Felder
    effect_fields = [
        "effect_search", "effect_destroy", "effect_negate", "effect_draw",
        "effect_special_summon", "effect_banish", "effect_send_gy", "effect_recover_lp",
        "effect_inflict_damage", "effect_equip", "effect_modify_stats", "effect_protect",
        "effect_discard", "effect_change_position", "effect_return", "effect_shuffle",
        "effect_copy", "effect_counter", "effect_token_summon", "effect_deck_manipulation"
    ]

    def ensure_card_in_graph(card_: Card):
        """
        Fügt (falls noch nicht vorhanden) ein Card-Individual für card_ in den Graphen g ein,
        mit den Grund-Properties (Name, ATK, etc.).
        Gibt card_uri zurück.
        """
        card_uri = URIRef(f"{base_iri}Card_{card_.id}")
        # (card_uri, RDF.type, YGO.Card) nur hinzufügen, wenn wir das Triple noch nicht haben
        # (rdflib nimmt's zwar nicht doppelt, aber wir können checken)
        if (card_uri, RDF.type, YGO.Card) not in g:
            g.add((card_uri, RDF.type, YGO.Card))

            if card_.name:
                g.add((card_uri, YGO.cardName, Literal(card_.name, datatype=XSD.string)))
            if card_.atk is not None:
                g.add((card_uri, YGO.atk, Literal(card_.atk, datatype=XSD.integer)))
            if card_.defense is not None:
                g.add((card_uri, YGO.defense, Literal(card_.defense, datatype=XSD.integer)))
            if card_.level is not None:
                g.add((card_uri, YGO.level, Literal(card_.level, datatype=XSD.integer)))
            if card_.attribute:
                g.add((card_uri, YGO.cardAttribute, Literal(card_.attribute, datatype=XSD.string)))

            # Ban-Status
            if card_.ban_tcg:
                g.add((card_uri, YGO.banTcg, Literal(card_.ban_tcg, datatype=XSD.string)))
            if card_.ban_ocg:
                g.add((card_uri, YGO.banOcg, Literal(card_.ban_ocg, datatype=XSD.string)))
            if card_.ban_goat:
                g.add((card_uri, YGO.banGoat, Literal(card_.ban_goat, datatype=XSD.string)))

            # Referenced Archetypes / Races
            if card_.referenced_archetypes:
                g.add((card_uri, YGO.referencedArchetypes, Literal(card_.referenced_archetypes, datatype=XSD.string)))
            if card_.referenced_races:
                g.add((card_uri, YGO.referencedRaces, Literal(card_.referenced_races, datatype=XSD.string)))

            # Bilder
            if hasattr(card_, "images") and card_.images:
                first_image = card_.images[0]
                if hasattr(first_image, "image_url_small") and first_image.image_url_small:
                    g.add((card_uri, YGO.ygoprodeckURL, Literal(first_image.image_url_small, datatype=XSD.anyURI)))

            # Effekt-Felder
            for eff in effect_fields:
                if getattr(card_, eff, False) is True:
                    eff_prop = URIRef(f"{base_iri}effect_{eff}")
                    g.add((card_uri, eff_prop, Literal(True, datatype=XSD.boolean)))

        return card_uri

    def ensure_slot_in_deck(card_uri: URIRef, section: str = "main", cnt: int = 1):
        """
        Falls es für card_uri in 'deck_id' und 'section' noch keinen Slot gibt, legt einen an.
        Gibt den Slot-URI zurück.
        """
        # z.B. http://example.org/ygo#Deck_Deck123_Slot_main_89631139
        slot_uri = URIRef(f"{base_iri}Deck_{deck_id}_Slot_{section}_{card_uri.split('#Card_')[-1]}")
        if (slot_uri, RDF.type, YGO.CardSlot) not in g:
            g.add((slot_uri, RDF.type, YGO.CardSlot))
            g.add((slot_uri, YGO["count"], Literal(cnt, datatype=XSD.integer)))
            g.add((slot_uri, YGO.section, Literal(section, datatype=XSD.string)))
            g.add((deck_uri, YGO.hasCardSlot, slot_uri))
            g.add((slot_uri, YGO.forCard, card_uri))
        return slot_uri

    def add_slots_for_section(counts: Counter, section_name: str):
        """
        Legt für jede ID in 'counts' (z.B. main_counts) einen Slot an
        und fügt Card-Properties / Referenzen hinzu.
        """
        for c_id, count_val in counts.items():
            # Versuche, Karte aus DB zu holen:
            card_obj = session.query(Card).filter(Card.id == c_id).one_or_none()
            if card_obj:
                # Karte + Properties in Graph
                card_uri = ensure_card_in_graph(card_obj)

                # Slot in Deck
                slot_uri = ensure_slot_in_deck(card_uri, section_name, count_val)


                # --- NEU: Referenzen per Name ---
                if hasattr(card_obj, "referenced_cards") and card_obj.referenced_cards:
                    try:
                        ref_name_list = json.loads(card_obj.referenced_cards)
                        for ref_name in ref_name_list:
                            ref_card_obj = session.query(Card).filter_by(name=ref_name).one_or_none()
                            if not ref_card_obj:
                                # Wenn nicht gefunden: Neue Karte in DB anlegen (nur mit Name)
                                ref_card_obj = Card(name=ref_name)
                                session.add(ref_card_obj)
                                session.commit()
                                print(f"Info: Neue Karte '{ref_name}' in DB angelegt (id={ref_card_obj.id}).")

                            # Nun Card_Individual + Slot anlegen
                            ref_card_uri = ensure_card_in_graph(ref_card_obj)
                            g.add((card_uri, YGO.refersTo, ref_card_uri))
                            # In 'main' einfügen, falls noch nicht vorhanden
                            ensure_slot_in_deck(ref_card_uri, "main", 1)

                    except Exception as e:
                        print(f"Fehler beim Parsen von referenced_cards_by_name (Name) für Card {c_id}: {e}")

            else:
                # Karte existiert nicht in DB => minimaler Eintrag
                card_uri = URIRef(f"{base_iri}Card_{c_id}")
                if (card_uri, RDF.type, YGO.Card) not in g:
                    g.add((card_uri, RDF.type, YGO.Card))

                # Slot in Deck
                ensure_slot_in_deck(card_uri, section_name, count_val)

    # Hauptteil: Slots für main/extra/side
    add_slots_for_section(main_counts, "main")
    add_slots_for_section(extra_counts, "extra")
    add_slots_for_section(side_counts, "side")

    # Zum Schluss den Graphen serialisieren
    g.serialize(destination=output_file, format="turtle")
    print(f"A-Box geschrieben in '{output_file}'.")




if __name__ == "__main__":
    # SQLAlchemy-Engine und Session einrichten (passe den DB-Pfad ggf. an)
    engine = create_engine("sqlite:///../../../data.sqlite")
    Session = sessionmaker(bind=engine)
    session = Session()

    # Pfad zur YDK-Datei (anpassen!)
    ydk_file = "Blue-eyes_Branded.ydk"

    # A-Box generieren: deck_id "Deck123", T-Box-Datei "ygo_ontology_base.ttl" im selben Verzeichnis
    build_abox_from_ydk(
        session,
        ydk_file,
        deck_id="Deck123",
        tbox_file="ygo_ontology_base.ttl",
        base_iri="http://example.org/ygo#",
        output_file="deck_blue_Eyes.ttl"
    )
