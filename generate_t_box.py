"""
Erzeugt eine T-Box (ygo_ontology_base.ttl) für das 'Card'-Datenmodell.
Darin legen wir die Klasse 'Card' und diverse DatatypeProperties an.
Spätere Instanzen (ABox) können diese T-Box importieren.
"""

from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, Literal

def build_cards_tbox(output_file="ygo_ontology_base.ttl"):
    g = Graph()

    # Haupt-Namespace
    YGO = Namespace("http://example.org/ygo#")
    g.bind("ygo", YGO)

    # Deklariere die Klasse 'Card'
    g.add((YGO.Card, RDF.type, OWL.Class))
    g.add((YGO.Card, RDFS.label, Literal("A generic Yu-Gi-Oh! Card entity", datatype=XSD.string)))

    # Beispiel: DatatypeProperty für 'name'
    g.add((YGO.cardName, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.cardName, RDFS.domain, YGO.Card))
    g.add((YGO.cardName, RDFS.range, XSD.string))

    # type
    g.add((YGO.cardType, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.cardType, RDFS.domain, YGO.Card))
    g.add((YGO.cardType, RDFS.range, XSD.string))

    # human_readable_card_type
    g.add((YGO.humanReadableCardType, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.humanReadableCardType, RDFS.domain, YGO.Card))
    g.add((YGO.humanReadableCardType, RDFS.range, XSD.string))

    # frame_type
    g.add((YGO.frameType, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.frameType, RDFS.domain, YGO.Card))
    g.add((YGO.frameType, RDFS.range, XSD.string))

    # desc
    g.add((YGO.cardDesc, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.cardDesc, RDFS.domain, YGO.Card))
    g.add((YGO.cardDesc, RDFS.range, XSD.string))

    # race
    g.add((YGO.race, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.race, RDFS.domain, YGO.Card))
    g.add((YGO.race, RDFS.range, XSD.string))

    # archetype
    g.add((YGO.archetype, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.archetype, RDFS.domain, YGO.Card))
    g.add((YGO.archetype, RDFS.range, XSD.string))

    # ygoprodeck_url
    g.add((YGO.ygoprodeckURL, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.ygoprodeckURL, RDFS.domain, YGO.Card))
    g.add((YGO.ygoprodeckURL, RDFS.range, XSD.anyURI))

    # is_staple
    g.add((YGO.isStaple, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.isStaple, RDFS.domain, YGO.Card))
    g.add((YGO.isStaple, RDFS.range, XSD.boolean))

    # atk
    g.add((YGO.atk, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.atk, RDFS.domain, YGO.Card))
    g.add((YGO.atk, RDFS.range, XSD.integer))

    # defense
    g.add((YGO.defense, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.defense, RDFS.domain, YGO.Card))
    g.add((YGO.defense, RDFS.range, XSD.integer))

    # level
    g.add((YGO.level, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.level, RDFS.domain, YGO.Card))
    g.add((YGO.level, RDFS.range, XSD.integer))

    # attribute
    g.add((YGO.cardAttribute, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.cardAttribute, RDFS.domain, YGO.Card))
    g.add((YGO.cardAttribute, RDFS.range, XSD.string))

    # ban_tcg
    g.add((YGO.banTcg, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.banTcg, RDFS.domain, YGO.Card))
    g.add((YGO.banTcg, RDFS.range, XSD.string))

    # ban_ocg
    g.add((YGO.banOcg, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.banOcg, RDFS.domain, YGO.Card))
    g.add((YGO.banOcg, RDFS.range, XSD.string))

    # ban_goat
    g.add((YGO.banGoat, RDF.type, OWL.DatatypeProperty))
    g.add((YGO.banGoat, RDFS.domain, YGO.Card))
    g.add((YGO.banGoat, RDFS.range, XSD.string))





    # Effekt-Kategorien (Boolean) -> je ein DatatypeProperty
    effect_fields = [
        "search", "destroy", "negate", "draw",
        "special_summon", "banish", "send_gy", "recover_lp",
        "inflict_damage", "equip", "modify_stats", "protect",
        "discard", "change_position", "return", "shuffle",
        "copy", "counter", "token_summon", "deck_manipulation"
    ]
    for ef in effect_fields:
        # Erzeugt z.B. ygo:effect_search, ygo:effect_destroy
        prop_uri = getattr(YGO, f"effect_{ef}")
        g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
        g.add((prop_uri, RDFS.domain, YGO.Card))
        g.add((prop_uri, RDFS.range, XSD.boolean))

    # Jetzt alles in TTL serialisieren (nur T-Box)
    g.serialize(destination=output_file, format="turtle")
    print(f"T-Box für Cards wurde in '{output_file}' gespeichert.")


if __name__ == "__main__":
    build_cards_tbox()
