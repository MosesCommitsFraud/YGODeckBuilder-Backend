import json
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Card


def update_atk_def_level_attribute(
        json_file="cards.json",
        db_url="sqlite:///data.sqlite"
):
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    if not os.path.exists(json_file):
        print(f"JSON-Datei {json_file} nicht gefunden!")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Die JSON hat wahrscheinlich data["data"] als Liste von Karten
    card_list = data.get("data", [])
    updated_count = 0

    for card_info in card_list:
        card_id = card_info.get("id")
        if card_id is None:
            continue

        # Karte in DB suchen
        card_obj = session.query(Card).filter(Card.id == card_id).one_or_none()
        if not card_obj:
            # Wenn die Karte nicht existiert, überspringen oder du könntest
            # sie neu anlegen – je nach Bedarf.
            continue

        # Prüfen, ob es ein Monster ist (z.B. "type" enthält das Wort "Monster")
        card_type = card_info.get("type", "")
        if "Monster" in card_type:
            # Aus JSON auslesen (falls nicht vorhanden → None)
            card_obj.atk = card_info.get("atk")
            card_obj.defense = card_info.get("def")
            card_obj.level = card_info.get("level")
            card_obj.attribute = card_info.get("attribute")
        else:
            # Für Zauber/Fallen: setze auf None
            card_obj.atk = None
            card_obj.defense = None
            card_obj.level = None
            card_obj.attribute = None

        updated_count += 1

    session.commit()
    session.close()
    print(f"Aktualisiert: {updated_count} Karten.")


if __name__ == "__main__":
    update_atk_def_level_attribute(
        json_file="cards.json",
        db_url="sqlite:///data.sqlite"
    )
