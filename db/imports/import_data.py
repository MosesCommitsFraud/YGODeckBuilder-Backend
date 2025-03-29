# import_data.py
import json
from db.models import Base, Card, CardSet, CardImage, CardPrice
from db.db import get_session, engine  # engine und get_session aus deinem DB-Modul

# Erstelle alle Tabellen, falls sie noch nicht existieren
Base.metadata.create_all(engine)

# Lese die JSON-Datei ein (z. B. "card_data.json") mit errors='replace', um ungültige Zeichen zu ersetzen
with open('../../cards.json', 'r', encoding='utf-8', errors='replace') as f:
    data = json.load(f)

# Benutze den Context Manager, um eine Session zu erhalten
with get_session() as session:
    # Gehe davon aus, dass die JSON-Daten in einem Objekt mit dem Schlüssel "data" als Liste vorliegen
    for card_data in data.get("data", []):
        # Erstelle ein Card-Objekt
        card = Card(
            id=card_data.get("id"),
            name=card_data.get("name"),
            type=card_data.get("type"),
            human_readable_card_type=card_data.get("humanReadableCardType"),
            frame_type=card_data.get("frameType"),
            desc=card_data.get("desc"),
            race=card_data.get("race"),
            archetype=card_data.get("archetype"),
            ygoprodeck_url=card_data.get("ygoprodeck_url")
        )

        # Füge die Kartensets hinzu
        for set_data in card_data.get("card_sets", []):
            card_set = CardSet(
                set_name=set_data.get("set_name"),
                set_code=set_data.get("set_code"),
                set_rarity=set_data.get("set_rarity"),
                set_rarity_code=set_data.get("set_rarity_code"),
                set_price=set_data.get("set_price")
            )
            card.sets.append(card_set)

        # Füge die Kartenbilder hinzu
        for image_data in card_data.get("card_images", []):
            card_image = CardImage(
                image_url=image_data.get("image_url"),
                image_url_small=image_data.get("image_url_small"),
                image_url_cropped=image_data.get("image_url_cropped")
            )
            card.images.append(card_image)

        # Füge die Kartenpreise hinzu
        for price_data in card_data.get("card_prices", []):
            card_price = CardPrice(
                cardmarket_price=price_data.get("cardmarket_price"),
                tcgplayer_price=price_data.get("tcgplayer_price"),
                ebay_price=price_data.get("ebay_price"),
                amazon_price=price_data.get("amazon_price"),
                coolstuffinc_price=price_data.get("coolstuffinc_price")
            )
            card.prices.append(card_price)

        session.add(card)

    session.commit()

print("Daten erfolgreich in die DB eingefügt.")
