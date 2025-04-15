import os
import glob
import re
from db.models import Base, Deck, DeckCard
from db.db import get_session, engine

# Erstelle die Tabellen (falls noch nicht vorhanden)
Base.metadata.create_all(engine)


def parse_ydk(file_path):
    """Parst eine .ydk-Datei und gibt ein Dictionary mit den Sektionen zurück."""
    deck = {"main": [], "extra": [], "side": []}
    current_section = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Erkenne Sektionen
            if line.startswith("#"):
                if line.lower() == "#main":
                    current_section = "main"
                elif line.lower() == "#extra":
                    current_section = "extra"
            elif line.startswith("!"):
                current_section = "side"
            else:
                if current_section:
                    deck[current_section].append(line)
    return deck


# Pfad zum Ordner, der alle .ydk-Dateien enthält
deck_directory = "ydk_download"  # <-- Passe diesen Pfad an
# Alle .ydk-Dateien im Ordner finden
ydk_files = glob.glob(os.path.join(deck_directory, "*.ydk"))

with get_session() as session:
    for file_path in ydk_files:
        deck_data = parse_ydk(file_path)
        # Verwende den Dateinamen (ohne Endung) als Deck-Name
        deck_name = os.path.splitext(os.path.basename(file_path))[0]
        deck = Deck(name=deck_name)
        session.add(deck)
        session.commit()  # Damit deck.id verfügbar ist

        # Füge Karten des Main-Decks hinzu
        for card_id in deck_data["main"]:
            deck_card = DeckCard(deck_id=deck.id, card_id=int(card_id), section="main")
            session.add(deck_card)

        # Füge Karten des Extra-Decks hinzu
        for card_id in deck_data["extra"]:
            deck_card = DeckCard(deck_id=deck.id, card_id=int(card_id), section="extra")
            session.add(deck_card)

        # Füge Karten des Side-Decks hinzu
        for card_id in deck_data["side"]:
            deck_card = DeckCard(deck_id=deck.id, card_id=int(card_id), section="side")
            session.add(deck_card)

        session.commit()

print("Alle Decks wurden erfolgreich in die Datenbank eingefügt.")
