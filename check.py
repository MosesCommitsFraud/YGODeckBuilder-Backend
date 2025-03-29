# update_banlist.py

import time
import requests
from urllib.parse import quote

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Modelle importieren
from db.models import Base, Card

# Datenbank-Verbindung aufbauen (anpassen, falls dein Pfad anders ist)
engine = create_engine('sqlite:///data.sqlite')
Session = sessionmaker(bind=engine)
session = Session()

def update_banlist_info():
    # Alle Karten laden
    cards = session.query(Card).all()
    print(f"Starte Update der Banlist-Info für {len(cards)} Karten...")

    for idx, card in enumerate(cards, start=1):
        # Name der Karte (mit URL-encoding) in die API-URL einsetzen
        # (Falls du lieber mit "id" arbeitest, könntest du ?id=... benutzen)
        url = f"https://db.ygoprodeck.com/api/v7/cardinfo.php?name={quote(card.name)}"

        try:
            response = requests.get(url, timeout=10)
            # Bei fehlerhaftem Statuscode sofort weitermachen
            if response.status_code != 200:
                print(f"[{idx}/{len(cards)}] Karte '{card.name}' nicht gefunden oder Fehler.")
                continue

            data = response.json()

            # Falls wir valide Daten zurückbekommen haben
            if "data" in data and len(data["data"]) > 0:
                card_info = data["data"][0]

                # banlist_info kann fehlen, wenn die Karte keinen Banstatus hat
                banlist_info = card_info.get("banlist_info", {})

                # Setze den Banstatus (oder None, falls nicht vorhanden)
                card.ban_tcg = banlist_info.get("ban_tcg")   # z.B. "Banned", "Limited", ...
                card.ban_ocg = banlist_info.get("ban_ocg")
                card.ban_goat = banlist_info.get("ban_goat")

                session.add(card)
                print(f"[{idx}/{len(cards)}] Aktualisiere Banlist: {card.name} (TCG={card.ban_tcg}, OCG={card.ban_ocg}, GOAT={card.ban_goat})")
            else:
                print(f"[{idx}/{len(cards)}] Keine Daten für '{card.name}' gefunden.")

        except Exception as e:
            print(f"[{idx}/{len(cards)}] Fehler bei '{card.name}': {e}")

        # Um nicht zu schnell zu viele Requests zu senden, kurz warten (Rate Limit!).
        # Die YGOPRODeck-API erlaubt 20 Requests/Sekunde, aber ein kleiner Sleep schadet nicht.
        time.sleep(0.1)

    # Am Ende Commit ausführen, um alle Änderungen zu speichern
    session.commit()
    session.close()
    print("Banlist-Update abgeschlossen.")

if __name__ == "__main__":
    update_banlist_info()

