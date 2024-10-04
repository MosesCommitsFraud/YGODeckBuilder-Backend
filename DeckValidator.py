# Use YGOProDeck API to fetch the forbidden, limited, and semi-limited cards
from collections import Counter

import requests


def fetch_banlist():
    url = "https://db.ygoprodeck.com/api/v7/cardinfo.php?banlist=tcg"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to retrieve banlist.")
        return None


# Validate deck (apply Yu-Gi-Oh rules)
def validate_deck(deck, side_deck, extra_deck):
    # Load banlist data
    banlist_data = fetch_banlist()
    forbidden_cards = []
    limited_cards = []
    semi_limited_cards = []

    # Parse banlist for forbidden, limited, and semi-limited cards
    if banlist_data:
        for card in banlist_data['data']:
            ban_status = card.get('banlist_info', {})
            if ban_status.get('ban_tcg') == "Forbidden":
                forbidden_cards.append(card['name'])
            elif ban_status.get('ban_tcg') == "Limited":
                limited_cards.append(card['name'])
            elif ban_status.get('ban_tcg') == "Semi-Limited":
                semi_limited_cards.append(card['name'])

    # Check main deck size
    if 40 <= len(deck) <= 60:
        print("Main deck size is valid.")
    else:
        print("Main deck size is invalid.")

    # Check extra deck size
    if len(extra_deck) <= 15:
        print("Extra deck size is valid.")
    else:
        print("Extra deck size is invalid.")

    # Check side deck size
    if len(side_deck) <= 15:
        print("Side deck size is valid.")
    else:
        print("Side deck size is invalid.")

    # Check for forbidden cards
    all_cards = deck + side_deck + extra_deck
    forbidden_in_deck = [card for card in all_cards if card in forbidden_cards]
    if forbidden_in_deck:
        print(f"Deck contains forbidden cards: {forbidden_in_deck}")
    else:
        print("Deck contains no forbidden cards.")

    # Check for limited cards
    card_counts = Counter(all_cards)
    for card, count in card_counts.items():
        if card in limited_cards and count > 1:
            print(f"Deck contains more than 1 copy of a limited card: {card}")

    # Check for semi-limited cards
    for card, count in card_counts.items():
        if card in semi_limited_cards and count > 2:
            print(f"Deck contains more than 2 copies of a semi-limited card: {card}")

    # Check for 3-copy limit (across main, extra, and side deck)
    for card, count in card_counts.items():
        if count > 3:
            print(f"Deck contains more than 3 copies of a card: {card}")

    print("Deck validation complete.")