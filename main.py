import os
import glob
import json
import requests
from gensim.models import Word2Vec

def get_card_data(cache_file="cards_cache.json"):
    """
    Fetch card data from the API and cache it locally.
    Returns a dictionary mapping card ID (as a string) to its details.
    """
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            card_data = json.load(f)
        print("Loaded card data from cache.")
    else:
        print("Fetching card data from API...")
        response = requests.get("https://db.ygoprodeck.com/api/v7/cardinfo.php")
        if response.status_code == 200:
            # The API returns a JSON with a key "data" that contains a list of card details.
            data = response.json()["data"]
            # Convert each card's ID to a string for consistency with the .ydk files.
            card_data = {str(card["id"]): card for card in data}
            with open(cache_file, "w") as f:
                json.dump(card_data, f)
            print("Card data cached locally.")
        else:
            raise Exception("Failed to fetch card data from API")
    return card_data

def parse_ydk_file(filepath):
    """
    Parse a .ydk file and return a dictionary with three keys:
      - "main": list of card IDs in the main deck.
      - "extra": list of card IDs in the extra deck.
      - "side": list of card IDs in the side deck.
    """
    main_deck = []
    extra_deck = []
    side_deck = []
    current_section = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Identify section headers
            if line.startswith("#"):
                if line.lower().startswith("#main"):
                    current_section = "main"
                elif line.lower().startswith("#extra"):
                    current_section = "extra"
                else:
                    current_section = None
                continue
            if line.startswith("!"):
                if line.lower().startswith("!side"):
                    current_section = "side"
                else:
                    current_section = None
                continue
            # Append card ID to the corresponding section list.
            if current_section == "main":
                main_deck.append(line)
            elif current_section == "extra":
                extra_deck.append(line)
            elif current_section == "side":
                side_deck.append(line)
    return {"main": main_deck, "extra": extra_deck, "side": side_deck}

def load_decks_from_folder(folder):
    """
    Load and parse all .ydk files from the specified folder.
    Only decks with a main deck size between 40 and 60 cards are retained.
    Returns a list of decks (each as a dictionary).
    """
    deck_files = glob.glob(os.path.join(folder, "*.ydk"))
    decks = []
    for file in deck_files:
        deck = parse_ydk_file(file)
        if 40 <= len(deck["main"]) <= 60:
            decks.append(deck)
    return decks

def train_word2vec_on_decks(decks, vector_size=100, window=5, min_count=1, workers=4, epochs=10):
    """
    Train a Word2Vec model using the main decks as the corpus.
    Each deck's main deck (list of card IDs) is considered a sentence.
    """
    # Create a corpus: list of sentences (here, main deck card ID lists)
    corpus = [deck["main"] for deck in decks]
    model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    model.build_vocab(corpus)
    print(f"Vocabulary size: {len(model.wv.index_to_key)}")
    model.train(corpus, total_examples=model.corpus_count, epochs=epochs)
    return model

if __name__ == '__main__':
    # Step 1: Cache and load card data from the API.
    card_data = get_card_data()

    # Create a mapping for convenience: card_id -> card_name.
    card_names = {card_id: details["name"] for card_id, details in card_data.items()}

    # Step 2: Load decks from the folder "ydk_download".
    decks = load_decks_from_folder("ydk_download")
    print(f"Loaded {len(decks)} decks.")

    # Step 3: Train the Word2Vec model on the main decks.
    model = train_word2vec_on_decks(decks)

    # Step 4: Example usage.
    # Find cards similar to a given card (by its ID).
    query_card = '89631139'
    if query_card in model.wv:
        similar_cards = model.wv.most_similar(query_card)
        print(f"\nCards similar to {query_card} ({card_names.get(query_card, 'Unknown')}):")
        for sim_card, similarity in similar_cards:
            print(f"Card: {sim_card} ({card_names.get(sim_card, 'Unknown')}), Similarity: {similarity:.4f}")
    else:
        print(f"\nCard {query_card} not found in the vocabulary.")
