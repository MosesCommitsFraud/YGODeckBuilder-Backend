import os
import json
import pandas as pd
import requests
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# 1. Parsing .ydk Files
def parse_ydk_file(file_path):
    """
    Parses a .ydk deck file and extracts the main, extra, and side deck card IDs.
    """
    main_deck = []
    extra_deck = []
    side_deck = []
    current_section = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Detect section markers
            if line.startswith("#"):
                if line.lower().startswith("#main"):
                    current_section = "main"
                elif line.lower().startswith("#extra"):
                    current_section = "extra"
                else:
                    current_section = None
            elif line.startswith("!"):
                if line.lower().startswith("!side"):
                    current_section = "side"
                else:
                    current_section = None
            else:
                # Assume line is a card ID
                if current_section == "main":
                    main_deck.append(line)
                elif current_section == "extra":
                    extra_deck.append(line)
                elif current_section == "side":
                    side_deck.append(line)
    return {"main": main_deck, "extra": extra_deck, "side": side_deck}


def load_decks(directory):
    """
    Loads all .ydk files from a directory and returns a list of main decks.
    Each main deck is represented as a list of card IDs (strings).
    """
    decks = []
    for filename in os.listdir(directory):
        if filename.endswith(".ydk"):
            file_path = os.path.join(directory, filename)
            deck = parse_ydk_file(file_path)
            # Only use decks with a valid main deck (Yu-Gi-Oh! main decks are 40-60 cards)
            if 40 <= len(deck["main"]) <= 60:
                decks.append(deck["main"])
    return decks


# 2. Build Transaction DataFrame for Association Rule Mining
def build_transaction_df(decks):
    """
    Converts a list of decks (transactions) into a one-hot encoded DataFrame.
    """
    te = TransactionEncoder()
    te_ary = te.fit(decks).transform(decks)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df


# 3. Mine Frequent Itemsets and Association Rules with adjusted parameters
def mine_rules(df, min_support=0.05, min_confidence=0.5, max_len=3):
    """
    Uses the Apriori algorithm to mine frequent itemsets and then generates association rules.
    Adjusted parameters: higher min_support and max_len to reduce memory usage.
    """
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=max_len)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules


# 4. Recommend Cards Based on Input Cards (Updated Matching Logic)
def recommend_cards(input_cards, rules, top_n=5):
    """
    Given a list of input card IDs (as strings) and association rules,
    recommends additional cards. This version relaxes matching:
    if any input card appears in the rule antecedents, the rest of the consequents are recommended.
    Returns a list of tuples: (card_id, average_confidence).
    """
    recommendations = {}
    input_set = set(input_cards)
    matched_rule_count = 0

    for idx, rule in rules.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])
        # Check if there's any overlap between input_set and antecedents
        if input_set.intersection(antecedents):
            matched_rule_count += 1
            new_cards = consequents - input_set  # avoid recommending cards already in the input
            for card in new_cards:
                recommendations.setdefault(card, []).append(rule['confidence'])

    print(f"Found {matched_rule_count} rules that match the input cards.")

    # Rank recommendations by average confidence.
    ranked = []
    for card, confidences in recommendations.items():
        avg_conf = sum(confidences) / len(confidences)
        ranked.append((card, avg_conf))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


# 5. Fallback Recommendation: Co-occurrence Frequency
def fallback_recommendations(input_cards, decks, top_n=5):
    """
    As a fallback, count the frequency of cards that co-occur with the input cards in decks.
    """
    input_set = set(input_cards)
    co_occur = {}
    for deck in decks:
        # If the deck contains the input cards (or at least one of them)
        if input_set.intersection(set(deck)):
            for card in deck:
                if card not in input_set:
                    co_occur[card] = co_occur.get(card, 0) + 1
    # Sort by frequency (highest first)
    ranked = sorted(co_occur.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


# 6. Fetch & Cache Card Details from YGOPRODeck API
def get_card_details(card_id, cache_file="card_cache.json"):
    """
    Retrieves card details from the YGOPRODeck API.
    Caches results locally to reduce API calls.
    """
    # Load cache if available
    try:
        with open(cache_file, "r") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    # Check if card details are already cached
    if str(card_id) in cache:
        return cache[str(card_id)]

    # Fetch card details from the API
    url = f"https://db.ygoprodeck.com/api/v7/cardinfo.php?id={card_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            card_info = data["data"][0]
            cache[str(card_id)] = card_info
            # Update the cache file
            with open(cache_file, "w") as f:
                json.dump(cache, f)
            return card_info
    return None


# 7. Example Main Function Tying Everything Together
def main():
    # --- SETTINGS ---
    decks_directory = "ydk_download"  # Update this path to your .ydk files directory.
    min_support = 0.05  # Adjust as needed: lowering this may generate more rules.
    min_confidence = 0.5  # Confidence threshold.
    max_itemset_length = 3  # Limit the size of itemsets (e.g., only pairs or triplets).
    top_n_recommendations = 5

    # --- Load and Process Deck Data ---
    print("Loading decks...")
    decks = load_decks(decks_directory)
    print(f"Loaded {len(decks)} decks.")

    print("Building transaction DataFrame...")
    df = build_transaction_df(decks)

    # --- Mine Association Rules ---
    print("Mining association rules...")
    rules = mine_rules(df, min_support=min_support, min_confidence=min_confidence, max_len=max_itemset_length)
    print(f"Mined {len(rules)} association rules.")

    # --- Get Recommendations ---
    # For example, assume the user has chosen a starting card with ID "89631139"
    input_cards = ["21767650"]
    recs = recommend_cards(input_cards, rules, top_n=top_n_recommendations)

    if not recs:
        print("No recommendations found using association rules. Falling back to co-occurrence counts.")
        recs = fallback_recommendations(input_cards, decks, top_n=top_n_recommendations)
        # The fallback returns (card_id, frequency), so we handle it slightly differently.
        for card, freq in recs:
            details = get_card_details(card)
            card_name = details["name"] if details and "name" in details else "Unknown Card"
            print(f"Fallback - Card ID: {card} - {card_name} | Co-occurrence Frequency: {freq}")
    else:
        print("Recommended cards based on your input:")
        for card_id, avg_conf in recs:
            details = get_card_details(card_id)
            card_name = details["name"] if details and "name" in details else "Unknown Card"
            print(f"Card ID: {card_id} - {card_name} | Avg Confidence: {avg_conf:.2f}")


if __name__ == "__main__":
    main()
