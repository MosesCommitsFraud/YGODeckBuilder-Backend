import pandas as pd
import requests
import json


def process_decks_and_fetch_card_data(file_paths):
    # Step 1: Load and clean CSV data
    dataframes = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Clean the DataFrame by selecting only the columns needed: 'Deck-Selector' and 'Card-Name'
    cleaned_df = combined_df[['Deck-Selector', 'Card-Name']].dropna()

    # Group the cards by 'Deck-Selector'
    grouped_decks = cleaned_df.groupby('Deck-Selector')['Card-Name'].apply(list).reset_index()

    # Save the cleaned deck structure to a CSV file
    grouped_decks.to_csv('cleaned_decks.csv', index=False)
    print("Deck structure saved to cleaned_decks.csv")

    # Step 2: Fetch card data from the YGOProDeck API
    api_url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
    response = requests.get(api_url)

    if response.status_code == 200:
        card_data = response.json()

        # Save the card data in JSON format
        with open('card_data.json', 'w') as json_file:
            json.dump(card_data, json_file)

        print("Card data saved to card_data.json")
    else:
        print(f"Failed to retrieve card data. Status code: {response.status_code}")

    return grouped_decks, response.status_code
