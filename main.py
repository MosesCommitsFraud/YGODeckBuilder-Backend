from ScrapeCleaner import process_decks_and_fetch_card_data
from DeckAnalyzer import analyze_cards


if __name__ == "__main__":

 file_paths = [
    'YGOPro (1).csv',
    'YGOPro (2).csv',
    'YGOPro (3).csv',
    'YGOPro (4).csv',
    'YGOPro (5).csv'
 ]

# Step 2: Call the single function to process decks and fetch card data
grouped_decks, status_code = process_decks_and_fetch_card_data(file_paths)

# Optional: Print the first few rows of the grouped decks
print(grouped_decks.head())

# Path to the card data JSON file (from the API)
card_data_file = 'card_data.json'

# Call the function to analyze the cards and save the output
analyze_cards(card_data_file)