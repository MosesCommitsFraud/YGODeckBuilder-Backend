import os
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

###############################
# 1. GRAPH CONSTRUCTION
###############################

# --- Step 1A: Load Cards and Build Lookups ---
with open("cards.json", "r") as f:
    cards_data = json.load(f)["data"]

# Lookup: lower-case card name -> card ID (as string)
card_name_to_id = {card["name"].lower(): str(card["id"]) for card in cards_data}
# Mapping from card ID to full card info
card_info = {str(card["id"]): card for card in cards_data}

# --- Step 1B: Initialize BERT NER Pipeline (using provided model) ---
ner_pipe = pipeline("token-classification", model="dslim/bert-base-NER", grouped_entities=True)
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model_ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")


def get_mentioned_card_ids(card_desc, card_name_to_id):
    """
    Uses the BERT NER pipeline to extract entities from a card description.
    Returns a set of card IDs if an entity exactly matches a known card name.
    """
    entities = ner_pipe(card_desc)
    mentioned_ids = set()
    for entity in entities:
        entity_text = entity["word"].strip().lower()
        if entity_text in card_name_to_id:
            mentioned_ids.add(card_name_to_id[entity_text])
    return mentioned_ids


# --- Step 1C: Parse .ydk Deck Files & Build Co-occurrence Dictionary ---
def parse_ydk(file_path):
    """
    Parses a .ydk file and returns a dict with keys 'main', 'extra', 'side'
    containing lists of card IDs.
    """
    sections = {"main": [], "extra": [], "side": []}
    current_section = None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                if "main" in line.lower():
                    current_section = "main"
                elif "extra" in line.lower():
                    current_section = "extra"
            elif line.startswith("!"):
                current_section = "side"
            elif line and current_section is not None:
                sections[current_section].append(line)
    return sections


def build_co_occurrence(decks_folder):
    """
    Iterates over all .ydk files in decks_folder and builds a dictionary that maps
    (card_id_i, card_id_j) to the number of decks in which they co-occur.
    """
    co_occurrence = defaultdict(int)
    deck_files = [f for f in os.listdir(decks_folder) if f.endswith(".ydk")]
    num_decks = len(deck_files)

    for deck_file in deck_files:
        deck_path = os.path.join(decks_folder, deck_file)
        deck = parse_ydk(deck_path)
        # Combine all card IDs from main, extra, and side
        cards_in_deck = set(deck.get("main", []) + deck.get("extra", []) + deck.get("side", []))
        for card_i in cards_in_deck:
            for card_j in cards_in_deck:
                if card_i != card_j:
                    co_occurrence[(card_i, card_j)] += 1
    return co_occurrence, num_decks


# Change this folder path to where your .ydk files are stored.
decks_folder = "ydk_download"
co_occurrence, total_decks = build_co_occurrence(decks_folder)

# --- Step 1D: Build the Weighted Graph Edges ---
# Weight multipliers:
alpha = 1.0  # Highest weight: BERT mention
beta = 0.5   # Moderate weight: normalized co-occurrence frequency
gamma = 0.1  # Very small bonus: shared attribute

graph_edges = {}
for card_id_i, info_i in card_info.items():
    desc_i = info_i.get("desc", "")
    mentioned_ids = get_mentioned_card_ids(desc_i, card_name_to_id) if desc_i else set()

    for card_id_j, info_j in card_info.items():
        if card_id_i == card_id_j:
            continue
        weight = 0.0
        # Factor 1: BERT NER mention
        if card_id_j in mentioned_ids:
            weight += alpha
        # Factor 2: Normalized co-occurrence frequency
        freq = co_occurrence.get((card_id_i, card_id_j), 0) / total_decks
        weight += beta * freq
        # Factor 3: Shared attribute bonus
        if info_i.get("attribute") and info_j.get("attribute"):
            if info_i["attribute"] == info_j["attribute"]:
                weight += gamma
        if weight > 0:
            graph_edges[(card_id_i, card_id_j)] = weight

print(f"Constructed graph with {len(graph_edges)} weighted edges.")

###############################
# 2. CREATE PYTORCH GEOMETRIC DATA OBJECT
###############################

# --- Step 2A: Build Node Features ---
# For demonstration, we create a simple feature vector:
#   - One-hot encoding for attribute (plus a "None" option)
#   - Normalized level, atk, and defense (if available)
attributes = list({card.get("attribute", "None") for card in cards_data})
attribute_to_idx = {attr: i for i, attr in enumerate(attributes)}

node_features = []
node_id_to_card_id = {}
card_id_to_node_id = {}
for i, (card_id, card) in enumerate(card_info.items()):
    one_hot = [0] * len(attributes)
    attr = card.get("attribute", "None")
    one_hot[attribute_to_idx[attr]] = 1
    level = (card.get("level") or 0) / 12.0  # normalized (assuming max level ~12)
    atk = (card.get("atk") or 0) / 5000.0
    defense = (card.get("def") or 0) / 5000.0
    feature = one_hot + [level, atk, defense]
    node_features.append(feature)
    node_id_to_card_id[i] = card_id
    card_id_to_node_id[card_id] = i

x = torch.tensor(node_features, dtype=torch.float)

# --- Step 2B: Build Edge Index ---
edge_index_list = []
for (card_id_i, card_id_j), weight in graph_edges.items():
    if card_id_i in card_id_to_node_id and card_id_j in card_id_to_node_id:
        i = card_id_to_node_id[card_id_i]
        j = card_id_to_node_id[card_id_j]
        # Here we ignore the weight (or you can store it in edge_attr)
        edge_index_list.append([i, j])

if edge_index_list:
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
else:
    edge_index = torch.empty((2, 0), dtype=torch.long)

# Create the PyTorch Geometric Data object.
data = Data(x=x, edge_index=edge_index)

###############################
# 3. BUILD THE GNN MODEL & TRAINING TARGETS
###############################

# --- Step 3A: Define a Simple GNN Model ---
class DeckGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(DeckGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # Output a single logit per node (we’ll use sigmoid for probability)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.out(x)
        return torch.sigmoid(x).squeeze()  # probability for each card

# --- Step 3B: Build Training Targets from Deck Files ---
# Each deck file gives a set of card IDs (across main, extra, side).
# We create a binary label vector (length = number of nodes) where 1 means inclusion.
training_decks = []
deck_files = [f for f in os.listdir(decks_folder) if f.endswith(".ydk")]

for deck_file in deck_files:
    deck_path = os.path.join(decks_folder, deck_file)
    deck = parse_ydk(deck_path)
    selected_cards = set(deck.get("main", []) + deck.get("extra", []) + deck.get("side", []))
    target = torch.zeros(x.shape[0], dtype=torch.float)
    for card_id in selected_cards:
        if card_id in card_id_to_node_id:
            target[card_id_to_node_id[card_id]] = 1.0
    training_decks.append(target)

###############################
# 4. TRAINING & DECK GENERATION FUNCTIONS
###############################

MODEL_SAVE_PATH = "deck_gnn.pt"

def train_model(num_epochs=50, model_save_path=MODEL_SAVE_PATH):
    """
    Trains the DeckGNN model on the global graph using the training decks,
    saves the model, and returns the trained model.
    """
    model_gnn = DeckGNN(in_channels=x.shape[1], hidden_channels=64)
    optimizer = optim.Adam(model_gnn.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        model_gnn.train()
        optimizer.zero_grad()
        out = model_gnn(data.x, data.edge_index)  # shape: (num_nodes,)
        loss_total = 0
        for target in training_decks:
            loss_total += criterion(out, target)
        loss_total = loss_total / len(training_decks)
        loss_total.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss_total.item():.4f}")

    torch.save(model_gnn.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return model_gnn


def load_model(model_save_path=MODEL_SAVE_PATH):
    """
    Loads the DeckGNN model from the specified path.
    """
    model = DeckGNN(in_channels=x.shape[1], hidden_channels=64)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    return model


def generate_deck(model, data, seed_card_ids=None, threshold=0.5, synergy_factor=1.0):
    """
    Uses the trained GNN model to generate a deck, conditioned on optional seed card IDs.
    Applies the following steps:
      - Runs the model to get base probabilities.
      - If seed_card_ids are provided, adds a bonus to candidates connected
        to the seed cards in the synergy graph.
      - Ensures the seed cards are included in the final deck.
      - Partitions selected cards into Main and Extra decks.
      - Adjusts the Main deck size to be between 40 and 60 cards.
    """
    model.eval()
    with torch.no_grad():
        probs = model(data.x, data.edge_index)

    # If seed cards are provided, adjust probabilities with a synergy bonus.
    if seed_card_ids is not None:
        # Convert provided card IDs to node indices.
        seed_node_ids = [card_id_to_node_id[cid] for cid in seed_card_ids if cid in card_id_to_node_id]
        bonus = torch.zeros_like(probs)
        for seed_node in seed_node_ids:
            seed_card_id = node_id_to_card_id[seed_node]
            # For each candidate, if there's an edge from the seed card to the candidate, add bonus.
            for candidate_node in range(len(probs)):
                candidate_card_id = node_id_to_card_id[candidate_node]
                edge_weight = graph_edges.get((seed_card_id, candidate_card_id), 0.0)
                bonus[candidate_node] += synergy_factor * edge_weight
        probs = probs + bonus

    # Sort node indices by descending adjusted probability.
    node_indices = torch.argsort(probs, descending=True)

    selected_nodes = set()
    # Always include the seed nodes (if provided).
    if seed_card_ids is not None:
        for cid in seed_card_ids:
            if cid in card_id_to_node_id:
                selected_nodes.add(card_id_to_node_id[cid])
    # Then add other nodes with probability above the threshold.
    for idx in node_indices:
        if idx.item() in selected_nodes:
            continue
        if probs[idx].item() < threshold:
            break
        selected_nodes.add(idx.item())

    # Convert node indices back to card IDs.
    selected_card_ids = [node_id_to_card_id[idx] for idx in selected_nodes]

    # Partition cards into Main and Extra decks.
    extra_keywords = {"XYZ", "Synchro", "Fusion", "Link"}
    main_deck = []
    extra_deck = []
    for card_id in selected_card_ids:
        card = card_info[card_id]
        card_type = card.get("type", "")
        if "Pendulum" in card_type or any(keyword in card_type for keyword in extra_keywords):
            extra_deck.append(card_id)
        else:
            main_deck.append(card_id)

    # Enforce Main deck size between 40 and 60 cards (basic adjustment).
    if len(main_deck) < 40:
        for idx in node_indices:
            card_id = node_id_to_card_id[idx.item()]
            if card_id not in main_deck and card_id not in extra_deck:
                main_deck.append(card_id)
            if len(main_deck) >= 40:
                break
    elif len(main_deck) > 60:
        main_deck = main_deck[:60]

    # Side deck generation is omitted for brevity.
    side_deck = []

    return main_deck, extra_deck, side_deck


def create_deck(seed_card_ids=None, threshold=0.5, synergy_factor=1.0):
    """
    Loads (or trains) the DeckGNN model and generates a deck based on optional seed card IDs.
    Returns the Main, Extra, and Side decks.
    """
    if os.path.exists(MODEL_SAVE_PATH):
        model = load_model(MODEL_SAVE_PATH)
    else:
        print("No saved model found. Training model now...")
        model = train_model(num_epochs=50, model_save_path=MODEL_SAVE_PATH)

    return generate_deck(model, data, seed_card_ids=seed_card_ids, threshold=threshold, synergy_factor=synergy_factor)


###############################
# 5. USAGE EXAMPLE
###############################

if __name__ == "__main__":
    # Prompt the user for seed card IDs.
    user_input = input("Enter one or more seed card IDs, separated by commas (or leave blank for none): ").strip()
    if user_input:
        seed_ids = [s.strip() for s in user_input.split(",")]
    else:
        seed_ids = None

    main_deck, extra_deck, side_deck = create_deck(seed_card_ids=seed_ids)
    print("\n--- Generated Deck ---")
    print("Main Deck ({} cards):".format(len(main_deck)), main_deck)
    print("Extra Deck ({} cards):".format(len(extra_deck)), extra_deck)
    print("Side Deck ({} cards):".format(len(side_deck)), side_deck)
