import os
import json
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import requests  # For fetching the banlist from the API
from tqdm import tqdm  # For progress bars

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# -------------------------
# GRAPH CACHING FUNCTIONS
# -------------------------

CACHE_FILE = "graph_cache.pt"

def build_graph():
    """Builds the graph from scratch using cards.json and deck files."""
    # --- Load Cards and Build Lookups ---
    with open("cards.json", "r") as f:
        cards_data = json.load(f)["data"]

    card_name_to_id = {card["name"].lower(): str(card["id"]) for card in cards_data}
    card_info = {str(card["id"]): card for card in cards_data}

    # --- Initialize BERT NER Pipeline ---
    ner_pipe = pipeline("token-classification", model="dslim/bert-base-NER", grouped_entities=True)
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model_ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    def get_mentioned_card_ids(card_desc, card_name_to_id):
        entities = ner_pipe(card_desc)
        mentioned_ids = set()
        for entity in entities:
            entity_text = entity["word"].strip().lower()
            if entity_text in card_name_to_id:
                mentioned_ids.add(card_name_to_id[entity_text])
        return mentioned_ids

    # -------------------------
    # Build Co-occurrence & Graph
    # -------------------------
    decks_folder = "ydk_download"

    def parse_ydk(file_path):
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
        co_occurrence = defaultdict(int)
        deck_files = [f for f in os.listdir(decks_folder) if f.endswith(".ydk")]
        num_decks = len(deck_files)
        for deck_file in deck_files:
            deck_path = os.path.join(decks_folder, deck_file)
            deck = parse_ydk(deck_path)
            # Combine all card IDs from main, extra, and side (keeping duplicates)
            cards_in_deck = deck.get("main", []) + deck.get("extra", []) + deck.get("side", [])
            unique_cards = set(cards_in_deck)
            for card_i in unique_cards:
                for card_j in unique_cards:
                    if card_i != card_j:
                        co_occurrence[(card_i, card_j)] += 1
        return co_occurrence, num_decks

    co_occurrence, total_decks = build_co_occurrence(decks_folder)

    # Weight multipliers
    alpha = 1.0  # BERT mention
    beta = 0.5   # Normalized co-occurrence frequency
    gamma = 0.1  # Shared attribute bonus

    graph_edges = {}
    for card_id_i, info_i in card_info.items():
        desc_i = info_i.get("desc", "")
        mentioned_ids = get_mentioned_card_ids(desc_i, card_name_to_id) if desc_i else set()
        for card_id_j, info_j in card_info.items():
            if card_id_i == card_id_j:
                continue
            weight = 0.0
            if card_id_j in mentioned_ids:
                weight += alpha
            freq = co_occurrence.get((card_id_i, card_id_j), 0) / total_decks
            weight += beta * freq
            if info_i.get("attribute") and info_j.get("attribute"):
                if info_i["attribute"] == info_j["attribute"]:
                    weight += gamma
            if weight > 0:
                graph_edges[(card_id_i, card_id_j)] = weight

    print(f"Constructed graph with {len(graph_edges)} weighted edges.")

    # -------------------------
    # Build PyTorch Geometric Data Object
    # -------------------------
    attributes = list({card.get("attribute", "None") for card in cards_data})
    attribute_to_idx = {attr: i for i, attr in enumerate(attributes)}

    node_features = []
    node_id_to_card_id = {}
    card_id_to_node_id = {}
    for i, (card_id, card) in enumerate(card_info.items()):
        one_hot = [0] * len(attributes)
        attr = card.get("attribute", "None")
        one_hot[attribute_to_idx[attr]] = 1
        level = (card.get("level") or 0) / 12.0
        atk = (card.get("atk") or 0) / 5000.0
        defense = (card.get("def") or 0) / 5000.0
        feature = one_hot + [level, atk, defense]
        node_features.append(feature)
        node_id_to_card_id[i] = card_id
        card_id_to_node_id[card_id] = i

    x = torch.tensor(node_features, dtype=torch.float)

    edge_index_list = []
    for (card_id_i, card_id_j), weight in graph_edges.items():
        if card_id_i in card_id_to_node_id and card_id_j in card_id_to_node_id:
            i = card_id_to_node_id[card_id_i]
            j = card_id_to_node_id[card_id_j]
            edge_index_list.append([i, j])

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)

    # Package everything in a dictionary.
    graph_dict = {
        "data": data,
        "node_id_to_card_id": node_id_to_card_id,
        "card_id_to_node_id": card_id_to_node_id,
        "graph_edges": graph_edges,
        "cards_data": cards_data,
        "card_info": card_info,
        "card_name_to_id": card_name_to_id,
        "attributes": attributes,
        "attribute_to_idx": attribute_to_idx,
        "decks_folder": decks_folder
    }
    return graph_dict

def get_cached_graph():
    """Ask the user if data has been added. Rebuild graph if yes; otherwise load cache."""
    answer = input("Has new data been added? (y/n): ").strip().lower()
    if answer == "y" or not os.path.exists(CACHE_FILE):
        print("Building graph from new data...")
        graph_dict = build_graph()
        torch.save(graph_dict, CACHE_FILE)
    else:
        print("Loading cached graph...")
        graph_dict = torch.load(CACHE_FILE)
    return graph_dict

# -------------------------
# Get (or build) the cached graph.
# -------------------------
graph_cache = get_cached_graph()
data = graph_cache["data"]
node_id_to_card_id = graph_cache["node_id_to_card_id"]
card_id_to_node_id = graph_cache["card_id_to_node_id"]
graph_edges = graph_cache["graph_edges"]
cards_data = graph_cache["cards_data"]
card_info = graph_cache["card_info"]
card_name_to_id = graph_cache["card_name_to_id"]
attributes = graph_cache["attributes"]
attribute_to_idx = graph_cache["attribute_to_idx"]
decks_folder = graph_cache["decks_folder"]

# -------------------------
# The rest of the code (model definition, training, deck generation, etc.) remains unchanged.
# -------------------------

# 3. BUILD THE GNN MODEL & TRAINING TARGETS

class DeckGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(DeckGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # Output 3 values per node: [main_count, extra_count, side_count]
        self.out = nn.Linear(hidden_channels, 3)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return F.relu(self.out(x))

# Build training targets from deck files.
training_decks = []
deck_files = [f for f in os.listdir(decks_folder) if f.endswith(".ydk")]

def parse_ydk(file_path):
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

for deck_file in deck_files:
    deck_path = os.path.join(decks_folder, deck_file)
    deck = parse_ydk(deck_path)
    main_counter = Counter(deck.get("main", []))
    extra_counter = Counter(deck.get("extra", []))
    side_counter = Counter(deck.get("side", []))
    target = torch.zeros(data.x.shape[0], 3, dtype=torch.float)
    for card_id, count in main_counter.items():
        if card_id in card_id_to_node_id:
            target[card_id_to_node_id[card_id], 0] = float(count)
    for card_id, count in extra_counter.items():
        if card_id in card_id_to_node_id:
            target[card_id_to_node_id[card_id], 1] = float(count)
    for card_id, count in side_counter.items():
        if card_id in card_id_to_node_id:
            target[card_id_to_node_id[card_id], 2] = float(count)
    training_decks.append(target)

# 4. TRAINING & DECK GENERATION FUNCTIONS

MODEL_SAVE_PATH = "deck_gnn.pt"

def train_model(num_epochs=50, model_save_path=MODEL_SAVE_PATH):
    model_gnn = DeckGNN(in_channels=data.x.shape[1], hidden_channels=64)
    optimizer = optim.Adam(model_gnn.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    print("Starting training...")
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        model_gnn.train()
        optimizer.zero_grad()
        out = model_gnn(data.x, data.edge_index)
        loss_total = 0
        for target in training_decks:
            loss_total += criterion(out, target)
        loss_total = loss_total / len(training_decks)
        loss_total.backward()
        optimizer.step()
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:03d} | Loss: {loss_total.item():.4f} | LR: {current_lr}")
    torch.save(model_gnn.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return model_gnn

def load_model(model_save_path=MODEL_SAVE_PATH):
    model = DeckGNN(in_channels=data.x.shape[1], hidden_channels=64)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    return model

def fetch_extra_deck_keywords(decks_folder):
    extra_keywords = set()
    deck_files = [f for f in os.listdir(decks_folder) if f.endswith(".ydk")]
    for deck_file in deck_files:
        deck_path = os.path.join(decks_folder, deck_file)
        deck = parse_ydk(deck_path)
        for card_id in deck.get("extra", []):
            if card_id in card_info:
                card_type = card_info[card_id].get("type", "")
                extra_keywords.update(card_type.split())
    return extra_keywords

def fetch_banlist():
    try:
        url = "https://db.ygoprodeck.com/api/v7/banlistinfo.php?banlist=tcg"
        response = requests.get(url)
        if response.status_code == 200:
            ban_data = response.json()
            banned_ids = set()
            for card in ban_data.get("data", []):
                ban_info = card.get("banlist_info", {})
                if ban_info.get("ban_tcg") == "Banned":
                    banned_ids.add(str(card["id"]))
            return banned_ids
        else:
            print("Failed to fetch banlist, status code:", response.status_code)
            return set()
    except Exception as e:
        print("Error fetching banlist:", e)
        return set()

def generate_deck(model, seed_card_ids=None, synergy_factor=1.0, banned_cards=None):
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index)
    if seed_card_ids is not None:
        bonus = torch.zeros_like(pred)
        seed_node_ids = [card_id_to_node_id[cid] for cid in seed_card_ids if cid in card_id_to_node_id]
        for seed_node in seed_node_ids:
            seed_card_id = node_id_to_card_id[seed_node]
            for candidate_node in range(len(pred)):
                candidate_card_id = node_id_to_card_id[candidate_node]
                edge_weight = graph_edges.get((seed_card_id, candidate_card_id), 0.0)
                bonus[candidate_node] += synergy_factor * edge_weight
        pred = pred + bonus
    if banned_cards is not None:
        for node_idx in range(len(pred)):
            card_id = node_id_to_card_id[node_idx]
            if card_id in banned_cards:
                pred[node_idx] = 0.0
    # Process Main Deck: cap at 3 copies.
    main_counts = torch.clamp(torch.round(pred[:, 0]), max=3)
    main_deck = []
    for idx, count in enumerate(main_counts):
        for _ in range(int(count.item())):
            main_deck.append(node_id_to_card_id[idx])
    total_main = len(main_deck)
    if total_main < 40:
        deficit = 40 - total_main
        sorted_idx = torch.argsort(pred[:, 0], descending=True)
        for idx in sorted_idx:
            card_id = node_id_to_card_id[idx.item()]
            current_count = main_deck.count(card_id)
            if current_count < 3:
                main_deck.append(card_id)
                deficit -= 1
                if deficit <= 0:
                    break
    elif total_main > 60:
        while len(main_deck) > 60:
            main_deck.sort(key=lambda cid: pred[card_id_to_node_id[cid], 0])
            main_deck.pop(0)
    # Process Extra Deck: cap at 1 copy.
    extra_counts = torch.clamp(torch.round(pred[:, 1]), max=1)
    extra_deck = []
    for idx, count in enumerate(extra_counts):
        for _ in range(int(count.item())):
            extra_deck.append(node_id_to_card_id[idx])
    # Process Side Deck: cap at 3 copies.
    side_counts = torch.clamp(torch.round(pred[:, 2]), max=3)
    side_deck = []
    for idx, count in enumerate(side_counts):
        for _ in range(int(count.item())):
            side_deck.append(node_id_to_card_id[idx])
    def sort_key(card_id):
        card_type = card_info[card_id].get("type", "").lower()
        if "monster" in card_type:
            return 0
        elif "trap" in card_type:
            return 1
        elif "spell" in card_type:
            return 2
        else:
            return 3
    main_deck.sort(key=sort_key)
    return main_deck, extra_deck, side_deck

def create_deck(seed_card_ids=None, synergy_factor=1.0, retrain=False, banned_cards=None):
    if retrain or not os.path.exists(MODEL_SAVE_PATH):
        print("Training model with current data...")
        model = train_model(num_epochs=50, model_save_path=MODEL_SAVE_PATH)
    else:
        print("Loading saved model...")
        model = load_model(MODEL_SAVE_PATH)
    return generate_deck(model, seed_card_ids=seed_card_ids,
                         synergy_factor=synergy_factor, banned_cards=banned_cards)

# 5. USAGE EXAMPLE & .YDK FILE CREATION

if __name__ == "__main__":
    retrain_input = input("Do you want to retrain the model with new data? (y/n): ").strip().lower()
    retrain_flag = retrain_input == "y"
    banlist_input = input("Should the banlist be respected? (y/n): ").strip().lower()
    if banlist_input == "y":
        banlist = fetch_banlist()
    else:
        banlist = None
    seed_input = input("Enter one or more seed card IDs, separated by commas (or leave blank for none): ").strip()
    seed_ids = [s.strip() for s in seed_input.split(",")] if seed_input else None
    main_deck, extra_deck, side_deck = create_deck(seed_card_ids=seed_ids, retrain=retrain_flag, banned_cards=banlist)
    deck_name = input("Enter a name for your deck: ").strip()
    if not deck_name:
        deck_name = "generated_deck"
    ydk_lines = []
    ydk_lines.append(f"#created by DeckGNN")
    ydk_lines.append(f"#name {deck_name}")
    ydk_lines.append("#main")
    for card in main_deck:
        ydk_lines.append(card)
    ydk_lines.append("#extra")
    for card in extra_deck:
        ydk_lines.append(card)
    ydk_lines.append("!side")
    for card in side_deck:
        ydk_lines.append(card)
    ydk_filename = deck_name.replace(" ", "_") + ".ydk"
    with open(ydk_filename, "w") as f:
        f.write("\n".join(ydk_lines))
    print(f"\n--- Generated Deck ---")
    print(f"Deck Name: {deck_name}")
    print("Main Deck ({} cards):".format(len(main_deck)), main_deck)
    print("Extra Deck ({} cards):".format(len(extra_deck)), extra_deck)
    print("Side Deck ({} cards):".format(len(side_deck)), side_deck)
    print(f"\nDeck file saved as {ydk_filename}")
