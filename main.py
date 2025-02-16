import os
import json
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import requests  # For fetching banlist (if needed)
from tqdm import tqdm  # For progress bars

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

#####################################
# 1. GRAPH CONSTRUCTION & CACHING
#####################################

CACHE_FILE = "graph_cache.pt"

def build_graph():
    print(">> Building graph from scratch...")

    # Load card data from cards.json
    with open("cards.json", "r") as f:
        cards_data = json.load(f)["data"]

    # Build lookups.
    card_name_to_id = {card["name"].lower(): str(card["id"]) for card in cards_data}
    card_info = {str(card["id"]): card for card in cards_data}

    # Initialize BERT NER Pipeline.
    print(">> Initializing BERT NER pipeline...")
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

    # Build co-occurrence dictionary from deck files.
    decks_folder = "ydk_download"
    def parse_ydk_inner(file_path):
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
        print(f">> Found {num_decks} deck files for co-occurrence stats.")
        for deck_file in deck_files:
            deck_path = os.path.join(decks_folder, deck_file)
            deck = parse_ydk_inner(deck_path)
            cards_in_deck = list(set(deck.get("main", []) + deck.get("extra", []) + deck.get("side", [])))
            for card_i in cards_in_deck:
                for card_j in cards_in_deck:
                    if card_i != card_j:
                        co_occurrence[(card_i, card_j)] += 1
        return co_occurrence, num_decks

    co_occurrence, total_decks = build_co_occurrence(decks_folder)

    # Build weighted graph edges.
    alpha = 1.0  # Mention weight
    beta = 0.5   # Co-occurrence frequency weight
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

    print(f">> Constructed graph with {len(graph_edges)} weighted edges.")

    # Build PyTorch Geometric Data Object.
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
    print(">> Graph built successfully.")
    return graph_dict

def get_cached_graph():
    answer = input("Has new data been added? (y/n): ").strip().lower()
    if answer == "y" or not os.path.exists(CACHE_FILE):
        graph_dict = build_graph()
        torch.save(graph_dict, CACHE_FILE)
    else:
        print(">> Loading cached graph...")
        graph_dict = torch.load(CACHE_FILE, map_location=torch.device('cpu'))
    return graph_dict

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

#####################################
# 2. GNN MODEL & TRAINING TARGETS (RANKING / CLASSIFICATION)
#####################################

# We'll reformulate the task as binary classification per deck section.
# For each card in a deck file, if it appears in that section, label = 1, otherwise 0.

# Model: We'll output 3 logits per card (for main, extra, and side), and use BCEWithLogitsLoss.
class DeckGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(DeckGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, 3)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # Output logits; BCEWithLogitsLoss will handle the sigmoid.
        return self.out(x)

# Build training targets as binary labels.
training_decks = []
deck_files = [f for f in os.listdir(decks_folder) if f.endswith(".ydk")]

def parse_ydk(file_path):
    if not os.path.exists(file_path):
        if not file_path.startswith(decks_folder):
            file_path = os.path.join(decks_folder, file_path)
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

print(">> Building training targets from deck files...")
for deck_file in deck_files:
    deck_path = os.path.join(decks_folder, deck_file)
    deck = parse_ydk(deck_path)
    # Binary labels: if a card appears (count>0) then label=1, else 0.
    main_counter = Counter(deck.get("main", []))
    extra_counter = Counter(deck.get("extra", []))
    side_counter = Counter(deck.get("side", []))
    target = torch.zeros(data.x.shape[0], 3, dtype=torch.float)
    for card_id, count in main_counter.items():
        if card_id in card_id_to_node_id:
            target[card_id_to_node_id[card_id], 0] = 1.0 if count > 0 else 0.0
    for card_id, count in extra_counter.items():
        if card_id in card_id_to_node_id:
            target[card_id_to_node_id[card_id], 1] = 1.0 if count > 0 else 0.0
    for card_id, count in side_counter.items():
        if card_id in card_id_to_node_id:
            target[card_id_to_node_id[card_id], 2] = 1.0 if count > 0 else 0.0
    training_decks.append(target)
print(">> Training targets built.")

# Debug: print average target value for main deck.
all_targets = torch.stack(training_decks)
mean_target = all_targets[:, :, 0].mean().item()
print(">> Mean training target (main deck):", mean_target)

#####################################
# 3. HISTORICAL STATISTICAL FUNCTIONS
#####################################

def get_section_frequency(deck_files, section):
    freq = Counter()
    for deck_file in deck_files:
        deck_path = os.path.join(decks_folder, deck_file)
        sections = parse_ydk(deck_path)
        freq.update(sections[section])
    return freq

def fill_from_sorted_recommendations(sorted_recs, desired_size, dropoff_ratio=0.3):
    deck_section = []
    if not sorted_recs:
        return deck_section
    max_freq = sorted_recs[0][1]
    for card, freq in sorted_recs:
        if len(deck_section) >= desired_size:
            break
        if freq < dropoff_ratio * max_freq:
            break
        if card not in deck_section:
            deck_section.append(card)
    return deck_section

def compute_average_copy(card_id, deck_files):
    total_copies = 0
    deck_count = 0
    for deck_file in deck_files:
        deck_path = os.path.join(decks_folder, deck_file)
        sections = parse_ydk(deck_path)
        main_d = sections["main"]
        if card_id in main_d:
            total_copies += main_d.count(card_id)
            deck_count += 1
    if deck_count == 0:
        return 1
    avg = total_copies / deck_count
    return max(1, min(3, round(avg)))

def adjust_main_deck_copy_counts(unique_deck, deck_files, desired_size):
    new_deck = []
    for card in unique_deck:
        rec_copies = compute_average_copy(card, deck_files)
        rec_copies = min(rec_copies, 3)
        for _ in range(rec_copies):
            if new_deck.count(card) < 3 and len(new_deck) < desired_size:
                new_deck.append(card)
        if len(new_deck) >= desired_size:
            break
    idx = 0
    while len(new_deck) < desired_size and unique_deck:
        card = unique_deck[idx % len(unique_deck)]
        if new_deck.count(card) < 3:
            new_deck.append(card)
        idx += 1
    return new_deck

def build_extra_deck(deck_files, desired_size=15, dropoff_ratio=0.3):
    freq_extra = get_section_frequency(deck_files, "extra")
    valid_extra = {}
    for card_id, freq in freq_extra.items():
        info = card_info.get(card_id, {})
        ctype = info.get("type", "").lower()
        if any(x in ctype for x in ["fusion", "synchro", "xyz", "link"]):
            valid_extra[card_id] = freq
    sorted_recs = sorted(valid_extra.items(), key=lambda x: x[1], reverse=True)
    return fill_from_sorted_recommendations(sorted_recs, desired_size, dropoff_ratio)

def build_side_deck(deck_files, desired_size=15, dropoff_ratio=0.3):
    freq_side = get_section_frequency(deck_files, "side")
    sorted_recs = sorted(freq_side.items(), key=lambda x: x[1], reverse=True)
    return fill_from_sorted_recommendations(sorted_recs, desired_size, dropoff_ratio)

#####################################
# New: Dynamic Main Deck Filling Function
#####################################

def dynamic_fill_main_deck(unique_candidates, main_pred, min_size=40, max_size=60, gap_threshold=0.5):
    """
    Given a list of candidate card IDs and the GNN main predictions (tensor of probabilities),
    sort candidates by descending predicted score and add them until:
      - At least min_size candidates are added.
      - Then, if a candidate's score is much lower than the previous candidate's score (gap > gap_threshold),
        stop adding further candidates.
    Only candidates present in card_id_to_node_id are considered.
    Returns a list of candidate card IDs.
    """
    candidate_scores = []
    for cid in unique_candidates:
        if cid not in card_id_to_node_id:
            continue
        score = main_pred[card_id_to_node_id[cid]].item()
        candidate_scores.append((cid, score))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    final_candidates = []
    for i, (cid, score) in enumerate(candidate_scores):
        if i == 0:
            final_candidates.append(cid)
        else:
            if len(final_candidates) >= min_size:
                prev_score = candidate_scores[i-1][1]
                if (prev_score - score) > gap_threshold:
                    break
            final_candidates.append(cid)
        if len(final_candidates) >= max_size:
            break
    return final_candidates

#####################################
# 4. DECK GENERATION FUNCTIONS (HYBRID: GNN + HISTORICAL)
#####################################

EXTRA_ONLY_TYPES = {"fusion", "synchro", "xyz", "link", "pendulum"}

def generate_deck(model, seed_card_ids=None, synergy_factor=1.0, banned_cards=None):
    print(">> Generating deck using GNN and historical stats...")

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)  # shape: (num_nodes, 3)
    # Convert logits to probabilities.
    probs = torch.sigmoid(logits)
    main_prob = probs[:, 0]
    print(">> GNN main prediction stats -- mean:", main_prob.mean().item(), "max:", main_prob.max().item())

    if seed_card_ids:
        bonus = torch.zeros_like(probs)
        seed_node_ids = [card_id_to_node_id[cid] for cid in seed_card_ids if cid in card_id_to_node_id]
        for seed_node in seed_node_ids:
            seed_card_id = node_id_to_card_id[seed_node]
            for candidate_node in range(len(probs)):
                candidate_card_id = node_id_to_card_id[candidate_node]
                edge_weight = graph_edges.get((seed_card_id, candidate_card_id), 0.0)
                bonus[candidate_node] += synergy_factor * edge_weight
        probs = probs + bonus

    if banned_cards:
        for node_idx in range(len(probs)):
            card_id = node_id_to_card_id[node_idx]
            if card_id in banned_cards:
                probs[node_idx] = 0.0

    # Build raw main deck from predicted probabilities.
    raw_main = []
    for idx, score in enumerate(probs[:, 0]):
        card_id = node_id_to_card_id[idx]
        info = card_info.get(card_id, {})
        ctype = info.get("type", "").lower()
        if any(keyword in ctype for keyword in EXTRA_ONLY_TYPES):
            continue
        # Use a threshold: if score > 0.5, then include it once.
        if score.item() > 0.5:
            raw_main.append(card_id)
    print(f">> Raw main deck from GNN has {len(raw_main)} cards.")

    # If the raw main deck is under-sized, supplement from historical frequencies.
    if len(raw_main) < 40:
        print(">> Main deck is under-sized; supplementing with popular main deck cards...")
        freq_main = get_section_frequency(deck_files, "main")
        sorted_main = sorted(freq_main.items(), key=lambda x: x[1], reverse=True)
        for rec_card, freq in sorted_main:
            if rec_card not in card_id_to_node_id:
                continue
            info = card_info.get(rec_card, {})
            ctype = info.get("type", "").lower()
            if any(x in ctype for x in EXTRA_ONLY_TYPES):
                continue
            if rec_card not in raw_main:
                raw_main.append(rec_card)
            if len(raw_main) >= 40:
                break

    unique_main = list(dict.fromkeys(raw_main))
    # Use dynamic filling: stop if there is a big gap after reaching min_size.
    final_candidates = dynamic_fill_main_deck(unique_main, main_prob, min_size=40, max_size=60, gap_threshold=0.5)
    print(">> Dynamic main deck candidate count:", len(final_candidates))

    final_main = adjust_main_deck_copy_counts(final_candidates, deck_files, desired_size=len(final_candidates))
    final_main = enforce_max_three(final_main)
    # final_main can be between 40 and 60 cards.

    def main_sort_key(card_id):
        ctype = card_info.get(card_id, {}).get("type", "").lower()
        if "monster" in ctype:
            return 0
        elif "trap" in ctype:
            return 1
        elif "spell" in ctype:
            return 2
        else:
            return 3
    final_main.sort(key=main_sort_key)

    final_extra = build_extra_deck(deck_files, desired_size=15, dropoff_ratio=0.3)
    final_side = build_side_deck(deck_files, desired_size=15, dropoff_ratio=0.3)

    print(">> Deck generation complete.")
    return final_main, final_extra, final_side

def create_deck(seed_card_ids=None, synergy_factor=1.0, retrain=False, banned_cards=None):
    MODEL_SAVE_PATH = "deck_gnn.pt"
    if retrain or not os.path.exists(MODEL_SAVE_PATH):
        print(">> Training model with current data...")
        model = train_model(num_epochs=50, model_save_path=MODEL_SAVE_PATH)
    else:
        print(">> Loading saved model...")
        model = load_model(MODEL_SAVE_PATH)
    return generate_deck(model, seed_card_ids=seed_card_ids, synergy_factor=synergy_factor, banned_cards=banned_cards)

#####################################
# Enforce Maximum Three Copies Function
#####################################

def enforce_max_three(deck):
    """Ensure that no card appears more than 3 times in the deck."""
    new_deck = []
    counts = {}
    for card in deck:
        counts[card] = counts.get(card, 0)
        if counts[card] < 3:
            new_deck.append(card)
            counts[card] += 1
    return new_deck

#####################################
# 5. MODEL TRAINING & LOADING FUNCTIONS
#####################################

MODEL_SAVE_PATH = "deck_gnn.pt"

def train_model(num_epochs=50, model_save_path=MODEL_SAVE_PATH):
    print(">> Starting model training...")
    model = DeckGNN(in_channels=data.x.shape[1], hidden_channels=64)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Use BCEWithLogitsLoss (it applies sigmoid internally).
    # We'll set pos_weight to help with imbalance (tune as needed).
    pos_weight = torch.tensor([5.0, 5.0, 5.0])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)  # shape: (num_nodes, 3)
        loss_total = 0
        for target in training_decks:
            loss_total += loss_fn(logits, target)
        loss_total = loss_total / len(training_decks)
        loss_total.backward()
        optimizer.step()
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f">> Epoch {epoch:03d} | Loss: {loss_total.item():.4f} | LR: {current_lr}")
    torch.save(model.state_dict(), model_save_path)
    print(f">> Model saved to {model_save_path}")
    return model

def load_model(model_save_path=MODEL_SAVE_PATH):
    print(">> Loading model from disk...")
    model = DeckGNN(in_channels=data.x.shape[1], hidden_channels=64)
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    model.eval()
    return model

#####################################
# 6. EXPORT DECK TO .YDK
#####################################

def export_deck_to_ydk(deck, deck_name):
    filename = f"{deck_name}.ydk"
    sorted_main = sorted(deck["main"], key=lambda cid: card_info.get(cid, {}).get("type", ""))
    sorted_extra = sorted(deck["extra"], key=lambda cid: card_info.get(cid, {}).get("type", ""))
    sorted_side = sorted(deck["side"], key=lambda cid: card_info.get(cid, {}).get("type", ""))
    with open(filename, "w") as f:
        f.write(f"!name {deck_name}\n")
        f.write("#main\n")
        for card in sorted_main:
            f.write(f"{card}\n")
        f.write("#extra\n")
        for card in sorted_extra:
            f.write(f"{card}\n")
        f.write("!side\n")
        for card in sorted_side:
            f.write(f"{card}\n")
    print(f"\n>> Deck exported to {filename}")

#####################################
# 7. MAIN EXECUTION
#####################################

if __name__ == "__main__":
    print("\n===== Deck Generation Script Started =====\n")
    retrain_input = input("Do you want to retrain the model with new data? (y/n): ").strip().lower()
    retrain_flag = retrain_input == "y"

    banlist_input = input("Should the banlist be respected? (y/n): ").strip().lower()
    if banlist_input == "y":
        banned_cards = set()  # Placeholder for banned card IDs.
    else:
        banned_cards = None

    seed_input = input("Enter one or more seed card IDs, separated by commas (or leave blank for none): ").strip()
    seed_ids = [s.strip() for s in seed_input.split(",")] if seed_input else None

    main_deck, extra_deck, side_deck = create_deck(seed_card_ids=seed_ids, synergy_factor=1.0, retrain=retrain_flag, banned_cards=banned_cards)

    deck_name = input("Enter a name for your deck: ").strip()
    if not deck_name:
        deck_name = "generated_deck"

    final_deck = {
        "main": main_deck,
        "extra": extra_deck,
        "side": side_deck
    }
    export_deck_to_ydk(final_deck, deck_name)

    print("\n===== Generated Deck =====")
    print(f"Deck Name: {deck_name}")
    print("Main Deck ({} cards):".format(len(main_deck)), main_deck)
    print("Extra Deck ({} cards):".format(len(extra_deck)), extra_deck)
    print("Side Deck ({} cards):".format(len(side_deck)), side_deck)
