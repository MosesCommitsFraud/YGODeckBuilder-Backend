import os
import json
import requests
import itertools
import networkx as nx
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE

# -------------------------------
# 1. Parse .ydk files into decks
# -------------------------------
def parse_ydk(file_path):
    """
    Parses a .ydk file into a dictionary with keys 'main', 'extra', and 'side'.
    """
    decks = {"main": [], "extra": [], "side": []}
    section = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                # A new section (e.g., "#main" or "#extra")
                section = line[1:].lower()
                if section not in decks:
                    decks[section] = []
            elif line.startswith("!"):
                # Sometimes the side deck is marked with "!" (e.g., "!side")
                section = line[1:].lower()
                if section not in decks:
                    decks[section] = []
            elif line and line[0].isdigit():
                if section is not None:
                    decks[section].append(line)
    return decks

# -------------------------------
# 2. Fetch card data from the API with caching
# -------------------------------
CACHE_FILE = "cards.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}
    return cache

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def fetch_card_data(card_id, cache):
    """
    Fetches card data from the API for a given card ID.
    Caches results in a local JSON file.
    """
    card_id_str = str(card_id)
    if card_id_str in cache:
        return cache[card_id_str]
    url = f"https://db.ygoprodeck.com/api/v7/cardinfo.php?id={card_id_str}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "data" in data:
            card_info = data["data"][0]  # Take the first result
            cache[card_id_str] = card_info
            return card_info
    else:
        print(f"Failed to fetch card {card_id_str}, status code: {response.status_code}")
    return None

# -------------------------------
# 3. Build a co-occurrence graph from decks
# -------------------------------
def build_cooccurrence_graph(decks_list):
    """
    Given a list of decks (each a dict with 'main', 'extra', and 'side'),
    build an undirected weighted graph where nodes are card IDs and edges
    indicate that two cards have appeared in the same deck.
    """
    G = nx.Graph()
    for deck in decks_list:
        # Combine cards from all sections (you could also focus on 'main' only)
        cards = deck.get("main", []) + deck.get("extra", []) + deck.get("side", [])
        unique_cards = list(set(cards))
        # Add nodes
        for card in unique_cards:
            if not G.has_node(card):
                G.add_node(card)
        # Add/update edges for each pair of cards in the deck
        for card1, card2 in itertools.combinations(unique_cards, 2):
            if G.has_edge(card1, card2):
                G[card1][card2]['weight'] += 1
            else:
                G.add_edge(card1, card2, weight=1)
    return G

# -------------------------------
# 4. Create node features for each card
# -------------------------------
def create_node_features(G, cache):
    """
    Creates a feature vector for each card (node).
    For simplicity, we extract a few numerical properties from the card data.
    If a card's data is missing, we fallback to a random vector.
    """
    num_features = 16  # Total feature vector size
    features = {}
    for node in G.nodes():
        card_info = cache.get(str(node))
        if card_info:
            # Use level, atk, and def (if available). Many cards may not have level.
            level = card_info.get("level", 0)
            atk = card_info.get("atk", 0)
            defense = card_info.get("def", 0)
            # Normalize (these numbers are approximate; adjust as needed)
            feat = torch.tensor([
                level / 12.0,
                atk / 5000.0,
                defense / 5000.0
            ], dtype=torch.float)
            # Pad to num_features
            if feat.shape[0] < num_features:
                pad = torch.zeros(num_features - feat.shape[0])
                feat = torch.cat([feat, pad], dim=0)
            features[node] = feat
        else:
            # Fallback to a random feature vector
            features[node] = torch.randn(num_features)
    return features, num_features

# -------------------------------
# 5. Convert the NetworkX graph to a PyTorch Geometric Data object
# -------------------------------
def nx_to_pyg_data(G, features, num_features):
    """
    Converts a NetworkX graph and node features dictionary into a PyG Data object.
    """
    node_list = list(G.nodes())
    # Build a feature matrix in the order of node_list
    x = torch.stack([features[node] for node in node_list])
    # Create a mapping from node ID to index
    node_idx = {node: i for i, node in enumerate(node_list)}
    # Build edge indices (make the graph undirected)
    edges = []
    for u, v in G.edges():
        i = node_idx[u]
        j = node_idx[v]
        edges.append([i, j])
        edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data, node_idx

# -------------------------------
# 6. Define the GNN model (GCN encoder for a Graph Autoencoder)
# -------------------------------
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# -------------------------------
# 7. Train the Graph Autoencoder
# -------------------------------
def train_gae(data, encoder, optimizer, epochs=200):
    """
    Trains a Graph Autoencoder (GAE) using the given encoder and data.
    """
    model = GAE(encoder)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")
    return model

# -------------------------------
# 8. Main pipeline function
# -------------------------------
def main():
    # (a) Locate and parse all .ydk files in the "decks" directory.
    decks_dir = "ydk_download"  # Adjust this path as needed.
    deck_files = [os.path.join(decks_dir, f) for f in os.listdir(decks_dir) if f.endswith(".ydk")]
    all_decks = []
    for deck_file in deck_files:
        deck = parse_ydk(deck_file)
        all_decks.append(deck)
    print(f"Parsed {len(all_decks)} decks.")

    # (b) Gather all unique card IDs from the decks.
    all_card_ids = set()
    for deck in all_decks:
        for section in deck:
            all_card_ids.update(deck[section])
    print(f"Found {len(all_card_ids)} unique card IDs.")

    # (c) Load cached card data and fetch missing data from the API.
    cache = load_cache()
    for card_id in tqdm(all_card_ids, desc="Fetching card data"):
        fetch_card_data(card_id, cache)
    save_cache(cache)
    print("Card data fetched and cache saved.")

    # (d) Build the co-occurrence graph from the decks.
    G = build_cooccurrence_graph(all_decks)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # (e) Create node features using card data.
    features, num_features = create_node_features(G, cache)

    # (f) Convert the graph to a PyG Data object.
    data, node_idx = nx_to_pyg_data(G, features, num_features)
    print("Converted graph to PyTorch Geometric Data object.")

    # (g) Define the GCN encoder and optimizer.
    hidden_channels = 32
    out_channels = 16  # Latent dimension for embeddings
    encoder = GCNEncoder(num_features, hidden_channels, out_channels)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

    # (h) Train the Graph Autoencoder.
    print("Training Graph Autoencoder...")
    model = train_gae(data, encoder, optimizer, epochs=200)

    # (i) Retrieve and print node embeddings.
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
    print("Node embeddings shape:", z.shape)

    # (j) (Bonus) From here you could implement a deck-generation routine
    # that selects a subset of card nodes based on synergy (using z) and game rules.
    print("Pipeline complete. You can now use the learned embeddings for deck generation!")

if __name__ == "__main__":
    main()
