#!/usr/bin/env python3
"""
pretrain_gat_reaching_defs.py — Pre-train GAT Encoder for 5 epochs on reaching-definitions task (supervised).
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from gat_encoder import HeteroGATEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GATReachingDefsPredictor(nn.Module):
    def __init__(self, encoder: HeteroGATEncoder):
        super().__init__()
        self.encoder = encoder
        # Inst node embedding dimension after GAT layers is 128
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32)  # Predict reaching definition bits for 32 registers
        )

    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        # HeteroGATEncoder forward returns (graph_embed, inst_embeds)
        _, inst_embeds = self.encoder(x_dict, edge_index_dict, batch_dict=batch_dict)
        logits = self.head(inst_embeds)  # Shape [num_insts, 32]
        return logits


def pretrain_5_epochs(dataset_path: str = "reaching_defs_dataset.pt", save_path: str = "pretrained_gat_encoder.pt", epochs: int = 5):
    print("=" * 70)
    print(" SUPERVISED PRE-TRAINING OF GAT ENCODER ON REACHING DEFINITIONS")
    print("=" * 70)
    print(f" Compute Device : {DEVICE}")
    print(f" Target Epochs  : {epochs}")

    raw_data = torch.load(dataset_path, weights_only=False)
    print(f" Loaded {len(raw_data)} target samples from {dataset_path}")

    # Create PyG Data objects with y attributes attached
    pyg_list = []
    for g, target in raw_data:
        g.y = target  # Shape [N_inst, 32]
        pyg_list.append(g)

    loader = DataLoader(pyg_list, batch_size=32, shuffle=True)

    encoder = HeteroGATEncoder(
        inst_in_dim=177,
        reg_in_dim=1,
        hidden_channels=128,
        out_channels=256,
        num_heads=8,
        num_layers=3
    )
    model = GATReachingDefsPredictor(encoder).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in loader:
            x_dict = {k: v.to(DEVICE) for k, v in batch.x_dict.items()}
            edge_index_dict = {k: v.to(DEVICE) for k, v in batch.edge_index_dict.items()}
            batch_dict = {
                'inst': batch['inst'].batch.to(DEVICE),
                'reg': batch['reg'].batch.to(DEVICE)
            }
            targets = batch.y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x_dict, edge_index_dict, batch_dict=batch_dict)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

        mean_loss = total_loss / max(1, total_samples)
        print(f" Epoch {epoch}/{epochs} | Loss (BCEWithLogits): {mean_loss:.6f}")

    elapsed = time.time() - start_time
    print(f"\n [+] Pre-training Completed in {elapsed:.2f}s!")

    # Save encoder state dict
    torch.save(encoder.state_dict(), save_path)
    print(f" [+] Pre-trained GAT Encoder weights saved to {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    pretrain_5_epochs()
