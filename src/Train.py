import os
import sys
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── IMPORTS CORRECTS — utilise ph2_loader, PAS dataset.get_loaders ──────────
from model import create_model, CombinedLoss
from ph2_loader import get_ph2_loaders

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
CONFIG = {
    "data_root"  : "data/images",   # dossier contenant les IMDxxx
    "val_split"  : 0.2,
    "epochs"     : 10,
    "batch_size" : 4,
    "lr"         : 1e-4,
    "model_path" : "models/unet_skin.pth",
    "seed"       : 42,
}


def set_seed(seed):
    import numpy as np, random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            predictions = model(images)
            loss = criterion(predictions, masks)
            total_loss += loss.item()
    return total_loss / len(loader)


def plot_training_curves(history, save_path="training_curves.png"):
    import matplotlib.pyplot as plt
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], 'b-o', label='Train Loss', linewidth=2)
    ax.plot(epochs, history["val_loss"],   'r-o', label='Val Loss',   linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Courbes d'entraînement — SkinVision U-Net")
    ax.legend()
    ax.grid(True, alpha=0.3)
    min_val   = min(history["val_loss"])
    min_epoch = history["val_loss"].index(min_val) + 1
    ax.annotate(f'Best: {min_val:.4f}', xy=(min_epoch, min_val),
                xytext=(min_epoch + 0.3, min_val + 0.02),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Courbes sauvegardées : {save_path}")
    plt.close()


def train(config=CONFIG):
    set_seed(config["seed"])
    device = torch.device("cpu")
    print(f"Device : {device}")

    os.makedirs(os.path.dirname(config["model_path"]), exist_ok=True)

    # ── Données — ph2_loader uniquement ──────────────────────────────────────
    print("\nChargement des données PH2...")
    train_loader, val_loader = get_ph2_loaders(
        data_root  = config["data_root"],
        val_split  = config["val_split"],
        batch_size = config["batch_size"],
    )

    # ── Modèle ────────────────────────────────────────────────────────────────
    print("\nInitialisation du modèle...")
    model     = create_model().to(device)
    optimizer = Adam(model.parameters(), lr=config["lr"])
    criterion = CombinedLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history      = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')

    print(f"\nDébut entraînement : {config['epochs']} epochs\n" + "─"*50)

    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Val   Loss : {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config["model_path"])
            print(f"  ✓ Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")

        print("─"*50)

    print(f"\n✓ Entraînement terminé. Meilleure val_loss : {best_val_loss:.4f}")
    print(f"  Modèle : {config['model_path']}")

    plot_training_curves(history)
    return model, history


if __name__ == "__main__":
    model, history = train(CONFIG)