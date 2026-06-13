import torch
import os
from model import create_model
from ph2_loader import get_ph2_loaders
import matplotlib.pyplot as plt

DEVICE = torch.device("cpu")

def save_predictions(model, loader, save_dir="predictions", num_images=5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    count = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            preds = (outputs > 0.5).float()

            for i in range(images.size(0)):
                if count >= num_images:
                    return

                img = images[i].cpu().permute(1, 2, 0)
                mask = masks[i].cpu().squeeze()
                pred = preds[i].cpu().squeeze()

                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(img)
                ax[0].set_title("Image")
                ax[1].imshow(mask, cmap="gray")
                ax[1].set_title("Ground Truth")
                ax[2].imshow(pred, cmap="gray")
                ax[2].set_title("Prediction")

                for a in ax:
                    a.axis("off")

                plt.tight_layout()
                plt.savefig(f"{save_dir}/result_{count}.png")
                plt.close()

                count += 1
def dice_score(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / \
           (preds.sum() + targets.sum() + smooth)


def iou_score(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def evaluate(model, loader):
    model.eval()
    dice_total = 0
    iou_total = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            preds = (outputs > 0.5).float()

            dice_total += dice_score(preds, masks)
            iou_total += iou_score(preds, masks)

    return dice_total / len(loader), iou_total / len(loader)


if __name__ == "__main__":

    # ✅ Charger les mêmes loaders que training
    train_loader, val_loader = get_ph2_loaders(
        data_root="data/images",
        val_split=0.2,
        batch_size=4,
    )

    model = create_model().to(DEVICE)

    model_path = "models/unet_skin.pth"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("✅ Modèle chargé :", model_path)
    else:
        print("❌ Modèle introuvable")

    dice, iou = evaluate(model, val_loader)
    save_predictions(model, val_loader)
    print("✅ Images sauvegardées dans le dossier 'predictions'")
    print("\n📊 Résultats sur Validation Set")
    print(f"Dice Score : {dice:.4f}")
    print(f"IoU Score  : {iou:.4f}")