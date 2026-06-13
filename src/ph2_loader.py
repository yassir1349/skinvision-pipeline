import os
import sys
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import SkinDataset, train_transform


def get_ph2_pairs(data_root: str = "data/images"):
    """
    Parcourt la structure PH2 et retourne les paires (image_path, mask_path).
    Structure attendue :
      data_root/
        IMDxxx/
          IMDxxx_Dermoscopic_Image/IMDxxx.bmp   ← image
          IMDxxx_lesion/IMDxxx_lesion.bmp        ← masque
    """
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Dossier introuvable : {data_root}")

    image_paths = []
    mask_paths  = []
    skipped     = []

    patients = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d)) and d.startswith("IMD")
    ])

    if len(patients) == 0:
        raise ValueError(
            f"Aucun dossier IMDxxx trouvé dans {data_root}\n"
            f"Vérifie que tes images sont bien dans : {data_root}/IMD002/..."
        )

    print(f"Patients trouvés : {len(patients)}")

    for patient in patients:
        patient_dir = os.path.join(data_root, patient)
        img_path  = os.path.join(patient_dir, f"{patient}_Dermoscopic_Image", f"{patient}.bmp")
        mask_path = os.path.join(patient_dir, f"{patient}_lesion", f"{patient}_lesion.bmp")

        if os.path.exists(img_path) and os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)
        else:
            skipped.append(patient)
            if len(skipped) <= 3:  # Afficher seulement les 3 premiers
                if not os.path.exists(img_path):
                    print(f"  ⚠ Image manquante : {img_path}")
                if not os.path.exists(mask_path):
                    print(f"  ⚠ Masque manquant : {mask_path}")

    if skipped:
        print(f"⚠ {len(skipped)} patients ignorés (fichiers manquants)")

    print(f"✓ Paires valides : {len(image_paths)}")
    return image_paths, mask_paths


def get_ph2_loaders(
    data_root:  str   = "data/images",
    val_split:  float = 0.2,
    batch_size: int   = 4,
):
    """Crée les DataLoaders train/val pour PH2."""
    image_paths, mask_paths = get_ph2_pairs(data_root)

    np.random.seed(42)
    indices   = np.random.permutation(len(image_paths))
    split_idx = int(len(indices) * (1 - val_split))
    train_idx = indices[:split_idx]
    val_idx   = indices[split_idx:]

    print(f"  Train : {len(train_idx)} | Val : {len(val_idx)}")

    train_dataset = SkinDataset(
        image_paths=[image_paths[i] for i in train_idx],
        mask_paths =[mask_paths[i]  for i in train_idx],
        transform  =train_transform,
    )
    val_dataset = SkinDataset(
        image_paths=[image_paths[i] for i in val_idx],
        mask_paths =[mask_paths[i]  for i in val_idx],
        transform  =None,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


if __name__ == "__main__":
    import torch
    print("=== Test PH2 Loader ===\n")
    train_loader, val_loader = get_ph2_loaders("data/images")
    images, masks = next(iter(train_loader))
    print(f"images shape : {images.shape}")
    print(f"masks  shape : {masks.shape}")
    print(f"image range  : [{images.min():.2f}, {images.max():.2f}]")
    print(f"mask  values : {masks.unique()}")
    print("\n✓ Loader PH2 fonctionnel")