import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def create_model() -> nn.Module:
    return smp.Unet(
        encoder_name    = "mobilenet_v2",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = 1,
        activation      = "sigmoid",
    )


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        pred_flat = predictions.view(-1)
        targ_flat = targets.view(-1)
        intersection = (pred_flat * targ_flat).sum()
        dice = (2.0 * intersection + self.smooth) / \
               (pred_flat.sum() + targ_flat.sum() + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(smooth=1.0)
        self.bce  = nn.BCELoss(reduction='mean')

    def forward(self, predictions, targets):
        return 0.5 * self.bce(predictions, targets) + \
               0.5 * self.dice(predictions, targets)


if __name__ == "__main__":
    model = create_model()
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres : {total:,}")
    dummy = torch.randn(2, 3, 128, 128)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {out.shape}")
    print(f"Range  : [{out.min():.3f}, {out.max():.3f}]")
    print("✓ Modèle OK")