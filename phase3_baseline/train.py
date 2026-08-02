import math
import os
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from phase3_baseline.model import PosePredictor
from phase3_baseline.dataset import LineModDataset
from common.data_split import prepare_data_and_splits
from phase3_baseline.losses import rotation_loss, translation_loss


def quaternion_angle_error(pred, target):
    """Return angular error (deg) between predicted and GT quaternions."""
    pred_norm = F.normalize(pred, dim=1)
    target_norm = F.normalize(target, dim=1)
    dot = torch.sum(pred_norm * target_norm, dim=1).clamp(-1.0, 1.0).abs()
    angles = 2.0 * torch.acos(dot) * (180.0 / math.pi)
    return angles


def log_and_print(message, log_file):
    print(message)
    with open(log_file, "a", encoding="utf-8") as log_fp:
        log_fp.write(message + "\n")


def set_backbone_trainable(model, trainable):
    for param in model.backbone.parameters():
        param.requires_grad = trainable


def build_optimizer(model, lr, weight_decay, backbone_unfrozen):
    set_backbone_trainable(model, backbone_unfrozen)
    params = model.parameters() if backbone_unfrozen else filter(lambda p: p.requires_grad, model.parameters())
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer):
    return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=12)


def train():
    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    BATCH_SIZE = 32
    BASE_LR = 1e-4
    BACKBONE_LR_FACTOR = 0.1
    WEIGHT_DECAY = 1e-4
    # Split ufficiale 15/85: 2016 immagini di train, ~63 step/epoca contro i ~296 di prima.
    # Il tetto sale, a fermare il training è l'early stopping sulla validation.
    EPOCHS = int(os.environ.get("EPOCHS", "250"))
    EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "30"))
    FREEZE_EPOCHS = int(os.environ.get("FREEZE_EPOCHS", "40"))
    EXTRA_EPOCHS_ON_RESUME = 50  # epoche aggiuntive quando si riprende da checkpoint
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    LAMBDA_T = 1.0  # weight for translation MSE relative to rotation loss

    RESULTS_DIR = "results_3"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    SAVE_PATH_BEST = os.path.join(RESULTS_DIR, "pose_resnet50_baseline_best.pth")
    CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "pose_resnet50_baseline_checkpoint.pth")
    LOG_FILE = os.path.join(RESULTS_DIR, f"train_rgb_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    wandb.init(
        project="linemod-pose-estimation",
        name="PosePredictor-RGB",
        resume="allow",
        config={
            "learning_rate": BASE_LR,
            "architecture": "PosePredictor",
            "dataset": "LineMod_RGB",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "freeze_epochs": FREEZE_EPOCHS,
            "extra_epochs_on_resume": EXTRA_EPOCHS_ON_RESUME,
            "weight_decay": WEIGHT_DECAY,
            "lambda_t": LAMBDA_T,
        },
    )

    train_samples, val_samples, _, gt_cache = prepare_data_and_splits(ROOT_DATASET)
    train_set = LineModDataset(ROOT_DATASET, train_samples, gt_cache, augment=True)
    val_set = LineModDataset(ROOT_DATASET, val_samples, gt_cache, augment=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = PosePredictor().to(DEVICE)
    backbone_unfrozen = False

    optimizer = build_optimizer(model, BASE_LR, WEIGHT_DECAY, backbone_unfrozen)
    scheduler = build_scheduler(optimizer)

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    total_epochs = EPOCHS

    if os.path.exists(CHECKPOINT_PATH):
        log_and_print(f"Loading checkpoint from {CHECKPOINT_PATH}...", LOG_FILE)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        backbone_unfrozen = checkpoint.get("backbone_unfrozen", False)
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)

        optimizer = build_optimizer(
            model,
            BASE_LR * (BACKBONE_LR_FACTOR if backbone_unfrozen else 1.0),
            WEIGHT_DECAY,
            backbone_unfrozen,
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler = build_scheduler(optimizer)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        log_and_print(f"Resuming from epoch {start_epoch} (best val loss {best_val_loss:.6f}).", LOG_FILE)

        # Se vogliamo proseguire oltre, aggiungiamo EXTRA_EPOCHS_ON_RESUME al punto di ripartenza
        total_epochs = start_epoch + EXTRA_EPOCHS_ON_RESUME if EXTRA_EPOCHS_ON_RESUME > 0 else max(EPOCHS, start_epoch)

    else:
        total_epochs = EPOCHS

    for epoch in range(start_epoch, total_epochs):
        if epoch >= FREEZE_EPOCHS and not backbone_unfrozen:
            log_and_print("Unfreezing ResNet-50 backbone for fine-tuning...", LOG_FILE)
            backbone_unfrozen = True
            optimizer = build_optimizer(model, BASE_LR * BACKBONE_LR_FACTOR, WEIGHT_DECAY, backbone_unfrozen)
            scheduler = build_scheduler(optimizer)

        model.train()
        train_loss_total = 0.0
        train_deg_total = 0.0
        train_tmse_total = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [Train]")
        for batch in pbar:
            inputs   = batch["rgb"].to(DEVICE)
            gt_quat  = batch["quaternion"].to(DEVICE)
            gt_T_m   = batch["T"].to(DEVICE) / 1000.0  # mm → meters

            optimizer.zero_grad()
            pred_quat, pred_tvec = model(inputs)

            loss_rot   = rotation_loss(pred_quat, gt_quat)
            loss_trans = translation_loss(pred_tvec, gt_T_m)
            loss       = loss_rot + LAMBDA_T * loss_trans
            loss.backward()
            optimizer.step()

            batch_deg = quaternion_angle_error(pred_quat, gt_quat).mean().item()

            train_loss_total  += loss.item()
            train_deg_total   += batch_deg
            train_tmse_total  += loss_trans.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "deg": f"{batch_deg:.1f}", "t_mse": f"{loss_trans.item():.4f}"})

        avg_train_loss = train_loss_total  / len(train_loader)
        avg_train_deg  = train_deg_total   / len(train_loader)
        avg_train_tmse = train_tmse_total  / len(train_loader)

        model.eval()
        val_loss_total  = 0.0
        val_deg_total   = 0.0
        val_tmse_total  = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs   = batch["rgb"].to(DEVICE)
                gt_quat  = batch["quaternion"].to(DEVICE)
                gt_T_m   = batch["T"].to(DEVICE) / 1000.0

                pred_quat, pred_tvec = model(inputs)

                loss_rot   = rotation_loss(pred_quat, gt_quat)
                loss_trans = translation_loss(pred_tvec, gt_T_m)
                loss       = loss_rot + LAMBDA_T * loss_trans

                val_loss_total  += loss.item()
                val_deg_total   += quaternion_angle_error(pred_quat, gt_quat).mean().item()
                val_tmse_total  += loss_trans.item()

        avg_val_loss = val_loss_total  / len(val_loader)
        avg_val_deg  = val_deg_total   / len(val_loader)
        avg_val_tmse = val_tmse_total  / len(val_loader)

        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        log_and_print(
            f"Epoch {epoch + 1}/{total_epochs} | LR {current_lr:.2e} | Train Loss {avg_train_loss:.6f} | Val Loss {avg_val_loss:.6f}",
            LOG_FILE,
        )
        log_and_print(
            f" -> Train Deg {avg_train_deg:.2f}° | Val Deg {avg_val_deg:.2f}° | Train T-MSE {avg_train_tmse:.6f} | Val T-MSE {avg_val_tmse:.6f}",
            LOG_FILE,
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train/loss": avg_train_loss,
                "train/deg": avg_train_deg,
                "train/t_mse": avg_train_tmse,
                "val/loss": avg_val_loss,
                "val/deg": avg_val_deg,
                "val/t_mse": avg_val_tmse,
                "backbone_unfrozen": int(backbone_unfrozen),
            },
            step=epoch + 1,
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "backbone_unfrozen": backbone_unfrozen,
            "epochs_no_improve": epochs_no_improve,
        }
        torch.save(checkpoint_payload, CHECKPOINT_PATH)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH_BEST)
            wandb.save(SAVE_PATH_BEST)  # Upload .pth as wandb run file (coerente con phase4)
            wandb.run.summary["best_val_loss"] = best_val_loss
            log_and_print(f"New best model saved to {SAVE_PATH_BEST}.", LOG_FILE)
        else:
            epochs_no_improve += 1
            log_and_print(f"No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs.", LOG_FILE)

        # dopo lo sblocco del backbone il training riparte, non fermarsi sulle epoche congelate
        if backbone_unfrozen and epochs_no_improve >= EARLY_STOP_PATIENCE:
            log_and_print(
                f"Early stop at epoch {epoch+1}: no improvement on the best "
                f"({best_val_loss:.6f}) for {EARLY_STOP_PATIENCE} epochs.", LOG_FILE)
            break

    log_and_print("Training completed.", LOG_FILE)


if __name__ == "__main__":
    train()
    wandb.finish()