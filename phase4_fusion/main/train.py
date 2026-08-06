import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import os
from datetime import datetime
import wandb

from phase4_fusion.main.model import RGBD_FusionPredictor
from phase4_fusion.main.dataset import LineModDatasetRGBD
from phase4_fusion.main.add_loss import ADDLoss
from common.data_split import prepare_data_and_splits
from common.gpu_augment import GPUAugmentation

def log_and_print(message, log_file):
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def load_info_cache(dataset_root, object_ids):
    info_cache = {}
    for obj_id in object_ids:
        obj_folder = f"{obj_id:02d}"
        info_path = os.path.join(dataset_root, 'data', obj_folder, 'info.yml')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info_cache[obj_id] = yaml.safe_load(f)
    return info_cache

def train():
    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
    LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))
    # Con lo split ufficiale 15/85 il train set è 2016 immagini: ~32 step/epoca contro i ~148
    # di prima, quindi il tetto di epoche sale e a decidere quando fermarsi è la validation.
    EPOCHS = int(os.environ.get("EPOCHS", "400"))
    EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "30"))
    # raw = baseline del paper · norm = depth ancorata (A) · xyz = mappa XYZ (B)
    DEPTH_MODE = os.environ.get("DEPTH_MODE", "raw")
    # Multi-seed: SEED fissa il caso, RUN_TAG separa checkpoint e run wandb — senza tag
    # un secondo run dello stesso modo farebbe resume del checkpoint del primo.
    SEED = int(os.environ.get("SEED", "42"))
    RUN_TAG = os.environ.get("RUN_TAG", "")
    import random as _random
    import numpy as _np
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    _np.random.seed(SEED); _random.seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    AMP_ENABLED = (DEVICE.type == "cuda")
    AMP_DTYPE = torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
    
    RESULTS_DIR = "results_4_main"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Per-mode files: the raw/norm/xyz ablation runs must not overwrite each other.
    SAVE_PATH_BEST = os.path.join(RESULTS_DIR, f"pose_rgbd_fusion_best_{DEPTH_MODE}{RUN_TAG}.pth")
    CHECKPOINT_PATH = os.path.join(RESULTS_DIR, f"pose_rgbd_checkpoint_{DEPTH_MODE}{RUN_TAG}.pth")
    LOG_FILE = os.path.join(RESULTS_DIR, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    N_POINTS = 500

    wandb.init(
        project="linemod-pose-estimation",
        name=f"RGBD_official15_{DEPTH_MODE}{RUN_TAG}",
        resume="allow",
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "RGBD_FusionPredictor",
            "dataset": "LineMod_RGBD_official15",
            "depth_mode": DEPTH_MODE,
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "weight_decay": 1e-4,
            "n_points": N_POINTS,
        }
    )

    train_samples, val_samples, _, gt_cache = prepare_data_and_splits(ROOT_DATASET)
    object_ids = sorted(gt_cache.keys())
    info_cache = load_info_cache(ROOT_DATASET, object_ids)
    
    train_set = LineModDatasetRGBD(ROOT_DATASET, train_samples, gt_cache, info_cache, n_points=N_POINTS, is_train=True,
                                   depth_mode=DEPTH_MODE)
    val_set = LineModDatasetRGBD(ROOT_DATASET, val_samples, gt_cache, info_cache, n_points=N_POINTS, is_train=False,
                                 depth_mode=DEPTH_MODE)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    model = RGBD_FusionPredictor().to(DEVICE)
    criterion = ADDLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    gpu_aug = GPUAugmentation().to(DEVICE)
    # bf16 doesn't need GradScaler (no overflow); fp16 does.
    scaler = GradScaler(DEVICE.type, enabled=(AMP_ENABLED and AMP_DTYPE == torch.float16))

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=12)

    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    if os.path.exists(CHECKPOINT_PATH):
        log_and_print(f"Loading checkpoint from {CHECKPOINT_PATH}...", LOG_FILE)
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        log_and_print(f"Resuming from epoch {start_epoch}", LOG_FILE)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss_total = 0.0
        train_trans_mse = 0.0
        train_rot_mse = 0.0
        
        # Recupero del Learning Rate corrente per il log
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [LR: {current_lr:.2e}]")
        for batch in pbar:
            rgb_u8 = batch["rgb"].to(DEVICE, non_blocking=True)
            depth_1ch = batch["depth"].to(DEVICE, non_blocking=True)
            meta = batch["meta_info"].to(DEVICE, non_blocking=True)
            gt_R = batch["R_matrix"].to(DEVICE, non_blocking=True)
            gt_T = batch["translation_3d"].to(DEVICE, non_blocking=True)
            model_points = batch["model_points"].to(DEVICE, non_blocking=True)
            t_anchor = batch["t_anchor"].to(DEVICE, non_blocking=True)

            rgb = gpu_aug(rgb_u8, training=True)
            depth = depth_1ch.expand(-1, 3, -1, -1)

            optimizer.zero_grad()
            with autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                pred_T, pred_R = model(rgb, depth, meta)
                # la testa regredisce il residuo rispetto all'ancora 3D (zero in raw)
                pred_T = pred_T + t_anchor
                batch_loss = criterion(pred_R, pred_T, gt_R, gt_T, model_points)

            with torch.no_grad():
                train_trans_mse += F.mse_loss(pred_T.float(), gt_T).item()
                train_rot_mse += F.mse_loss(pred_R.float(), gt_R).item()

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += batch_loss.item()
            pbar.set_postfix({'ADD': batch_loss.item()})

        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_t_mse = train_trans_mse / len(train_loader)
        avg_train_r_mse = train_rot_mse / len(train_loader)

        # Log metriche di training
        wandb.log({
            "train/epoch": epoch + 1,
            "train/loss_add": avg_train_loss,
            "train/t_mse": avg_train_t_mse,
            "train/r_mse": avg_train_r_mse,
            "train/lr": current_lr
        })

        # --- VALIDAZIONE ---
        model.eval()
        val_loss_total = 0.0
        val_trans_mse = 0.0
        val_rot_mse = 0.0
        obj_errors = {obj_id: 0.0 for obj_id in object_ids}
        obj_counts = {obj_id: 0 for obj_id in object_ids}

        with torch.no_grad():
            for batch in val_loader:
                rgb_u8 = batch["rgb"].to(DEVICE, non_blocking=True)
                depth_1ch = batch["depth"].to(DEVICE, non_blocking=True)
                meta = batch["meta_info"].to(DEVICE, non_blocking=True)
                gt_R = batch["R_matrix"].to(DEVICE, non_blocking=True)
                gt_T = batch["translation_3d"].to(DEVICE, non_blocking=True)
                model_points, ids = batch["model_points"].to(DEVICE, non_blocking=True), batch["obj_id"]
                t_anchor = batch["t_anchor"].to(DEVICE, non_blocking=True)

                rgb = gpu_aug(rgb_u8, training=False)
                depth = depth_1ch.expand(-1, 3, -1, -1)

                with autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                    pred_T, pred_R = model(rgb, depth, meta)
                pred_T = pred_T.float() + t_anchor
                pred_R = pred_R.float()

                val_trans_mse += F.mse_loss(pred_T, gt_T).item()
                val_rot_mse += F.mse_loss(pred_R, gt_R).item()

                # Un solo forward per batch invece di uno per campione: la validation
                # dominava il tempo di epoca (14s su 18) girando la loss 64 volte in
                # Python. Il valore per campione e' identico a prima.
                sample_losses = criterion(
                    pred_R, pred_T, gt_R, gt_T, model_points, per_sample=True
                )
                val_loss_total += sample_losses.sum().item()
                for i, sample_loss in enumerate(sample_losses.tolist()):
                    curr_id = ids[i].item()
                    obj_errors[curr_id] += sample_loss
                    obj_counts[curr_id] += 1

        avg_val_loss = val_loss_total / len(val_set)
        avg_val_t_mse = val_trans_mse / len(val_loader)
        avg_val_r_mse = val_rot_mse / len(val_loader)
        
        scheduler.step(avg_val_loss)

        # --- REPORTISTICA FINALE CON LR ---
        log_and_print(f"\n--- Epoch {epoch+1} Summary ---", LOG_FILE)
        log_and_print(f"Learning Rate: {current_lr:.2e}", LOG_FILE) # Riga richiesta
        log_and_print(f"Avg Train ADD: {avg_train_loss:.6f} m | T-MSE: {avg_train_t_mse:.6f} | R-MSE: {avg_train_r_mse:.6f}", LOG_FILE)
        log_and_print(f"Avg Val ADD:   {avg_val_loss:.6f} m | T-MSE: {avg_val_t_mse:.6f} | R-MSE: {avg_val_r_mse:.6f}", LOG_FILE)
        

        # Log metriche di validazione
        val_metrics = {
            "val/loss_add": avg_val_loss,
            "val/t_mse": avg_val_t_mse,
            "val/r_mse": avg_val_r_mse,
        }


        for obj_id in object_ids:
            if obj_counts[obj_id] > 0:
                err_mm = (obj_errors[obj_id] / obj_counts[obj_id]) * 1000
                val_metrics[f"val_obj/error_mm_{obj_id:02d}"] = err_mm
                log_and_print(f" Object {obj_id:02d}: {err_mm:.2f} mm", LOG_FILE)

        wandb.log(val_metrics)

        # Salvataggio checkpoint e best model
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'epochs_no_improve': epochs_no_improve
        }
        torch.save(checkpoint_data, CHECKPOINT_PATH)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            wandb.save(SAVE_PATH_BEST) # Carica il file .pth su wandb
            torch.save(model.state_dict(), SAVE_PATH_BEST)
            log_and_print(f"⭐ NEW BEST! Model saved to {SAVE_PATH_BEST}", LOG_FILE)
        else:
            epochs_no_improve += 1
            log_and_print(f"No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs", LOG_FILE)

        log_and_print("-" * 40, LOG_FILE)

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            log_and_print(
                f"Early stop at epoch {epoch+1}: val ADD has not improved on the best "
                f"({best_val_loss:.6f} m) for {EARLY_STOP_PATIENCE} epochs.", LOG_FILE)
            break

if __name__ == "__main__":
    train()
    wandb.finish()