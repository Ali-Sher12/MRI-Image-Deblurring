import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Data_loader import DataSet
from Model import ResUNet3D

# ----------------------------------------------------------------------
# Global config
# ----------------------------------------------------------------------
CHUNK_SIZE = 64
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

TRAIN_FOLDER = "/content/train_data/Train"
VAL_FOLDER = "/content/train_data/Val"
TEST_FOLDER = "/content/train_data/Test"

# Save checkpoints to the mounted Drive (train+val account), not just local
# disk, so a Colab disconnect doesn't wipe out your progress.
CHECKPOINT_DIR = "/content/drive/MyDrive/checkpoints"
BEST_CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/best_checkpoint.pth"
LATEST_CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/latest_checkpoint.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_crop_batch(blurred_batch, sharp_batch, chunk_size):
    B, C, D, H, W = blurred_batch.shape

    if D < chunk_size or H < chunk_size or W < chunk_size:
        raise ValueError(
            f"CHUNK_SIZE={chunk_size} is larger than volume dims {(D, H, W)}"
        )

    cropped_blurred = torch.empty((B, C, chunk_size, chunk_size, chunk_size))
    cropped_sharp = torch.empty((B, C, chunk_size, chunk_size, chunk_size))

    for i in range(B):
        d_start = torch.randint(0, D - chunk_size + 1, (1,)).item()
        h_start = torch.randint(0, H - chunk_size + 1, (1,)).item()
        w_start = torch.randint(0, W - chunk_size + 1, (1,)).item()

        cropped_blurred[i] = blurred_batch[
            i, :,
            d_start:d_start + chunk_size,
            h_start:h_start + chunk_size,
            w_start:w_start + chunk_size,
        ]
        cropped_sharp[i] = sharp_batch[
            i, :,
            d_start:d_start + chunk_size,
            h_start:h_start + chunk_size,
            w_start:w_start + chunk_size,
        ]

    return cropped_blurred, cropped_sharp


def run_validation(model, val_dataloader, loss_fn):
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for blurred_vol, sharp_vol in val_dataloader:
            blurred_patch, sharp_patch = random_crop_batch(
                blurred_vol, sharp_vol, CHUNK_SIZE
            )
            blurred_patch = blurred_patch.to(DEVICE, non_blocking=True)
            sharp_patch = sharp_patch.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast():
                predicted_patch = model(blurred_patch)
                loss = loss_fn(predicted_patch, sharp_patch)

            running_val_loss += loss.item()

    return running_val_loss / len(val_dataloader)


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    train_dataset = DataSet(TRAIN_FOLDER)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_dataset = DataSet(VAL_FOLDER)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,   # no need to shuffle validation data
        num_workers=2,
        pin_memory=True,
    )

    model = ResUNet3D(in_channels=1, out_channels=1, base_ch=16).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_train_loss = 0.0

        for blurred_vol, sharp_vol in train_dataloader:
            blurred_patch, sharp_patch = random_crop_batch(
                blurred_vol, sharp_vol, CHUNK_SIZE
            )
            blurred_patch = blurred_patch.to(DEVICE, non_blocking=True)
            sharp_patch = sharp_patch.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                predicted_patch = model(blurred_patch)
                loss = loss_fn(predicted_patch, sharp_patch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_dataloader)
        avg_val_loss = run_validation(model, val_dataloader, loss_fn)

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"- Train Loss: {avg_train_loss:.6f} "
            f"- Val Loss: {avg_val_loss:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }

        # Always keep the latest epoch (so you can resume after a disconnect)
        torch.save(checkpoint, LATEST_CHECKPOINT_PATH)

        # Separately keep the best-performing epoch on validation data
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, BEST_CHECKPOINT_PATH)
            print(f"  -> New best val loss ({best_val_loss:.6f}), checkpoint saved.")


if __name__ == "__main__":
    train()
