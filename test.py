import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import matplotlib.pyplot as plt

from Data_loader import DataSet
from Model import ResUNet3D

# ----------------------------------------------------------------------
# Global config
# ----------------------------------------------------------------------
TEST_FOLDER = "/content/train_data/Test"       # match your actual folder name/case
CHECKPOINT_PATH = "/content/best_checkpoint.pth"  # local copy from step 2

TRAIN_FOLDER = "/content/train_data/Train"
VAL_FOLDER = "/content/train_data/Val"
TEST_FOLDER = "/content/train_data/Test"
#CHECKPOINT_PATH = "/content/drive/MyDrive/checkpoints/best_checkpoint.pth"
RESULTS_DIR = "/content/test_results"
NUM_VISUAL_SAMPLES = 5   # how many subjects to save slice images for

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_numpy_2d_slice(volume_tensor):
    """
    volume_tensor shape: (1, 1, D, H, W) -> take the middle axial slice
    and return a plain 2D numpy array for plotting/SSIM/PSNR.
    """
    volume = volume_tensor.squeeze(0).squeeze(0).cpu().numpy()  # (D, H, W)
    mid_slice_idx = volume.shape[0] // 2
    return volume[mid_slice_idx]


def save_comparison_image(blurred_slice, predicted_slice, sharp_slice, subject_idx):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Blurred (input)", "Deblurred (predicted)", "Sharp (ground truth)"]
    slices = [blurred_slice, predicted_slice, sharp_slice]

    for ax, title, sl in zip(axes, titles, slices):
        ax.imshow(sl, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"subject_{subject_idx}_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def test():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    test_dataset = DataSet(TEST_FOLDER)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,     # whole volumes, one subject at a time
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = ResUNet3D(in_channels=1, out_channels=1, base_ch=16).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val loss at save time: {checkpoint['val_loss']:.6f})")

    l1_scores, psnr_scores, ssim_scores = [], [], []

    with torch.no_grad():
        for subject_idx, (blurred_vol, sharp_vol) in enumerate(test_dataloader):
            blurred_vol = blurred_vol.to(DEVICE)
            sharp_vol = sharp_vol.to(DEVICE)

            # NOTE: if this OOMs on a large volume, see the sliding-window
            # fallback note below the script.
            predicted_vol = model(blurred_vol)

            # --- Metrics (computed on full 3D volume) ---
            l1 = torch.mean(torch.abs(predicted_vol - sharp_vol)).item()

            pred_np = predicted_vol.squeeze().cpu().numpy()
            sharp_np = sharp_vol.squeeze().cpu().numpy()

            data_range = sharp_np.max() - sharp_np.min()
            psnr = compute_psnr(sharp_np, pred_np, data_range=data_range)
            ssim = compute_ssim(sharp_np, pred_np, data_range=data_range)

            l1_scores.append(l1)
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

            print(f"Subject {subject_idx}: L1={l1:.6f}  PSNR={psnr:.2f}dB  SSIM={ssim:.4f}")

            # --- Save a visual comparison for the first few subjects ---
            if subject_idx < NUM_VISUAL_SAMPLES:
                blurred_slice = to_numpy_2d_slice(blurred_vol)
                predicted_slice = to_numpy_2d_slice(predicted_vol)
                sharp_slice = to_numpy_2d_slice(sharp_vol)
                save_comparison_image(blurred_slice, predicted_slice, sharp_slice, subject_idx)

    print("\n--- Overall Test Set Results ---")
    print(f"Mean L1:   {np.mean(l1_scores):.6f}")
    print(f"Mean PSNR: {np.mean(psnr_scores):.2f} dB")
    print(f"Mean SSIM: {np.mean(ssim_scores):.4f}")

    # Save metrics to a text file for your records
    with open(os.path.join(RESULTS_DIR, "metrics_summary.txt"), "w") as f:
        f.write(f"Mean L1:   {np.mean(l1_scores):.6f}\n")
        f.write(f"Mean PSNR: {np.mean(psnr_scores):.2f} dB\n")
        f.write(f"Mean SSIM: {np.mean(ssim_scores):.4f}\n")


if __name__ == "__main__":
    test()
