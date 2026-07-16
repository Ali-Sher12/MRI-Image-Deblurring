import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from Model import ResUNet3D

# ----------------------------------------------------------------------
# Global config
# ----------------------------------------------------------------------
INPUT_PATH = "/content/some_volume.npy"     # .npy or .pkl file
CHECKPOINT_PATH = "/content/best_checkpoint.pth"
OUTPUT_DIR = "/content/inference_results"

# Which slice(s) to save as PNG:
#   "middle" -> just the single middle slice
#   "all"    -> every slice along SLICE_AXIS
#   [10, 40, 90] -> a specific list of slice indices
SLICE_SELECTION = "middle"
SLICE_AXIS = 0   # which axis of the volume represents "slices" (0 = D in (D,H,W))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_input(path):
    """
    Loads either a .npy or .pkl file and returns a numpy array.
    Handles: a 3D volume (D,H,W), a 2D single slice (H,W), or a pickled
    dict/list that wraps one of these.
    """
    if path.endswith(".npy"):
        array = np.load(path)
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, np.ndarray):
            array = obj
        elif isinstance(obj, dict):
            # try common key names
            for key in ["volume", "data", "image", "slices"]:
                if key in obj:
                    array = np.asarray(obj[key])
                    break
            else:
                raise ValueError(
                    f"Pickled dict has keys {list(obj.keys())}; none matched "
                    f"expected names (volume/data/image/slices). Adjust load_input()."
                )
        elif isinstance(obj, (list, tuple)):
            array = np.stack(obj, axis=0)   # list of 2D slices -> (D, H, W)
        else:
            raise ValueError(f"Unsupported pickled object type: {type(obj)}")
    else:
        raise ValueError("INPUT_PATH must end in .npy or .pkl")

    return array.astype(np.float32)


def pad_to_multiple(volume, multiple=8):
    """
    Pads a (D,H,W) volume so each dimension is divisible by `multiple`.
    Returns the padded volume and the original shape (for cropping back later).
    """
    original_shape = volume.shape
    pad_amounts = []
    for dim in original_shape:
        remainder = dim % multiple
        pad = 0 if remainder == 0 else (multiple - remainder)
        pad_amounts.append(pad)

    padded = np.pad(
        volume,
        pad_width=[(0, p) for p in pad_amounts],
        mode="reflect",
    )
    return padded, original_shape


def crop_to_shape(volume, target_shape):
    slices = tuple(slice(0, s) for s in target_shape)
    return volume[slices]


def run_inference(model, volume_np):
    """
    volume_np: (D, H, W) numpy array.
    Returns the deblurred (D, H, W) numpy array, same size as input.
    """
    original_ndim = volume_np.ndim

    if original_ndim == 2:
        # Single 2D slice: replicate along depth so the 3D model has
        # something to convolve/pool across. Quality caveat: this is
        # not real 3D context.
        fake_depth = 8
        volume_np = np.repeat(volume_np[np.newaxis, :, :], fake_depth, axis=0)
        was_2d = True
    else:
        was_2d = False

    padded_volume, original_shape = pad_to_multiple(volume_np, multiple=8)

    tensor = torch.from_numpy(padded_volume).float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        output = model(tensor)

    output_np = output.squeeze(0).squeeze(0).cpu().numpy()
    output_np = crop_to_shape(output_np, original_shape)

    if was_2d:
        mid = output_np.shape[0] // 2
        output_np = output_np[mid]   # back to a single 2D slice

    return output_np


def save_slice_png(slice_2d, out_path):
    plt.figure(figsize=(5, 5))
    plt.imshow(slice_2d, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = ResUNet3D(in_channels=1, out_channels=1, base_ch=16).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    raw_input = load_input(INPUT_PATH)
    print(f"Loaded input with shape {raw_input.shape}")

    if raw_input.ndim == 2:
        deblurred = run_inference(model, raw_input)
        out_path = os.path.join(OUTPUT_DIR, "deblurred_slice.png")
        save_slice_png(deblurred, out_path)
        print(f"Saved: {out_path}")
        return

    # ndim == 3: full volume
    deblurred_volume = run_inference(model, raw_input)

    num_slices = deblurred_volume.shape[SLICE_AXIS]

    if SLICE_SELECTION == "middle":
        indices = [num_slices // 2]
    elif SLICE_SELECTION == "all":
        indices = list(range(num_slices))
    else:
        indices = SLICE_SELECTION  # assume a list of ints was given

    for idx in indices:
        slice_2d = np.take(deblurred_volume, idx, axis=SLICE_AXIS)
        out_path = os.path.join(OUTPUT_DIR, f"deblurred_slice_{idx}.png")
        save_slice_png(slice_2d, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
