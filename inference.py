import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image
from Model import ResUNet3D

CHECKPOINT_PATH = "checkpoints/best_checkpoint.pth"
OUTPUT_DIR = "inference_results"

SLICE_AXIS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_input():
    root = Tk()
    root.withdraw()
    path = askopenfilename(
        title="Select one or more files",
        filetypes=[
            ("All supported", "*.npy *.pkl")
        ]
    )
    root.destroy()

    array = []
    if path.endswith(".npy"):
        array = np.load(path)
    elif path.endswith(".pkl"):
        obj = pickle.load(open(path, "rb"))
        if isinstance(obj, np.ndarray):
            array = obj
        else:
            raise ValueError(f"Unsupported pickle format, use numpy arrays or pngs.")
    return array.astype(np.float32)


def pad_to_multiple(volume, multiple=8):
    original_shape = volume.shape
    pad_amounts = []
    for dim in original_shape:
        remainder = dim % multiple
        if remainder == 0:
            pad = 0
        else:
            (multiple - remainder)
        pad_amounts.append(pad)

    pad_width_providing = []
    for i in pad_amounts:
        pad_width_providing.append((0,i))
    padded = np.pad(
        volume,
        pad_width=pad_width_providing,
        mode="reflect",
    )
    return padded, original_shape


def crop_to_shape(volume, target_shape):
    slices = tuple(slice(0, i) for i in target_shape)
    return volume[slices]


def run_inference(model, volume_np):

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


def save_slice_png(array, filename):
    img_uint8 = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(filename)


def main():
#    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = ResUNet3D(in_channels=1, out_channels=1, base_ch=16).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    raw_input = load_input()
    print(f"Loaded input with shape {raw_input.shape}")

    if raw_input.ndim == 2:
        deblurred = run_inference(model, raw_input)
        out_path = os.path.join(OUTPUT_DIR, "slice_deblurred.png")
        save_slice_png(deblurred, out_path)
        out_path = os.path.join(OUTPUT_DIR, "slice.png")
        save_slice_png(raw_input, out_path)
        print("Saved")
        return

    # ndim == 3: full volume
    deblurred_volume = run_inference(model, raw_input)
    num_slices = deblurred_volume.shape[SLICE_AXIS]

    for i in range(0,num_slices):
        slice_2d = np.take(deblurred_volume, i, axis=SLICE_AXIS)
        out_path = os.path.join(OUTPUT_DIR, f"slice_{i}_deblurred.png")
        save_slice_png(slice_2d, out_path)
        slice_2d = np.take(raw_input, i, axis=SLICE_AXIS)        
        out_path = os.path.join(OUTPUT_DIR, f"slice_{i}.png")
        save_slice_png(slice_2d, out_path)        
    print(f"Saved")

if __name__ == "__main__":
    main()
