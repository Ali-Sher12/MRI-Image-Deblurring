import pickle
import random
import os
from scipy.ndimage import gaussian_filter
import numpy as np
from PIL import Image

keep_fraction_either_side=0.6
max_shift = 0.25
sigma = 0.05 # for rican noise
sigmaG = 2 # for gaussian noise

ignore_val = 0

sigmaG_min = 0.5
sigmaG_max = 3
keep_fraction_either_side_min=0.3
keep_fraction_either_side_max=0.6
max_shift_min = 0.05
max_shift_max = 0.35
sigma_min = 0.03 # for rican noise
sigma_max = 0.09

save_as_volume = 0
save_as_slice = 0
save_images = 0

def save_as_png(array, filename):
    # Our data is floats in roughly [0, 1] - PNG needs integers in [0, 255]
    img_uint8 = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(filename)


def deletion_kspace(kspace):
    h, w = kspace.shape
    mask = np.zeros((h, w), dtype=bool) #all false by default
    keep_h, keep_w = int(h * keep_fraction_either_side / 2), int(w * keep_fraction_either_side / 2)
    mask[(h//2) - keep_h:(h//2) + keep_h, (w//2) - keep_w:(w//2) + keep_w] = True
    return kspace * mask

def add_motion_distortion(kspace):
    h, w = kspace.shape
    corrupted = kspace.astype(np.complex128).copy()
    for row in range(h):
        shift = np.random.uniform(-max_shift, max_shift)
        phase_ramp = np.exp(-2j * np.pi * shift * np.arange(w) / w)
        corrupted[row] *= phase_ramp
    return corrupted

def add_rician_noise(image):
    real_channel = image + np.random.normal(0, sigma, image.shape)
    imag_channel = np.random.normal(0, sigma, image.shape)
    noisy_image = np.sqrt(real_channel**2 + imag_channel**2)
    return noisy_image

def pkload(fname):
    return pickle.load(open(fname, 'rb'))


folder1 = "/home/jarvis/Documents/MRI Data/IXI_data/IXI_data/Test/"
folder2 = "/home/jarvis/Documents/MRI Data/IXI_data/IXI_data/Train/"
folder3 = "/home/jarvis/Documents/MRI Data/IXI_data/IXI_data/Val/"
folderList = [folder1,folder2,folder3]

save_as_volume = int(input("\nSave files as entire volumes(1 for yes, 0 for no) : "))
save_as_slice = int(input("\nSave files as slices(1 for yes, 0 for no) : "))
save_images = int(input("\nSave files as images(1 for yes, 0 for no) : "))

print("\n")
if __name__ == "__main__":
    for folderName in folderList:
        for filename in os.listdir(folderName):
            name, ext = os.path.splitext(filename)
            if ext!=".pkl":
                continue
            VoxelCluster,ignore_val = pkload(folderName +"/"+ filename)
            tamperedVoxelCluster = VoxelCluster.copy()
            slice_index = 0
            if save_as_volume == 1:    
                np.save(folderName +"/" + name + ".npy", VoxelCluster)
            for slice in VoxelCluster:
                if save_images == 1:
                    save_as_png(slice,folderName + "/" + name +"_"+ str(slice_index) + ".png")
                if save_as_slice == 1:    
                    np.save(folderName +"/" + name +"_"+ str(slice_index)+ ".npy", slice)

                max_shift = random.uniform(max_shift_min, max_shift_max)
                keep_fraction_either_side = random.uniform(keep_fraction_either_side_min,keep_fraction_either_side_max)
                sigma = random.uniform(sigma_min,sigma_max)
                sigmaG = random.uniform(sigmaG_min,sigmaG_max)              

                tamperedSlice = gaussian_filter(slice,sigmaG)
                k_space = np.fft.fftshift(np.fft.fft2(tamperedSlice))
                k_space = add_motion_distortion(k_space)
                k_space = deletion_kspace(k_space)
                tamperedSlice = np.abs(np.fft.ifft2(np.fft.ifftshift(k_space)))
                tamperedSlice = add_rician_noise(tamperedSlice)

                tamperedVoxelCluster[slice_index] = tamperedSlice

                if save_as_slice == 1:
                    np.save(folderName + "/" + name + "_blurred_"+ str(slice_index) + ".npy", tamperedSlice)
                if save_images == 1:
                    save_as_png(tamperedSlice,folderName + "/" + name + "_blurred_"+ str(slice_index) + ".png")

                slice_index+=1
            if save_as_volume == 1:    
                np.save(folderName +"/blurred_" + name + ".npy", tamperedVoxelCluster)
