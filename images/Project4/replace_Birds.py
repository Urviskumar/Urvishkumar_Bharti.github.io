# -*- coding: utf-8 -*-
"""Replace_Birds.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LUkYYnUWRiGTqOunneYl1gywAVtI0eDH
"""

!pip install diffusers
!pip install transformers

from PIL import Image
import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diffusers import StableDiffusionInpaintPipeline as SDIP

def load_inpainting_model():
    """
    Load the pre-trained inpainting model for stable diffusion inpainting.

    Returns:
        inpainting_model: Loaded inpainting model.
    """
    inpainting_model = SDIP.from_pretrained("runwayml/stable-diffusion-inpainting")
    return inpainting_model

def apply_inpainting(image_files, mask_files, inpainting_model):
    """
    Apply inpainting on each image using the provided mask.

    Args:
        image_files (list): List of image filenames.
        mask_files (list): List of mask filenames corresponding to the images.
        inpainting_model: Loaded inpainting model for stable diffusion inpainting.
    """
    for i in range(len(image_files)):
        original_image_file = image_files[i]
        mask_file = mask_files[i]
        initial_image = Image.open(original_image_file).resize((512, 512))
        mask_image = Image.open(mask_file).resize((512, 512))
        prompt_text = 'Replace bird, high resolution'
        inpainted_image = inpainting_model(prompt=prompt_text, image=initial_image, mask_image=mask_image).images[0]

        # Save the inpainted image with a new filename
        result_filename = f"birds_Replaced{i+2}.jpg"
        inpainted_image.save(result_filename)

def main():
    # Define the paths to the image files and their corresponding masks
    image_files = [
        "birds1.jpg",
        "birds2.jpg",
        "birds3.jpg",
        "birds4.jpg",
        "birds5.jpg",
    ]
    mask_files = [
        "mask_birds1.jpg",
        "mask_birds2.jpg",
        "mask_birds3.jpg",
        "mask_birds4.jpg",
        "mask_birds5.jpg",
    ]

    # Load the pre-trained inpainting model
    inpainting_model = load_inpainting_model()

    # Apply inpainting on each image
    apply_inpainting(image_files, mask_files, inpainting_model)

# Call the main function
if __name__ == "__main__":
    main()

