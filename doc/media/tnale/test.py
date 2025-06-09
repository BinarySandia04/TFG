import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image

def convert(name):
    # Load the image (as RGB)
    image = mpimg.imread(name)  # Replace with your image path
    
    # If image is RGBA, drop the alpha channel
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    
    # Convert RGB to grayscale using luminosity method
    grayscale = image @ [0.2989, 0.5870, 0.1140]  # Weighted average for human perception
    
    # Normalize to 0–255 and convert to uint8 for image representation
    grayscale_uint8 = (grayscale * 255).clip(0, 255).astype(np.uint8)
    
    # Convert to PIL Image to display/save
    grayscale_image = Image.fromarray(grayscale_uint8)
    grayscale_image.save('AAA' + name)


png_files = [f for f in os.listdir('.') if f.endswith('.png')]
for p in png_files:
    convert(p)
