import PIL
from PIL import Image, ImageOps
import requests
import os
import sys
import csv
import matplotlib.pyplot as plt
from typing import Union

def load_image(image: Union[str, PIL.Image.Image])-> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    try:
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                image = PIL.Image.open(requests.get(image, stream=True).raw)
            elif os.path.isfile(image):
                image = PIL.Image.open(image)
            else:
                raise ValueError(
                    f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
                )
        elif isinstance(image, PIL.Image.Image):
            image = image
        else:
            raise ValueError(
                "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
            )
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image
    
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)


def load_image_ratio(image: Union[str, Image.Image]) -> Image.Image:
    """
    Loads `image` as a PIL Image and resizes it to a width of 512 pixels while maintaining the aspect ratio.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.

    Returns:
        `PIL.Image.Image`:
            A PIL Image resized to a width of 512 pixels, maintaining the aspect ratio.
    """
    try:
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                image = Image.open(requests.get(image, stream=True).raw)
            elif os.path.isfile(image):
                image = Image.open(image)
            else:
                raise ValueError(
                    f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
                )
        elif isinstance(image, Image.Image):
            image = image
        else:
            raise ValueError(
                "Incorrect format used for image. Should be a URL linking to an image, a local path, or a PIL Image."
            )
        
        # Correct image orientation if needed
        image = ImageOps.exif_transpose(image)
        
        # Convert the image to RGB
        image = image.convert("RGB")
        
        # Resize the image to a width of 512 pixels, maintaining the aspect ratio
        target_width = 512
        aspect_ratio = image.height / image.width
        new_height = int(target_width * aspect_ratio)
        image = image.resize((target_width, new_height), Image.Resampling.LANCZOS)
        
        return image

    except Exception as e:
        raise ValueError(f"An error occurred while loading the image: {str(e)}")



def image_grid(images, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 18))
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img)
        ax.axis('off')
    plt.show()


def show_image_grid_parameters(dict_list, image_key='image', title_keys=None, grid_size=(4, 8)):
    """
    Displays a grid of images with titles corresponding to other parameters in the list of dictionaries.

    Parameters:
    - dict_list (list of dict): List of dictionaries, each containing an image and other parameters.
    - image_key (str): The key in the dictionary where the image is stored.
    - title_keys (list of str): Keys for the parameters to display as the title of each image.
    - grid_size (tuple): Number of rows and columns for the grid (rows, columns).

    Returns:
    - None: Displays the grid of images with titles.
    """
    # Calculate grid size based on the number of images
    rows, cols = grid_size
    num_images = len(dict_list)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()

    for i, data in enumerate(dict_list):
        if i >= rows * cols:
            break  # Only display the number of images that fit in the grid
        
        image = data[image_key]

        axes[i].imshow(image)
        axes[i].axis('off')
        
        # Create title from the specified title_keys
        if title_keys:
            title = "\n".join(f"{key}: {data[key]}" for key in title_keys)
            axes[i].set_title(title, fontsize=8)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def show_image_grid(dict_list, image_key='image', title_keys=None, grid_size=(4, 6)):
    """
    Displays a grid of images with titles corresponding to other parameters in the list of dictionaries.

    Parameters:
    - dict_list (list of dict): List of dictionaries, each containing an image and other parameters.
    - image_key (str): The key in the dictionary where the image is stored.
    - title_keys (list of str): Keys for the parameters to display as the title of each image.
    - grid_size (tuple): Number of rows and columns for the grid (rows, columns).

    Returns:
    - None: Displays the grid of images with titles.
    """
    # Calculate grid size based on the number of images
    rows, cols = grid_size
    num_images = len(dict_list)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    for i, data in enumerate(dict_list):
        if i >= rows * cols:
            break  # Only display the number of images that fit in the grid
        
        image = data[image_key]

        axes[i].imshow(image)
        axes[i].axis('off')
        
        # Create title from the specified title_keys
        if title_keys:
            title = "\n".join(f"{key}: {data[key]}" for key in title_keys)
            axes[i].set_title(title, fontsize=8)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def write_results_csv(data: dict, results_csv: str):
    """ Write data dictionary to a CSV file """

    file_exists = os.path.isfile(results_csv)
    
    with open(results_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())

        if not file_exists: # Write header only if file is being created
            writer.writeheader()
        writer.writerow(data) # Write data as a row in the CSV

def print_bold(text):
    """Print text in bold"""
    print(f"\033[1m{text}\033[0m")