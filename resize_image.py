from PIL import Image
import numpy as np


# resizes image below given dimension while retaining original ratio
def resize(image_path, max_size, save_path):
    original_image = Image.open(image_path)
    image_width, image_height = original_image.size
    resize_scale = max(np.ceil(image_width / max_size), np.ceil(image_height / max_size))
    resized_image = original_image.resize((int(image_width / resize_scale), int(image_height / resize_scale)))
    resized_image.save(save_path, icc_profile=None)


if __name__ == '__main__':
    resize(image_path='Images/flower.png', max_size=1000, save_path='Images/flower.png')
