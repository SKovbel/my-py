import json
import ast  # Used to convert string representation of lists to actual lists
import numpy as np
from PIL import Image, ImageDraw

# Path to the CSV file
json_file_path = '/home/ksm/System/segmentation.json'  # Replace with the actual path to your CSV file

# Directory paths for images and masks
images_directory = '/home/ksm/System/'  # Adjust the path as needed
masks_directory = '/home/ksm/System/mask'  # Adjust the path as needed

with open(json_file_path, 'r') as json_file:
    segmentation_data = json.load(json_file)
# Read CSV file and process segmentation information
for image_filename, segmentation_info in segmentation_data['_via_img_metadata'].items():
    print(image_filename)
    print(segmentation_info)
    # Load image
    image_path = f'{images_directory}{segmentation_info["filename"]}'
    image = Image.open(image_path)

    # Create a blank mask
    mask = Image.new('L', image.size, 0)

    # Draw segments on the mask
    draw = ImageDraw.Draw(mask)
    for segment in segmentation_info['regions']:
        shape_attributes = segment['shape_attributes']
        if shape_attributes['name'] == 'polyline':
            all_points_x = shape_attributes['all_points_x']
            all_points_y = shape_attributes['all_points_y']
            
            if all_points_x and all_points_y and len(all_points_x) == len(all_points_y):
                polyline_coordinates = list(zip(all_points_x, all_points_y))
                draw.line(polyline_coordinates, fill=255, width=2)

    # Save the mask as an image
    mask_filename = f'{masks_directory}{image_filename.split(".")[0]}_mask.png'
    mask.save(mask_filename)

    # Optionally, you can also display the original image and the generated mask
    image.show()
    mask.show()