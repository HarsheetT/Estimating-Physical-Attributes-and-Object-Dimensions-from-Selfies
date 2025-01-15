import torch
import os
import argparse
from PIL import Image

def detect_objects(image_path, save_path='results/output.png'):
    # Ensure the save directory exists
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Perform detection
    results = model(image_path)

    # Extract the rendered image from the results
    rendered_image = results.render()[0]  # Get the first image (as numpy array)

    # Save the result to the specified save_path
    rendered_pil_image = Image.fromarray(rendered_image)
    rendered_pil_image.save(save_path)

    print(f"Detection complete. Results saved in: {save_path}")

# Setup command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects in an image using YOLOv5.")
    parser.add_argument('image_path', type=str, help="Path to the input image.")
    parser.add_argument('--save_path', type=str, default='results/output.png', help="Path to save the output image.")
    
    args = parser.parse_args()
    
    # Call the detect_objects function with the arguments
    detect_objects(args.image_path, save_path=args.save_path)
