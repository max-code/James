import cv2
from rembg import remove
from PIL import Image
import numpy as np
from typing import Optional, Tuple
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(funcName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        """
        Initialize the ImageProcessor class by loading the super-resolution model.
        """
        logging.info(f"Loading model")
        self.upscaler = cv2.dnn_superres.DnnSuperResImpl_create()
        self.upscaler.readModel("ESPCN_x4.pb")
        self.upscaler.setModel("espcn", 4)
        self.image_width = 1000
        self.image_height = 1000
        self.object_padding = 50

    def convert_to_png(self, input_path: str, output_path: Optional[str] = None) -> bool:
        """
        Convert an image to PNG format with an alpha channel.

        :param input_path: Path to the input image.
        :param output_path: Path to save the converted image. Must have .png extension
        :return: Whether or not it could be converted to a PNG.
        """
        
        logging.info(f"Converting image to png. input_path={input_path} output_path={output_path}")
        
        if output_path is None:
            logging.warning(f"Output path not set. Setting it to input path ({input_path})")
            output_path = input_path
            
        _, extension = os.path.splitext(output_path)
        if extension != ".png":
            logging.warning(f"Couldnt convert file to PNG. output_path ({output_path}) requires .png extension")
            return False

        image = Image.open(input_path).convert("RGBA")
        image.save(output_path, "PNG")
        
        return True

    def remove_background(self, input_path: str, output_path: Optional[str] = None) -> None:
        """
        Remove the background from an image.

        :param input_path: Path to the input image.
        :param output_path: Path to save the image with the background removed.
        """
        
        logging.info(f"Removing image background. input_path={input_path} output_path={output_path}")
        
        if output_path is None:
            logging.warning(f"Output path not set. Setting it to input path ({input_path})")
            output_path = input_path
            
        # Open the input image file
        with open(input_path, "rb") as input_file:
            input_data = input_file.read()

        # Use rembg to remove the background
        output_data = remove(input_data)

        # Save the output data
        with open(output_path, "wb") as output_file:
            output_file.write(output_data)

    def upscale(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Upscale an image using a pre-trained super-resolution model.

        :param input_path: Path to the input image.
        :param output_path: Path to save the upscaled image.
        """
        
        logging.info(f"Upscaling image. input_path={input_path} output_path={output_path}")
        
        if output_path is None:
            logging.warning(f"Output path not set. Setting it to input path ({input_path})")
            output_path = input_path
        
        # Read the image with alpha channel
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError("Input image file not found.")
        
        # Separate the color and alpha channels
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]

        # Upscale the color channels
        upscaled_bgr = self.upscaler.upsample(bgr)
        
        # Resize the alpha channel to match the upscaled image
        upscaled_alpha = cv2.resize(alpha, (upscaled_bgr.shape[1], upscaled_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Merge the upscaled color channels with the resized alpha channel
        upscaled_image = cv2.merge((upscaled_bgr, upscaled_alpha))
        
        # Save the upscaled image
        cv2.imwrite(output_path, upscaled_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def overlay(self, input_path: str, output_path: Optional[str] = None) -> None:
        """
        Overlay an upscaled image on a white background.

        :param input_path: Path to the upscaled image.
        :param output_path: Path to save the final image.
        """
        logging.info(f"Overlaying object onto {self.image_width}x{self.image_height}px background. input_path={input_path} output_path={output_path}")
        
        if output_path is None:
            logging.warning(f"Output path not set. Setting it to input path ({input_path})")
            output_path = input_path
        
        # Read the upscaled image
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError("Upscaled image file not found.")
        
        # Get bounding box of the object in the transparent image
        x, y, w, h = self.__get_bounding_box(image)

        # Extract the object using the bounding box
        object_img = image[y:y+h, x:x+w]

        # Create a white background with alpha channel
        bg = np.full((self.image_width, self.image_height, 4), 255, dtype=np.uint8)

        # Calculate the desired size maintaining aspect ratio
        max_width = self.image_width - 2 * self.object_padding
        max_height = self.image_height - 2 * self.object_padding
        aspect_ratio = w / h

        if w > h:
            new_w = min(max_width, w)
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = min(max_height, h)
            new_w = int(new_h * aspect_ratio)

        # Resize the object if necessary
        if new_w < w or new_h < h:
            object_img = cv2.resize(object_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Calculate position to center the object
        x_offset = (self.image_width - new_w) // 2
        y_offset = (self.image_height - new_h) // 2

        # Overlay the object on the white background
        result = self.__overlay_image(bg, object_img, x_offset, y_offset)

        # Save the result
        cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def __get_bounding_box(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get the bounding box of the non-transparent region of an image.

        :param image: Input image with alpha channel.
        :return: Bounding box as (x, y, width, height).
        """
        # Use the alpha channel to find the bounding box of the object
        alpha_channel = image[:, :, 3]  # Extract the alpha channel
        coords = cv2.findNonZero(alpha_channel)  # Find all non-zero points (object)
        x, y, w, h = cv2.boundingRect(coords)  # Find the bounding rectangle
        
        return x, y, w, h
    
    def __overlay_image(self, bg: np.ndarray, fg: np.ndarray, x_offset: int = 0, y_offset: int = 0) -> np.ndarray:
        """
        Overlay a foreground image on a background image at the given offsets.

        :param bg: Background image.
        :param fg: Foreground image with alpha channel.
        :param x_offset: X offset for the foreground image.
        :param y_offset: Y offset for the foreground image.
        :return: Combined image with the foreground overlayed on the background.
        """
        y1, y2 = y_offset, y_offset + fg.shape[0]
        x1, x2 = x_offset, x_offset + fg.shape[1]

        alpha_fg = fg[:, :, 3] / 255.0
        alpha_bg = 1.0 - alpha_fg

        for c in range(0, 3):
            bg[y1:y2, x1:x2, c] = (alpha_fg * fg[:, :, c] + alpha_bg * bg[y1:y2, x1:x2, c])

        return bg


if __name__ == "__main__":
    processor = ImageProcessor()
    input_path = 'dog1.jpg'
    file, _ = os.path.splitext(input_path)
    output_path = f"{file}.output.png"
    
    # Convert to PNG with alpha channel first - ensures consistent format so every following function can assume certain things
    converted_to_png = processor.convert_to_png(input_path, output_path)
    if converted_to_png: # Only continue if we have the PNG image
        # Everything here assumes things about the image that are only true if we managed to convert to png
        processor.remove_background(output_path)
        processor.upscale(output_path)
        processor.overlay(output_path)
    else:
        logging.error(f"Failed. Couldnt convert to PNG.")