from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageChops


IMAGE_SIZE = 28
DIGIT_SIZE = 20
INK_THRESHOLD = 245


def shift_with_white_fill(image: Image.Image, shift_x: int, shift_y: int) -> Image.Image:
    shifted = ImageChops.offset(image.convert("L"), shift_x, shift_y)
    pixels = np.array(shifted, dtype=np.uint8, copy=True)
    if shift_x > 0:
        pixels[:, :shift_x] = 255
    elif shift_x < 0:
        pixels[:, shift_x:] = 255
    if shift_y > 0:
        pixels[:shift_y, :] = 255
    elif shift_y < 0:
        pixels[shift_y:, :] = 255
    return Image.fromarray(pixels, mode="L")


def to_grayscale_28(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    grayscale = background.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return ensure_black_ink_on_white(grayscale)


def ensure_black_ink_on_white(image: Image.Image) -> Image.Image:
    grayscale = image.convert("L")
    pixels = np.asarray(grayscale, dtype=np.uint8)
    corners = np.array([pixels[0, 0], pixels[0, -1], pixels[-1, 0], pixels[-1, -1]], dtype=np.uint8)
    if float(np.median(corners)) < 128:
        return Image.fromarray(255 - pixels, mode="L")
    return grayscale


def center_digit(image: Image.Image) -> Image.Image:
    grayscale = to_grayscale_28(image)
    pixels = np.asarray(grayscale, dtype=np.uint8)
    ink_rows, ink_cols = np.where(pixels < INK_THRESHOLD)
    if len(ink_rows) == 0 or len(ink_cols) == 0:
        return grayscale

    left = int(ink_cols.min())
    right = int(ink_cols.max()) + 1
    upper = int(ink_rows.min())
    lower = int(ink_rows.max()) + 1
    digit = grayscale.crop((left, upper, right, lower))

    scale = min(DIGIT_SIZE / digit.width, DIGIT_SIZE / digit.height)
    resized_width = max(1, int(round(digit.width * scale)))
    resized_height = max(1, int(round(digit.height * scale)))
    digit = digit.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
    offset_x = (IMAGE_SIZE - resized_width) // 2
    offset_y = (IMAGE_SIZE - resized_height) // 2
    canvas.paste(digit, (offset_x, offset_y))

    centered_pixels = np.asarray(canvas, dtype=np.uint8)
    rows, cols = np.where(centered_pixels < INK_THRESHOLD)
    if len(rows) == 0 or len(cols) == 0:
        return canvas
    shift_x = int(round((IMAGE_SIZE - 1) / 2 - float(cols.mean())))
    shift_y = int(round((IMAGE_SIZE - 1) / 2 - float(rows.mean())))
    return shift_with_white_fill(canvas, shift_x, shift_y)


def preprocess_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    processed = center_digit(image)
    pixels = np.asarray(processed, dtype=np.float32) / 255.0
    return pixels.reshape(784), processed


def image_to_feature_vector(image: Image.Image) -> np.ndarray:
    features, _processed = preprocess_image(image)
    return features


def image_to_preview_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
