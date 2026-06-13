import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image introuvable : {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    img_lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(img_lab_enhanced, cv2.COLOR_Lab2RGB)


def remove_specular(image: np.ndarray, threshold: int = 240) -> np.ndarray:
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = img_hsv[:, :, 2]
    specular_mask = (v_channel > threshold).astype(np.uint8) * 255
    if specular_mask.sum() == 0:
        return image
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_inpainted = cv2.inpaint(img_bgr, specular_mask, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, size: tuple = (128, 128)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (image_float - mean[np.newaxis, np.newaxis, :]) / std[np.newaxis, np.newaxis, :]


def preprocess_pipeline(image_path: str, size: tuple = (128, 128)) -> np.ndarray:
    img = load_image(image_path)
    img = apply_clahe(img)
    img = remove_specular(img)
    img = resize_image(img, size)
    img = normalize_image(img)
    return img


def preprocess_mask(mask_path: str, size: tuple = (128, 128)) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Masque introuvable : {mask_path}")
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)