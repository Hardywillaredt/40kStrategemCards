import os
import numpy as np
import cv2
from PIL import Image

# === Configuration ===
INPUT_FOLDER = "C:/temp/TSonsStrategems/aeldari_spiritconclave/"
OUTPUT_FOLDER = "C:/temp/output/"
DEBUG_FOLDER = "C:/temp/debug/"
MARGIN = 30
HEIGHT_EXPAND_RATIO = 1.3

MIN_BAR_WIDTH = 20
MAX_BAR_WIDTH = 80
MORPH_KERNEL_SIZE = 10
BAR_EXTENSION_OVERLAP = 5

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# === Helper Functions ===
def get_background_color(img_np):
    pixels = img_np.reshape(-1, img_np.shape[-1])
    unique, counts = np.unique(pixels, axis=0, return_counts=True)
    return tuple(unique[np.argmax(counts)])

def crop_to_content(img_np, bg_color):
    mask = np.any(np.abs(img_np - bg_color) > 5, axis=-1)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img_np, img_np.shape[1::-1]
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img_np[y0:y1, x0:x1]
    return cropped, (x1 - x0, y1 - y0)

def get_foreground_mask(image_np, bg_color, tolerance=5):
    diff = np.abs(image_np.astype(int) - np.array(bg_color, dtype=int))
    mask = np.any(diff > tolerance, axis=-1).astype(np.uint8) * 255
    return mask

def extend_bar_via_connected_components(image_np, name, bg_color):
    # Step 1: Get strict binary mask after morph opening
    mask = get_foreground_mask(image_np, bg_color)
    Image.fromarray(mask).save(os.path.join(DEBUG_FOLDER, f"{name}_masknow.png"))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    Image.fromarray(cleaned).save(os.path.join(DEBUG_FOLDER, f"{name}_cleano.png"))

    # Step 2: Connected Components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    best_label = -1
    max_area = 0

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if MIN_BAR_WIDTH <= w <= MAX_BAR_WIDTH and area > max_area:
            max_area = area
            best_label = i

    if best_label == -1:
        print(f"[WARN] No bar found for {name}")
        return image_np

    # Step 3: Bounding box + color
    mask = labels == best_label
    ys, xs = np.where(mask)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cx, cy = int(xs.mean()), int(ys.mean())
    bar_color = tuple(image_np[cy, cx])

    y_start = max(y1 - BAR_EXTENSION_OVERLAP, y0)
    image_np[y_start:, x0:x1+1] = bar_color

    # Step 4: Debug images
    debug_mask = np.zeros_like(image_np)
    debug_mask[mask] = [255, 0, 255]  # Magenta
    boxed = image_np.copy()
    cv2.rectangle(boxed, (x0, y0), (x1, y1), (0, 255, 0), 2)

    Image.fromarray(cleaned).save(os.path.join(DEBUG_FOLDER, f"{name}_mask_cleaned.png"))
    Image.fromarray(debug_mask).save(os.path.join(DEBUG_FOLDER, f"{name}_bar_mask.png"))
    Image.fromarray(boxed).save(os.path.join(DEBUG_FOLDER, f"{name}_bar_overlay.png"))

    return image_np

# === Step 1: Load, crop, and collect sizes ===
image_paths = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.endswith(".PNG")]
cropped_images = []
sizes = []

for path in image_paths:
    with Image.open(path) as img:
        name = os.path.splitext(os.path.basename(path))[0]
        img = img.convert("RGB")
        img_np = np.array(img)
        bg_color = get_background_color(img_np)

        cropped_np, (w, h) = crop_to_content(img_np, bg_color)
        cropped_images.append((cropped_np, bg_color, name))
        sizes.append((w, h))
        Image.fromarray(cropped_np).save(os.path.join(DEBUG_FOLDER, f"{name}_cropped.png"))

# === Step 2: Compute base padded size ===
max_width = max(w for w, _ in sizes)
max_height = max(h for _, h in sizes)
base_width = max_width + 2 * MARGIN
base_height = int((max_height + 2 * MARGIN) * HEIGHT_EXPAND_RATIO)

# === Step 3: Pad, mask, extend bar ===
for cropped_np, bg_color, name in cropped_images:
    h, w = cropped_np.shape[:2]
    canvas = np.full((base_height, base_width, 3), bg_color, dtype=np.uint8)
    canvas[MARGIN:MARGIN+h, MARGIN:MARGIN+w] = cropped_np

    fg_mask = get_foreground_mask(canvas, bg_color)
    Image.fromarray(fg_mask).save(os.path.join(DEBUG_FOLDER, f"{name}_foreground_mask.png"))

    fg_only = canvas.copy()
    fg_only[fg_mask == 0] = 0
    Image.fromarray(fg_only).save(os.path.join(DEBUG_FOLDER, f"{name}_foreground_only.png"))

    extended = extend_bar_via_connected_components(canvas.copy(), name, bg_color)

    Image.fromarray(extended).save(os.path.join(OUTPUT_FOLDER, f"{name}_final.png"))
    Image.fromarray(extended).save(os.path.join(DEBUG_FOLDER, f"{name}_final_preview.png"))

print("✅ Full pipeline complete. Final images and debug output saved.")
