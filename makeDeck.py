# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 09:14:08 2025

@author: hardy
"""

from PIL import Image
import os
import math

# === Parameters ===
card_front_dir = "C:/temp/output/"  # folder with front images
card_back_path = "C:/temp/TSonsStrategems/adeptasororitas/bg/finalBack.png"

# === Collect card fronts ===
card_front_paths = sorted([
    os.path.join(card_front_dir, fname)
    for fname in os.listdir(card_front_dir)
    if fname.endswith(".png")
])
num_cards = len(card_front_paths)

# === Determine grid size ===
rows = int(math.floor(math.sqrt(num_cards)))
cols = int(math.ceil(num_cards / rows))

# === Load images ===
fronts = [Image.open(path).convert("RGB") for path in card_front_paths]
card_width, card_height = fronts[0].size
back = Image.open(card_back_path).convert("RGB").resize((card_width, card_height))
blank = Image.new("RGB", (card_width, card_height), "white")

# === Pad with blanks ===
total_slots = rows * cols
while len(fronts) < total_slots:
    fronts.append(blank)

# === Create sheets ===
sheet_width = cols * card_width
sheet_height = rows * card_height
sheet_fronts = Image.new("RGB", (sheet_width, sheet_height), "white")
sheet_backs = Image.new("RGB", (sheet_width, sheet_height), "white")

for idx, front in enumerate(fronts):
    x = (idx % cols) * card_width
    y = (idx // cols) * card_height
    sheet_fronts.paste(front, (x, y))
    sheet_backs.paste(back, (x, y))

# === Save outputs ===
front_out_path = "C:/dev/40kStrategemCards/adeptaSororitasFront.png"
back_out_path = "C:/dev/40kStrategemCards/adeptaSororitasBack.png"
sheet_fronts.save(front_out_path)
sheet_backs.save(back_out_path)

front_out_path, back_out_path
