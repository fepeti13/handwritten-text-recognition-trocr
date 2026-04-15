#!/usr/bin/env python3
"""
Visually verify that PageXML bounding boxes align with the original image.
Opens a random page from a register and draws 5 random text line boxes on it.

Usage:
    python scripts/verify-crops.py data/raw/register-71-101-morti-1
    python scripts/verify-crops.py data/raw/register-71-101-morti-1 --page 3
    python scripts/verify-crops.py data/raw/register-71-101-morti-1 --n 10
    python scripts/verify-crops.py data/raw/register-71-101-morti-1 --all
"""

import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
from tqdm import tqdm

NS = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]


def strip_page_prefix(filename):
    if '_' in filename:
        parts = filename.split('_', 1)
        if parts[0].isdigit():
            return parts[1]
    return filename


def load_page(xml_file, images_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    page = root.find('.//ns:Page', NS)
    if page is None:
        raise ValueError(f"No <Page> element in {xml_file}")

    image_filename = strip_page_prefix(page.get('imageFilename'))
    image_path = os.path.join(images_dir, image_filename)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

    # Collect all text lines with their bboxes and text
    lines = []
    for textline in root.findall('.//ns:TextLine', NS):
        coords_elem = textline.find('ns:Coords', NS)
        if coords_elem is None:
            continue
        points = coords_elem.get('points', '').split()
        if not points:
            continue

        x_coords = [int(float(p.split(',')[0])) for p in points]
        y_coords = [int(float(p.split(',')[1])) for p in points]
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

        unicode_elem = textline.find('.//ns:Unicode', NS)
        text = unicode_elem.text.strip() if unicode_elem is not None and unicode_elem.text else ""

        lines.append({"bbox": bbox, "text": text})

    return image, lines


def draw_boxes(image, selected_lines):
    draw = ImageDraw.Draw(image)

    # Try to load a font; fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    for i, line in enumerate(selected_lines):
        color = COLORS[i % len(COLORS)]
        x0, y0, x1, y1 = line["bbox"]

        # Draw rectangle (3px border)
        for offset in range(3):
            draw.rectangle(
                [x0 - offset, y0 - offset, x1 + offset, y1 + offset],
                outline=color
            )

        # Label with index and ground-truth text
        label = f"[{i+1}] {line['text'][:40]}"
        draw.text((x0, max(0, y0 - 32)), label, fill=color, font=font)

    return image


def process_single_page(xml_file, images_dir, n, out_path):
    image, lines = load_page(xml_file, images_dir)
    if not lines:
        print(f"  No text lines found in {xml_file.name}, skipping.")
        return False
    selected = random.sample(lines, min(n, len(lines)))
    annotated = draw_boxes(image.copy(), selected)
    annotated.save(out_path, quality=90)
    return True


def main(args):
    register_path = args.register.rstrip('/')
    images_dir = os.path.join(register_path, "images")
    pagexml_dir = os.path.join(register_path, "pagexml", "page")

    xml_files = sorted(Path(pagexml_dir).glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {pagexml_dir}")
        return

    # -- All pages mode --
    if args.all:
        register_name = os.path.basename(register_path)
        out_dir = os.path.join("data/processed/verify", register_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Generating verification images for all {len(xml_files)} pages...")
        print(f"Output folder: {out_dir}\n")

        for xml_file in tqdm(xml_files, desc="Pages"):
            out_path = os.path.join(out_dir, f"{xml_file.stem}.jpg")
            process_single_page(xml_file, images_dir, args.n, out_path)

        print(f"\nDone — {len(xml_files)} images saved to: {out_dir}")
        os.system(f'xdg-open "{out_dir}" &')
        return

    # -- Single page mode --
    if args.page is not None:
        idx = args.page - 1
        if idx < 0 or idx >= len(xml_files):
            print(f"Page {args.page} out of range (1–{len(xml_files)})")
            return
        xml_file = xml_files[idx]
    else:
        xml_file = random.choice(xml_files)

    print(f"Page XML : {xml_file.name}")
    image, lines = load_page(xml_file, images_dir)
    print(f"Total text lines on page: {len(lines)}")

    if not lines:
        print("No text lines found in this page.")
        return

    n = min(args.n, len(lines))
    selected = random.sample(lines, n)
    for i, line in enumerate(selected):
        print(f"  [{i+1}] bbox={line['bbox']}  text='{line['text']}'")

    annotated = draw_boxes(image.copy(), selected)

    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "verify_crops.jpg")
    annotated.save(out_path, quality=90)
    print(f"\nSaved to: {out_path}")
    os.system(f'xdg-open "{out_path}" &')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify PageXML crop alignment on original images.")
    parser.add_argument("register", help="Path to register folder (e.g. data/raw/register-71-101-morti-1)")
    parser.add_argument("--page", type=int, default=None, help="Page number to use (default: random)")
    parser.add_argument("--n", type=int, default=5, help="Number of boxes to draw (default: 5)")
    parser.add_argument("--all", action="store_true", help="Generate verification image for every page")
    args = parser.parse_args()

    main(args)
