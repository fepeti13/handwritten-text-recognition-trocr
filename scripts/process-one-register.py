"""
Process ONE register: extract text lines from PageXML and crop images.
Register ID comes from LAST number in folder name.
"""
import xml.etree.ElementTree as ET
from PIL import Image
import os
import csv
from pathlib import Path
from tqdm import tqdm


def get_register_id_from_path(register_path):
    """
    Extract register ID from LAST number in folder name.
    register-71-101-morti-1 → r001
    register-71-101-morti-2 → r002
    register-72-105-nascuti-3 → r003
    """
    folder_name = os.path.basename(register_path.rstrip('/'))
    
    # Get the last part after final hyphen
    parts = folder_name.split('-')
    if parts:
        last_part = parts[-1]
        if last_part.isdigit():
            num = int(last_part)
            return f"r{num:03d}"
    
    # Fallback
    print(f"Warning: Could not extract register ID from '{folder_name}'")
    return "r999"


def strip_page_prefix(filename):
    """
    Strip page number prefix from filename.
    0001_IMG_20250226_113245.jpg → IMG_20250226_113245.jpg
    """
    if '_' in filename:
        parts = filename.split('_', 1)
        if parts[0].isdigit():
            return parts[1]
    return filename


def process_register(register_path, output_dir):
    """
    Process one register folder.
    
    Args:
        register_path: Path to register folder (e.g., data/raw/register-71-101-morti-1)
        output_dir: Where to save processed data (e.g., data/processed)
    """
    
    # Auto-detect register ID from LAST number in folder name
    register_id = get_register_id_from_path(register_path)
    print(f"Processing: {os.path.basename(register_path)}")
    print(f"Register ID: {register_id}")
    
    images_dir = os.path.join(register_path, "images")
    pagexml_dir = os.path.join(register_path, "pagexml", "page")
    
    # Create output directories
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    
    # Get all XML files
    xml_files = sorted(Path(pagexml_dir).glob("*.xml"))
    
    print(f"Found {len(xml_files)} PageXML files")
    
    ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    
    all_lines = []
    line_counter = 0
    
    for xml_file in tqdm(xml_files, desc="Processing pages"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get page info
        page = root.find('.//ns:Page', ns)
        if page is None:
            print(f"Warning: No Page element in {xml_file.name}")
            continue
        
        # Get image filename and strip prefix
        image_filename_xml = page.get('imageFilename')
        image_filename = strip_page_prefix(image_filename_xml)
        
        # Get page number from metadata
        metadata = root.find('.//ns:TranskribusMetadata', ns)
        page_nr = metadata.get('pageNr') if metadata is not None else '0'
        
        # Load original image
        image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            print(f"  XML expected: {image_filename_xml}")
            print(f"  After stripping: {image_filename}")
            continue
        
        try:
            original_image = Image.open(image_path)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            continue
        
        # Find all TextLine elements (in all table cells)
        textlines = root.findall('.//ns:TextLine', ns)
        
        for textline in textlines:
            # Get coordinates
            coords_elem = textline.find('ns:Coords', ns)
            if coords_elem is None:
                continue
            
            coords_str = coords_elem.get('points')
            if not coords_str:
                continue
            
            # Parse coordinates to bounding box
            points = coords_str.split()
            x_coords = [int(float(p.split(',')[0])) for p in points]
            y_coords = [int(float(p.split(',')[1])) for p in points]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            bbox = (x_min, y_min, x_max, y_max)
            
            # Get text
            unicode_elem = textline.find('.//ns:Unicode', ns)
            if unicode_elem is None or unicode_elem.text is None:
                continue
            
            text = unicode_elem.text.strip()
            if not text:  # Skip empty lines
                continue
            
            # Crop text line
            try:
                line_image = original_image.crop(bbox)
            except Exception as e:
                print(f"Error cropping line: {e}")
                continue
            
            # Save: r001_p001_l0000.png
            line_filename = f"{register_id}_p{int(page_nr):03d}_l{line_counter:04d}.png"
            line_path = os.path.join(output_images_dir, line_filename)
            line_image.save(line_path)
            
            # Store metadata
            all_lines.append({
                'image_path': line_filename,
                'text': text,
                'register_id': register_id,
                'page_nr': page_nr,
                'line_id': line_counter,
                'bbox': f"{bbox}"
            })
            
            line_counter += 1
    
    # Save metadata to CSV
    csv_path = os.path.join(output_dir, "metadata.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['image_path', 'text', 'register_id', 'page_nr', 'line_id', 'bbox']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_lines)
    
    print(f"\n{'='*60}")
    print(f"✓ Processed {len(xml_files)} pages")
    print(f"✓ Extracted {len(all_lines)} text lines")
    print(f"✓ Register ID: {register_id}")
    print(f"✓ Images saved to: {output_images_dir}")
    print(f"✓ Metadata saved to: {csv_path}")
    print(f"{'='*60}")
    
    return all_lines


if __name__ == "__main__":
    # Configuration - UPDATE THIS to match your folder name
    register_path = "data/raw/register-71-101-morti-1"  # ← Add -1 at end!
    output_dir = "data/processed"
    
    # Process (register ID extracted from last number)
    process_register(register_path, output_dir)