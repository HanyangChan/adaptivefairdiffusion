import os
from PIL import Image, ImageDraw, ImageFont

def create_image_grid(prompt_name, display_name):
    seeds = [42, 123, 2024]
    
    # Define columns
    columns = [
        {"title": "No Guidance", "path_template": "results_optimized/{prompt}_baseline_none_seed_{seed}.png"},
        {"title": "Fair Diffusion (Constant 3.0)", "path_template": "results_optimized/{prompt}_baseline_constant_3.0_seed_{seed}.png"},
        {"title": "Ours (Reverse Cosine 7.0 Unoptimized)", "path_template": "results_reverse/{prompt}_reverse_cosine_7.0_seed_{seed}.png"},
        {"title": "Ours (Reverse Cosine 7.0 Optimized)", "path_template": "results_optimized/{prompt}_reverse_cosine_7.0_seed_{seed}.png"}
    ]
    
    # Load first image to get dimensions
    test_img_path = columns[0]["path_template"].format(prompt=prompt_name, seed=seeds[0])
    try:
        with Image.open(test_img_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        print(f"Could not load image {test_img_path}: {e}")
        return

    # Dimensions for the grid
    img_w, img_h = 256, 256
    padding = 20
    header_height = 80
    row_label_width = 150
    
    grid_w = row_label_width + len(columns) * img_w + (len(columns) + 1) * padding
    grid_h = header_height + len(seeds) * img_h + (len(seeds) + 1) * padding
    
    grid_img = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid_img)
    
    # Try loading a font
    try:
        # Use a larger standard font on Mac
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 25)
    except:
        font = ImageFont.load_default()
        
    # Draw Column Headers
    for col_idx, col in enumerate(columns):
        x = row_label_width + padding + col_idx * (img_w + padding)
        y = padding
        # Draw centered text
        text = col["title"]
        # Approximate centering
        draw.text((x + img_w/2 - len(text)*6, y + 25), text, fill=(0,0,0), font=font)
        
    # Draw Rows
    for row_idx, seed in enumerate(seeds):
        y = header_height + padding + row_idx * (img_h + padding)
        
        # Row label
        draw.text((padding, y + img_h/2 - 15), f"Seed {seed}", fill=(0,0,0), font=font)
        
        # Images
        for col_idx, col in enumerate(columns):
            x = row_label_width + padding + col_idx * (img_w + padding)
            
            img_path = col["path_template"].format(prompt=prompt_name, seed=seed)
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    img = img.resize((img_w, img_h), Image.Resampling.LANCZOS)
                    grid_img.paste(img, (x, y))
            else:
                print(f"Missing {img_path}")
                # draw empty box
                draw.rectangle([x, y, x+img_w, y+img_h], outline=(255,0,0))
                
    out_filename = f"grid_{prompt_name}.png"
    grid_img.save(out_filename)
    print(f"Saved {out_filename}")

if __name__ == '__main__':
    create_image_grid("A_portrait_of_a_doctor", "A portrait of a doctor")
    create_image_grid("A_photo_of_a_nurse", "A photo of a nurse")
