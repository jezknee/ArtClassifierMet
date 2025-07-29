from pathlib import Path
import pandas as pd
import PIL.Image as Image
import webcolors
import numpy as np
from collections import Counter

pd.set_option("display.max_columns", None)

def create_colour_df_optimized(filename):
    """Optimized version using numpy array operations"""
    im = Image.open(filename)
    rgb_im = im.convert('RGB')
    
    # Convert to numpy array for vectorized operations
    img_array = np.array(rgb_im)
    
    # Reshape to get all pixels as (R,G,B) tuples
    pixels = img_array.reshape(-1, 3)
    
    # Convert to tuples and count occurrences
    pixel_tuples = [tuple(pixel) for pixel in pixels]
    pixel_counts = Counter(pixel_tuples)
    
    colour_dictionary = {}
    
    # Process unique colors only (much faster)
    for rgb, count in pixel_counts.items():
        try:
            color_name = webcolors.rgb_to_name(rgb)
            colour_dictionary[color_name] = colour_dictionary.get(color_name, 0) + count
        except ValueError:
            # Skip unknown colors or handle them as needed
            pass
    
    if not colour_dictionary:
        return pd.DataFrame(columns=['Colour', 'Count', 'Percentage', 'Filename'])
    
    # Create DataFrame
    df = pd.DataFrame(list(colour_dictionary.items()), columns=['Colour', 'Count'])
    df['Percentage'] = df['Count'] / df['Count'].sum() * 100
    df['Filename'] = filename
    
    return df


def process_images_batch(image_paths, method='optimized', sample_rate=0.1):
    """Process all images and return combined DataFrame"""
    all_dfs = []
    total_images = len(image_paths)
    
    print(f"Total images found: {total_images}")
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing {image_path.name} ({i + 1}/{total_images})")
        
        try:
            df = create_colour_df_optimized(str(image_path))
            
            all_dfs.append(df)
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue
    
    # Combine all DataFrames at once (more efficient than repeated concat)
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=['Colour', 'Count', 'Percentage', 'Filename'])


# Main execution
if __name__ == "__main__":
    all_path = Path.cwd() / "Data" / "Images" / "Raw"
    jpg_files = sorted(all_path.glob('*.jpg'))
    
    # Method 1: Optimized (processes all pixels but more efficiently)
    print("Using optimized method...")
    all_images_df = process_images_batch(jpg_files, method='optimized')
    
    print(all_images_df.head())
    all_images_df.to_csv(Path.cwd() / "Data" / "ImageColoursRaw.csv", index=False)