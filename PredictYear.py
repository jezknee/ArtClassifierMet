from pathlib import Path
import pandas as pd
import PIL.Image as Image
import webcolors

pd.set_option("display.max_columns", None)
#pd.set_option("low_memory",False)



def create_colour_df(filename):
    colour_dictionary = dict()
    #print(filename)
    im = Image.open(filename)
    rgb_im = im.convert('RGB')

    for x in range(rgb_im.width):
        for y in range(rgb_im.height):
            # Get the RGB values of the pixel at (x, y)
            rgb = rgb_im.getpixel((x, y))
            color_name = webcolors.rgb_to_name(rgb)
            #print(r, g, b)
            colour_dictionary[color_name] = colour_dictionary.get(color_name, 0) + 1

    # Create a DataFrame from the colour dictionary
    df = pd.DataFrame(list(colour_dictionary.items()), columns=['Colour', 'Count'])
    df['Percentage'] = df['Count'] / df['Count'].sum() * 100
    df['Filename'] = filename

    # Print the DataFrame
    print(df.head())
    return df



all_path = Path.cwd() / "Data" / "Images" / "Processed"
png_files = sorted(all_path.glob('*.jpg'))
#print(png_files)

for i in png_files:
    filename = i.name
    create_colour_df(filename)
