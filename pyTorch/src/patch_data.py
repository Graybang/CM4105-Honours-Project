from PIL import Image
import os

# define the input image directory
input_dir = "C:/Users/Percy/Flickr1024/Validation/"

# define the output directory for the patches
output_dir = "C:/Users/Percy/Flickr1024/Validation_patched/"

# define the size of the patches
patch_size = (128, 128)

# loop through all the files in the input directory
for filename in os.listdir(input_dir):
    # open the image file
    with Image.open(os.path.join(input_dir, filename)) as img:
        # get the size of the image
        width, height = img.size
        # loop through the image, creating patches
        for x in range(0, width, patch_size[0]):
            for y in range(0, height, patch_size[1]):
                # define the patch coordinates
                left = x
                top = y
                right = x + patch_size[0]
                bottom = y + patch_size[1]
                # crop the patch from the image
                patch = img.crop((left, top, right, bottom))
                # construct the new filename with patch coordinates
                new_filename = f"{x}_{y}_{filename.split('.')[0]}.png"
                # save the patch to the output directory
                patch.save(os.path.join(output_dir, new_filename))
