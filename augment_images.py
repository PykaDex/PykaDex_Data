import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import os
from PIL import Image

def transparency_checker(image):
    img = Image.open(image, 'r')
    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True
    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            return True
    return False

def bg_swap(poke_file,bg_file,comb_file):
    """
    """
    pokemon = Image.open(poke_file, 'r')
    background = Image.open(bg_file,'r')
    background= background.resize(pokemon.size)
    comb_img = Image.new('RGB', (pokemon.size), (0, 0, 0, 0))
    comb_img.paste(background, (0,0))
    comb_img.paste(pokemon, (0,0), mask=pokemon)
    final = comb_img.convert('RGB')
    final.save(comb_file, format="jpeg")
    
def augment_image(image_loc,aug_loc,N,show=True,sub_dirs=None):
    """
    image_loc = path to orginal image
    aug_loc   = path to augmented images
    N = number of augmented images generated
    """

    ##########################################################
    # check if image transparent
    ##########################################################

    if transparency_checker(image_loc) == True:
        print( ' - TRANSPARENT')

        if not os.path.exists(bg_folder):
            print('cant locate backgrounds')

        for dirpath, dirs, files in os.walk(bg_folder):  
            for filename in files:
                background = os.path.join(dirpath,filename)
                
                new_combined_filename = os.path.join(comb_folder,image_loc[len(source_loc):])

                if not os.path.exists(os.path.join(comb_folder,sub_dirs)):
                    os.makedirs(os.path.join(comb_folder,sub_dirs))

                bg_swap(image_loc,background,new_combined_filename)
                augment_image(new_combined_filename,aug_loc,N,show=False,sub_dirs=sub_dirs)

    #########################
    # get image
    #########################


    image = imageio.imread(image_loc, as_gray=False, pilmode="RGB")
    if show==True:
        print("Original:")
        ia.imshow(image)

    ###########################
    # augment
    ###########################

    seq = iaa.Sequential([
        iaa.Affine(rotate=(-25, 25)),
        iaa.AdditiveGaussianNoise(scale=(30, 90)),
        iaa.Crop(percent=(0, 0.4)),
        iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),  # crop and pad images
        iaa.ElasticTransformation(alpha=90, sigma=9),  # water-like effect
        iaa.Cutout()  # replace one squared area within the image by a constant intensity value
    ], random_order=True)

    images_aug = [seq(image=image) for _ in range(N)]

    if show==True:
        print("Augmented:")
        ia.imshow(ia.draw_grid(images_aug[:8], cols=4, rows=2),)

    ###########################
    # save augmented images
    ###########################

    if not os.path.exists(aug_loc):
            os.makedirs(aug_loc)

    for n in range(0,len(images_aug)):
        new_image_name = "{}/{}-A{}.jpg".format(aug_loc,(image_loc.split('/')[-1]).split('.')[0],n)
        imageio.imwrite(new_image_name, images_aug[n])
    print(" [] - augmented images saved to '{}'".format(new_image_name))

    #############################


def augment_folder(folder_loc,aug_loc,N):
    """
    source_loc = path to orginal folder containing images to be augmented
    aug_loc   = path to augmented images
    N = number of augmented images generated per orginal image
    """

    for dirpath, dirs, files in os.walk(source_loc):  
        #sprint(dirs)
        for filename in files:
            #print(filename)

            sub_dirs = dirpath[len(source_loc):]
            if filename[0] == '.':
               print(' # image error - "{}"'.format(os.path.join(sub_dirs,filename)))
               continue

            image_loc = os.path.join(dirpath,filename)
            aug_loc_ = os.path.join(aug_loc,sub_dirs)
            #print(image_loc)
           

            #try:
            augment_image(image_loc,aug_loc_,N,show=False,sub_dirs=sub_dirs)
            #except:
            #    print(' # image error - "{}"'.format(os.path.join(dirpath[len(source_loc):],filename)))
            #    pass

##############################################################
# SETUP
##############################################################

print('starting...')

# source dir containg images to be augmented
path_to_training_data = 'Data/Images/'
source_loc = path_to_training_data+'GenX/'

# dir for augmented images to be placed (file structure will be kept i.e augmented/bublsaur/image,jpg)
aug_loc = path_to_training_data+'GenX_Augmented/'

# dir containg bg images for transparent pokemon
bg_folder = path_to_training_data+'backgrounds/'

# dir for combined transparent images with new backgrrounds
comb_folder = path_to_training_data+'GenX_BGswap/'

#temporarily set number of augmentations per image to 1 to test npy data creating
N=1
augment_folder(source_loc,aug_loc,N)
print('finished.')