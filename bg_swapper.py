import os
from PIL import Image

def bg_swap(poke_file,bg_file,comb_file):
    """
    """
    pokemon = Image.open(poke_file, 'r')
    background = Image.open(bg_file,'r')
    background = background.resize(pokemon.size)
    comb_img = Image.new('RGBA', (pokemon.size), (0, 0, 0, 0))
    comb_img.paste(background, (0,0))
    comb_img.paste(pokemon, (0,0), mask=pokemon)
    comb_img.save(comb_file, format="jpg")

poke_file = '/mnt/c/Users/benja/Programming/Python_Projects/PykaDex/Training_Data/Pokemon_source_images/Bulbasaur/00000097.jpg'
bg_file = '/mnt/c/Users/benja/Programming/Python_Projects/PykaDex/Training_Data/backgrounds/forest.png'

bg_swap(poke_file,bg_file,'new_file.jpg')

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
    print('ju5h6yrtegre')
    return False

transparency_checker('/mnt/c/Users/benja/Programming/Python_Projects/PykaDex/Training_Data/combined/00082.jpg')
