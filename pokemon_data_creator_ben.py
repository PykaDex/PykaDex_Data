import os
import cv2
import numpy as np 
from tqdm import tqdm
import pathlib
import shutil
from config import *

def data_structure():
    """
    """
    dirs_to_ignore = ['backgrounds','README.md']

    # will be a list of dictionairs containg info of each image based on file structure
    image_data = []

    # go through each image dir clean and augmented 
    for dir_name in os.listdir(image_dir):

        # skip folder if in ignore list
        if dir_name in dirs_to_ignore:
            continue

        dir_data = []

        # get each pokemon name
        for pokemon_name in os.listdir(os.path.join(image_dir,dir_name)):

            # get each image
            for image in os.listdir(os.path.join(image_dir,dir_name,pokemon_name)):

                # create a dictionary of each image contiang its info
                image_info = {
                'path':os.path.join(image_dir,dir_name,pokemon_name,image),
                'name':pokemon_name,
                'augmented':dir_name[-9:]=='Augmented',
                }
                dir_data.append(image_info)

        image_data.append({'dir_name': dir_name,'data' :dir_data})

    return image_data

def make_training_data(image_data):
    """
    """

    cheap_label_fix = ['Bulbasaur','Charmander','Squirtle']

    training_data = []
    
    for dir_data in image_data:
        print(f"Fetching {dir_data['dir_name']}'s images")

        counts = [0]*len(dir_data['data'])
        click = 0
        for image in tqdm(dir_data['data']):
            try:
                path = image['path']
               # print(image)
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # It makes sure that cv2 reads images as RGB instead of the default BGR
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                if img.shape == (IMG_SIZE,IMG_SIZE): #This was written to check for any grayscale images (single channel images)
                    print(path)
                else:
                    pass                    
                        
                training_data.append([np.array(img), np.eye(len(cheap_label_fix))[cheap_label_fix.index(image['name'])]]) # Training data has both the numpy array of the image and the associated label of the image appended 

            except Exception as e:
                #print('uhoh')
                pass
            
            counts[click] += 1 

        click += 1

        #print(counts)
        np.random.shuffle(training_data)

        current_folder_name = dir_data['dir_name']
        save_path = os.path.join(model_directory,f"Pokemon_Data_Colour_{current_folder_name}_{IMG_SIZE}.npy") # Directory path where numpy file will be saved in 

        if not os.path.exists(model_directory): # Saving the numpy file in preferred directory
            os.makedirs(model_directory)
            np.save(save_path, training_data)
        if os.path.exists(model_directory):
            np.save(save_path, training_data)
            print(f'data saved to {save_path}')

#############################################################

REBUILD_DATA = True

if REBUILD_DATA:

    IMG_SIZE = 80 # Size that you want the image to be
    

    if os.path.exists(f"{image_dir}/.DS_Store"): # This is mainly for mac user. It removes hidden folders within the images file
        shutil.rmtree(f"{image_dir}/.DS_Store")
    if os.path.exists(f"{image_dir}/._.DS_Store"):
        shutil.rmtree(f"{image_dir}/._.DS_Store")
    else:
        pass

    image_data = data_structure()
    make_training_data(image_data)
