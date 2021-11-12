import os
import cv2
import numpy as np 
from tqdm import tqdm
import pathlib
import shutil
from config import *


data_dir = image_dir+'GenX/'

def Pokemon():

    pokemon = os.listdir(data_dir) # Creating a list with the names of Pokemon which are taken from the name of the folders
    pokemon.sort()
    

    directories = [] # List of directories for each pokemon folder
    LABELS = {} # Dictionary where the directories of each pokemon and their label will be stored

    for i in range(len(pokemon)):
        pokemon_directory = data_dir + "/" + pokemon[i]
        directories.append(pokemon_directory)

        LABELS[directories[i]] = i

    return pokemon, LABELS


def make_training_data(pokemon, LABELS, IMG_SIZE):

    training_data = []
    counts = [0]*len(pokemon)
    click = 0

    for label in LABELS:
        print(f"Fetching {pokemon[click]}'s images")

        for f in tqdm(os.listdir(label)):
            try:
                path = os.path.join(label, f) 
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # It makes sure that cv2 reads images as RGB instead of the default BGR
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
                if img.shape == (IMG_SIZE,IMG_SIZE): #This was written to check for any grayscale images (single channel images)
                    print(path)
                else:
                    pass
    
                training_data.append([np.array(img), np.eye(len(pokemon))[LABELS[label]]]) # Training data has both the numpy array of the image and the associated label of the image appended 

            except Exception as e:
                #print('uhoh')
                pass
                
            counts[click] += 1 

        click += 1

    print(counts)
    np.random.shuffle(training_data)

    current_folder_name = os.path.basename(os.path.normpath(data_dir))
    data_save_name = f"Pokemon_Data_Colour_{current_folder_name}_{IMG_SIZE}.npy" 

    model_directory = "Data/numpy_files"
    save_path = model_directory + "/" + data_save_name # Directory path where numpy file will be saved in 

    if not os.path.exists(model_directory): # Saving the numpy file in preferred directory
        os.makedirs(model_directory)
        np.save(save_path, training_data)
    if os.path.exists(model_directory):
        np.save(save_path, training_data)

#########################################################################

REBUILD_DATA = True

if REBUILD_DATA:

    if os.path.exists(f"{data_dir}/.DS_Store"): # This is mainly for mac user. It removes hidden folders within the images file
        shutil.rmtree(f"{data_dir}/.DS_Store")
    if os.path.exists(f"{data_dir}/._.DS_Store"):
        shutil.rmtree(f"{data_dir}/._.DS_Store")
    else:
        pass

    pokemon, LABELS = Pokemon()
    make_training_data(pokemon, LABELS, 80)
