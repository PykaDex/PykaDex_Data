import os
import cv2
import numpy as np 
from tqdm import tqdm
import pathlib
import shutil

REBUILD_DATA = True


class Pokemon():

    IMG_SIZE = 80
    data_dir = "../../GenX"

    data_save_name = "Pokemon_color2_80_test.npy"
    model_directory = "Compiled_Data"
    save_path = model_directory + "/" + data_save_name


    if os.path.exists(f"{data_dir}/.DS_Store"):
        shutil.rmtree(f"{data_dir}/.DS_Store")
    if os.path.exists(f"{data_dir}/._.DS_Store"):
        shutil.rmtree(f"{data_dir}/._.DS_Store")
    else:
        pass

    pokemon = os.listdir(data_dir)
    pokemon.sort()
    

    directories = []
    LABELS = {}

    for i in range(len(pokemon)):
        directory = data_dir + "/" + pokemon[i]
        directories.append(directory)

        LABELS[directories[i]] = i


    training_data = []
    counts = [0]*len(pokemon)
    click = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(f"Fetching {self.pokemon[self.click]}'s images")

            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

                    if img.shape == (self.IMG_SIZE,self.IMG_SIZE):
                        print(path)
                    else:
                        pass

                    self.training_data.append([np.array(img), np.eye(len(self.pokemon))[self.LABELS[label]]])

                except Exception as e:
                    pass
                
                self.counts[self.click] += 1 

            self.click += 1

        print(self.counts)
        np.random.shuffle(self.training_data)


        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
            np.save(self.save_path, self.training_data)
        if os.path.exists(self.model_directory):
            np.save(self.save_path, self.training_data)


if REBUILD_DATA:
    pokemon_data = Pokemon()
    pokemon_data.make_training_data()

