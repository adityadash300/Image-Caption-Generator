from os import listdir
from PIL import Image,ImageOps
import numpy as np

PATH = "C:/Users/AMAR/Documents/GitHub/Image-Caption-Generator/Dataset/Images/"
RPATH = "C:/Users/AMAR/Documents/GitHub/Image-Caption-Generator/Dataset/resized_Dataset/"
TPATH = 'C:/Users/Aditya Dash/Documents/GitHub/Image-Caption-Generator/Dataset/Captions/descriptions.txt'

imagesList = listdir(PATH)

NEW_DIM = (200,200)

def loadImages(load_path):

    loadedImages = []
    for image in imagesList:
        img = Image.open(load_path+image)
        loadedImages.append(img)
        img.close()

    return loadedImages



def preprocess_Img(load_path,save_path):

    images = loadImages(load_path)                                              #Loading the images from the dataset using loadImages(load_path)

#resizing the loaded images

    pixel_img = []
    preprocessed_img = []
    for img in images:
        img = img.resize(NEW_DIM)
        img = ImageOps.expand(img,border=50,fill="black")                       #Padding the resized images
        preprocessed_img.append(img)                                            #converting the padded and resized images into array(pixel values)
        pixel_img.append(np.asarray(img))

#saving the Preprocessed images

    j=0
    for i in preprocessed_img:
        i.save(save_path+imagesList[j],"JPEG")
        j+=1

    return pixel_img


def preprocess_text():                                                          #Preprocessing the captions (labels)
    with open(TPATH, 'r') as file:
        captions = [line for line in file]

    captions = [ele.strip().split('\t')[-1].lower() for ele in captions]
    grouped_cap = [captions[i : i + 5] for i in range(0, len(captions), 5)]

    return captions, grouped_cap


def glove_vec(glove_file):                                                      #Creating embedding matrix

    with open(glove_file, 'r') as f:
        words = set()
        word2vec = {}
        for line in f:
            line  = line.strip().split()
            word = line[0]
            words.add(word)
            word2vec[word] = np.asarray(line[1:], dtype = np.float64)

    index_to_word = {}                                                          #Tokenizing text corpus
    word_to_index = {}
    for index, word in enumerate(sorted(words)):
        word_to_index[word] = index
        index_to_word[index] = word

    return word_to_index, index_to_word, word2vec

def load_dataset():                                                             #Dataset Creation

    X = preprocess_Img(PATH, RPATH)
    y, y_grp = preprocess_text()

    return X, y, y_grp
