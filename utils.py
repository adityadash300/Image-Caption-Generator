TPATH = 'C:/Users/Aditya Dash/Documents/GitHub/Image-Caption-Generator/Dataset/Captions/descriptions.txt'

def preprocess_text():
    with open(PATH, 'r') as file:
        captions  = [line for line in file]
        captions  = [ele.strip().split('\t')[-1] for ele in captions]

<<<<<<< HEAD
        
=======
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

def load_dataset():





TPATH = 'C:/Users/Aditya Dash/Documents/GitHub/Image-Caption-Generator/Dataset/Captions/descriptions.txt'

def preprocess_text():
    with open(TPATH, 'r') as file:
        captions = [line for line in file]
        captions = [ele.strip().split('\t')[-1] for ele in captions]

    
>>>>>>> 23a8ba91ea1b96994bdafb9abe4b7735a55db7e3
