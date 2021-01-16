TPATH = 'C:/Users/Aditya Dash/Documents/GitHub/Image-Caption-Generator/Dataset/Captions/descriptions.txt'

def preprocess_text():
    with open(PATH, 'r') as file:
        captions  = [line for line in file]
        captions  = [ele.strip().split('\t')[-1] for ele in captions]

        
