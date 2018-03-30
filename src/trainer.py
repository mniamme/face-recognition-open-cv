import os
import cv2
import numpy as np
from PIL import Image

path = 'dataset/'

def getImages():
    images_paths = os.listdir(path)
    faces = []
    ids = []

    for image_path in images_paths:

        face_img = Image.open(path + image_path).convert('L')

        # get np array of the image
        face_np = np.array(face_img, 'uint8')

        # get the username of the person in the image
        ID = image_path.split('_')[2]

        # add the face and the username to the database
        faces.append(face_np)
        ids.append(int(ID))

        # show the training
        cv2.imshow('Training', face_np)
        cv2.waitKey(10)
        
    return faces, np.array(ids)



# get faces and names
faces, ids = getImages()

# train the program
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, ids)
recognizer.save('recognizer/training_data.yml')

# close the training program
cv2.destroyAllWindows()
print 'End of Training'
