# Importing the packages

import os
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
import face_recognition
from PIL import Image

image_path = '/content/content/My Drive/Our - 1st Collaborated Github project :)/Face Recognition & Matching/temp'
all_image_path = '/content/content/My Drive/Our - 1st Collaborated Github project :)/Face Recognition & Matching/train'
move_path = '/content/content/My Drive/Our - 1st Collaborated Github project :)/Face Recognition & Matching/test'

def face_recognition_function(known_image_loc, unknown_image_loc, copy_image_loc, size = 300, tolerance = 0.6):
  """
  The above function is to compare faces and return the pictures which has the face based on the tolerance.
  
  INPUT: {
      known_face_dir: This location argument should be a path to the directory where a known image is present and good to have one image of the person you want to recognize.
                      
      unknown_faces_dir: This location argument should be a path to the directory where you want to compare the faces from known_face_dir. This directory should have multiple images.
                         
      transfer_dir: This location argument should be the path to the directory where all matching images will be transfered to.
      
      size: This default argument can be specified to resize the images when they are opened as they may provide better accuracy in recognising faces.
            (takes in int values)
            
      tolerance: This default argument can be specified to measure how similar the compared faces are (i.e. 60%/ 0.6). lower the better.
                (takes in float values)
  }
  
  OUTPUT: {
        This function populates the transfer_dir with all matching imagees. 
  }
  """
  # open all images 
  for i in os.listdir(image_path):
    image = cv2.imread(os.path.join(image_path, i))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size,size))
  
    print(image.shape)
 
  # creates encoding for all faces in the original image (returns a list)
    face_image_encodings = face_recognition.face_encodings(image)

    for single_encoding in face_image_encodings:

    # for loop to iterate over all images in the target dir
     for x in os.listdir(all_image_path):

      new_image = cv2.imread(os.path.join(all_image_path, x))
      new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
      new_image = cv2.resize(new_image, (size,size))

      # encoding for faces in the target dir (returns a list)
      new_face_image_encodings = face_recognition.face_encodings(new_image)

      for each_encoding in new_face_image_encodings:

            # if cond to compare faces - currently facing an error where u cant have 2 encoding apparently.
        if face_recognition.compare_faces([single_encoding], each_encoding, tolerance = tolerance)[0]:
          
          shutil.copyfile(os.path.join(all_image_path, x), os.path.join(move_path, x))
          print('copied')
          break

        else:
            print('no face found in image or below threshold (matching %)')
            
 
face_recognition_function(image_path, all_image_path, move_path, size = 300, tolerance = 0.6)
