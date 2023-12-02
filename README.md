# HAM10000SkinLesionDetector
Multi Class classification using CNN models on HAM10000 Skin Lesion Dataset


# Skin-Lesion-Detector-TOOLS
 
 Here Datasets must be downloaded from the Kaggle.
 Link to download datasets : https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000 , https://www.kaggle.com/discdiver/mnist1000-with-one-image-folder (contains all images in 1 folder only)
  Then extract the datasets with .csv file into same folder Skin-disease-detection
      That is : HAM10000_metadata.csv
                HAM10000_images_part_1
                HAM10000_images_part_2
                These above should be extracted in that above folder.

Required Libraries :
    Web framework : Flask 
    Tensorflow
    Matplotlib
    Keras
    Numpy
    Pandas
    Sklearn
 These above libraries are mandatory.
 
 Steps to follow :
 Step 1 : Run the ‘MobileNetV2.ipynb’ file in either Jupyter/Visual Studio Code
 Step 2 : At the final step of Training the model , save that model in the same folder in  which  the ‘app.py’ file is present.
 Step 3 : Give the path of saved Model in app.py 
 (e.g:  Model= load_model('model.h5') )
 Step 4 : Now run ‘app.py’ file to get the UI of Model. Follow the localhost link to open the User Interface in Web Browser
