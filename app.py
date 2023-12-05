from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from PIL import Image as pil_image

# Keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import Model , load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


Model= load_model('model2.h5')     


# Melanocytic nevi (nv)
# Melanoma (mel)
# Benign keratosis-like lesions (bkl)
# Basal cell carcinoma (bcc) 
# Actinic keratoses (akiec)
# Vascular lesions (vas)
# Dermatofibroma (df)

melanoma = " Melanoma is a type of skin cancer that originates from the pigment-producing cells called melanocytes. These cells are responsible for the production of melanin, the pigment that gives color to the skin, hair, and eyes. Melanoma is known for its potential to spread to other parts of the body, making early detection and treatment crucial. Treatment for melanoma typically involves surgical removal of the cancerous tissue. In cases where melanoma has spread, additional treatments such as immunotherapy, targeted therapy, chemotherapy, or radiation therapy may be recommended. The specific treatment plan depends on the stage of the melanoma, its location, and the overall health of the patient. It's important to note that early detection is key to successful treatment, and regular skin checks, especially for individuals with risk factors such as a family history of melanoma or a history of excessive sun exposure, are recommended. If you notice any changes in the color, shape, or size of moles or pigmented areas on your skin, it's essential to consult a healthcare professional promptly for evaluation."
Actinic_keratoses = " Actinic keratoses (AK) are precancerous skin lesions that typically develop on sun-exposed areas such as the face, ears, neck, scalp, chest, backs of hands, or forearms. They appear as rough, scaly patches or crusty bumps."
bcc = "Basal cell carcinoma (BCC) is the most common type of skin cancer, accounting for about 75% of all skin cancers. It is caused by sun exposure and is most common on areas of the skin that get a lot of sun, such as the face, ears, and neck. BCC is a slow-growing cancer that rarely spreads to other parts of the body. However, it can be locally destructive, and it is important to have it treated early to prevent it from disfiguring the skin."
bkl = "Benign keratosis-like lesions (BKLLs) are a group of skin lesions that resemble seborrheic keratoses, which are the most common type of benign skin growth. BKLLs are also benign, meaning they are not cancerous. However, they can be difficult to distinguish from seborrheic keratoses, and they may require biopsy to confirm the diagnosis."
df = "Dermatofibroma, also known as benign fibrous histiocytoma, is a common, harmless skin growth that typically appears on the legs, arms, or trunk. It usually presents as a small, firm, painless nodule that ranges in color from brown to red or pink. Dermatofibromas are more common in women than in men and tend to develop during adulthood."

information = {
    0:Actinic_keratoses,
    1:bcc,
    2:bkl,
    3:df,
    4:melanoma,
    5:"fff",
    6:"fff",
}

lesion_classes_dict = {
    0 : 'Actinic keratoses',
    1 : 'Basal cell carcinoma',
    2 : 'Benign keratosis-like lesions ',
    3 : 'Dermatofibroma',
    4 : 'Melanoma',
    5 : 'Melanocytic nevi',
    6 : 'Vascular lesions',
    
}



def model_predict(img_path, Model):
    img = image.load_img(img_path, target_size=(32,32,3))
  
    #img = np.asarray(pil_image.open('img').resize((120,90)))
    #x = np.asarray(img.tolist())

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = Model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path , Model)

        # Process your result for human
        

        pred_class = preds.argmax(axis=-1)         # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   
        pr = lesion_classes_dict[pred_class[0]]
        result =str(pr)         
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
