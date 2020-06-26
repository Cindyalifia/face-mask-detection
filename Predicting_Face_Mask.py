
# Clear the output such as INFO, WARNING, and ERROR messages 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt


def arg_parse():
    """
    Parse arguements to the detect module
    """

    parser = argparse.ArgumentParser(description='Face Mask Prediction')

    parser.add_argument(
        "--images", 
        dest='images', 
        required=True,
        help="Image / Directory containing images to perform detection upon", 
        type=str
    )

    return parser.parse_args()


def class_result(list_predict):
    """
    Define class of prediction result
    Param : list_predict = is the prediction result
    """
    list_predict = list_predict.tolist()
    label = "{:.2f}%".format(max(list_predict) * 100)

    # include the probability in the label
    if list_predict.index(max(list_predict)) == 0:
        return 'WithMask '+str(label)
    elif list_predict.index(max(list_predict)) == 1:
        return 'WithoutMask '+str(label)


def predict_class(img_name, model):
    """
    Predict new image
    """
    i_image = load_img(img_name, target_size=(224, 224))
    test_image = image.img_to_array(i_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image /= 255.
    result = model.predict(test_image)
    plt.imshow(i_image)
    plt.show()
    print('Predicted : ', class_result(result[0]))

args = arg_parse()

# Load model
print("[INFO] LOADING A MODEL")
classifier = load_model('MODEL/model.05-0.00.h5')

# Classify
print("[INFO] PREDICTING . . .")
predict_class(args.images, classifier)
