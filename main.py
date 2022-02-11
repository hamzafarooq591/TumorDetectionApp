import theano
import keras

from multiprocessing.sharedctypes import Value
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model

# Creating flask app
app = Flask(__name__)

def prediction_Image(Path):
   # load model
   from tensorflow.keras.models import load_model
   model = keras.models.load_model("Tumore_model_3.h5")

   # upload_img & convert in numpy
   img_path = Path
   img = image.load_img(img_path, target_size=(224, 224))
   sh = image.img_to_array(img)
   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)
   x = preprocess_input(x)
   #predication
   preds = model.predict(x)
   pred_img = np.argmax(preds)
   # print(preds)
   import matplotlib.pyplot as plt
   # plt.imshow(sh/255)
   pic = plt.show()
   if preds[0][0] == 1:
      return "We regret to inform that Tumor is exist :("
   else:
      return "Congratulations Tumor does not exist :)"


# Default Page
@app.route('/')
def home():
   return render_template('index.html',)

# Page Form submit post request
@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']     ## Fetch image file from request
      f.save(secure_filename("Image.jpg"))  ##Saving file named as Image.jpg

      x = prediction_Image("Image.jpg")

      return render_template("index.html", value=x)    ## x varaible has a return value of prediction function.



if __name__ == '__main__':
   app.run(debug = True)