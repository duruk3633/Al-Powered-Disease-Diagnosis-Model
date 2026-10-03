import numpy as b
import tensorflow as tf
from tensorflow import keras
import pyrebase


model = tf.keras.models.load_model("akciger.h5")


config = {
"apiKey": "AIzaSyBQb2gq-1Nh2UqvDNaY5eet_4sinwsArms",
"authDomain": "uploads-b37cd.firebaseapp.com",
"databaseURL": "https://uploads-b37cd-default-rtdb.firebaseio.com/",
"storageBucket": "gs://uploads-b37cd.appspot.com/",
}



firebase = pyrebase.initialize_app(config) 
CATEGORIES = ["normal", "tuberkulos","zature","covid-19"]
firebase.storage.download("images/ornek.jpg","ornek.jpg")
ORNEK = r"ornek.jpg"
img_height= 180
img_width = 180
img = tf.keras.utils.load_img(
 ORNEK, target_size=(img_height, img_width)
  )
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
hastalikyazi = "Bu fotograf su hastaliga benziyor: {}".format(CATEGORIES[b.argmax(score)])
print(hastalikyazi)
