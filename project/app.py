from __future__ import division, print_function
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
global graph
graph = tf.get_default_graph()
sess = tf.Session(graph=graph)
app = Flask(__name__)
model = load_model('cnnproj.h5')
print("Model loaded. Check http://127.0.0.1:5000")

@app.route('/', methods=['GET'])
def index():
    return render_template('base.html')

@app.route('/predict', methods = ['GET', 'POST'])
def upload():
    if request.method=='POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        file_path = os.path.join(basepath,'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size = (64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis = 0)
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            preds = model.predict_classes(x)
            print("prediction",preds)
        index = [' Normal', ' Pneumonia']
        text = ""+index[preds[0][0]]
        return text

if __name__ == '__main__':
    app.run(debug = True, threaded = False)     

