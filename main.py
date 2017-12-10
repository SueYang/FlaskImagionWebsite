import sys
import os

sys.path.insert(1, os.path.join(os.path.abspath("."), './venv/Lib/site-packages'))

import result
import model
import jsonpickle
from flask import Flask, render_template, request, url_for, send_from_directory, json, redirect, flash
from werkzeug.utils import secure_filename
import numpy as np
# import cv2
from PIL import Image

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploaded')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])
MODEL_WEIGHTS_FOLDER = os.path.join(APP_ROOT, 'modelweights')

app = Flask(__name__)
app.config['APP_ROOT'] = APP_ROOT
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_WEIGHTS_FOLDER'] = MODEL_WEIGHTS_FOLDER


model_ob = model.Model(app)
global imagion_model
imagion_model = model_ob.get_model()
# To show directory folder, enable below two lines. For debugging
# from flask_autoindex import AutoIndex
# AutoIndex(app, browse_root=os.path.curdir)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/picture_single')
def picture_single():
    return render_template('portfolio-single.html')


@app.route('/upload_picture', methods=['GET', 'POST'])
def upload_picture():
    reslist = []
    if request.method == 'POST':
        files = request.files.getlist('uploadimgs')
        for imgfile in files:
            if imgfile and allowed_file(imgfile.filename):
                filename = secure_filename(imgfile.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                imgfile.save(filepath)
                score = get_prediction(filepath)   # model is global variable declared and loaded when initializing app
                res = result.Result(filename, score)
                # d = {"name": filename, "rank": str(rank), "score": str(score)}
                # d = [filename, str(rank), str(score)]
                reslist.append(res)
                # executor.submit(get_score(imgfile))
        # sort by score in descending order
        reslist.sort(key=lambda x: x.score, reverse=True)
        return jsonpickle.encode(reslist), 200, {'ContentType': 'application/json'}
    else:
        return render_template('upload-picture.html')


# def build_model():
#     # If you want to specify input tensor
#     input_tensor = Input(shape=(160, 160, 3))
#     vgg_model = applications.VGG16(weights='imagenet',
#                                    include_top=False,
#                                    input_tensor=input_tensor)
#
#     # Creating dictionary that maps layer names to the layers
#     layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
#
#     # Getting output tensor of the last VGG layer that we want to include
#     x = layer_dict['block4_pool'].output
#
#     # Stacking a new simple convolutional network on top of it
#     x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Flatten()(x)
#     x = Dense(256, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(10, activation='softmax')(x)
#
#     # my_model = pickle.load(open("file_predict.pkl", "rb"))
#     # my_model = model_from_json(json_data)
#     # Creating new model. Please note that this is NOT a Sequential() model.
#     from keras.models import Model
#     custom_model = Model(inputs=vgg_model.input, outputs=x)
#
#     # Make sure that the pre-trained bottom layers are not trainable
#     for layer in custom_model.layers[:15]:
#         layer.trainable = False
#
#     # Do not forget to compile it
#     custom_model.compile(loss='categorical_crossentropy',
#                          optimizer='rmsprop',
#                          metrics=['accuracy'])
#     return custom_model
#
#
# def model_weights_path():
#     return os.path.join(app.config['MODEL_WEIGHTS_FOLDER'], MODEL_WEIGHTS_NAME)
#
#
# def get_model():
#     custom_model = build_model()
#     weights_path = model_weights_path()
#     custom_model.load_weights(weights_path)
#     return custom_model


def get_prediction(filepath):
    # # using cv2(GCE)
    # img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    # img = cv2.resize(img, (100, 100))
    # # save for result display
    # imgtosave = cv2.resize(img, (200, 200))
    # cv2.imwrite(filepath, imgtosave)
    # img = img/255
    # img = np.expand_dims(img, axis=0)

    # using PIL(Windows local testing)decode image
    img = Image.open(filepath)
    print(img)
    img = img.resize((100, 100), Image.ANTIALIAS)
    img.save(filepath)
    img = np.asarray(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # do some fancy processing here....
    # np array goes into nueral network, prediction comes out
    # print(img.shape)
    y_hat = imagion_model.predict(img)[0][0]
    return round(float(y_hat*10.0),1)


@app.route('/blog')
def blog():
    return render_template('blog.html')


@app.route('/blog_single')
def blog_single():
    return render_template('blog-single.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/uploads')
def upload_path():
    return app.config['UPLOAD_FOLDER']


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/image-score', methods=['GET', 'POST'])
def image_score():
    # name = request.args.get("name")
    # # path = request.args.get("path")
    # rank = request.args.get("rank")
    # score = request.args.get("score")
    # print("########",name, rank, score)
    reslist = request.args.get("dictlist")
    # reslist = jsonpickle.decode(resjson)
    # print(type(reslist))
    # print("*********************")
    # return render_template('image-score.html', name=name, rank=rank, score=score)
    return render_template('image-score.html', reslist=reslist)


@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)


def clear_directory(dirPath):
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)


if __name__ == '__main__':
    # create a list of dicts to store analysis result for each picture
    # global resultdicts   # using global variable is bad practice. Need to refine code structure later
    # resultdicts = []
    app.run()

