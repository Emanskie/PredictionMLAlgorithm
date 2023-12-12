# import libraries
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, request, send_file
from joblib import load
import os
import uuid
import flask
import urllib
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img , img_to_array

# load model Calories Prediction
model = load('Calories_model.pkl')

# load scaler
scalerfile = 'scaler.save'
scaler = pickle.load(open(scalerfile, 'rb'))




# load model Classification
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classification_model = load_model(os.path.join(BASE_DIR, 'imagepredict.hdf5'))

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def make_prediction(filename, model):
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)

    img = img.astype('float32')
    img = img / 255.0
    result = model.predict(img)

    dict_result = {}
    for i in range(10):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]

    prob_result = []
    class_result = []
    for i in range(3):
        prob_result.append((prob[i] * 100).round(2))
        class_result.append(dict_result[prob[i]])

    return class_result, prob_result

# flask constructor
app = Flask(__name__)

# set '/home' as the main dashboard
@app.route('/')
@app.route('/home', methods=["GET"])
def home():
    # render the main dashboard (Index.html)
    return render_template('home.html')

# route classify animal
@app.route('/predict_classification', methods=['GET', 'POST'])
def predict_classification():
    # render the main dashboard (Index.html)
    return render_template('index2.html')


@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images')
    if request.method == 'POST':
        if (request.form):
            link = request.form.get('link')
            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                output = open(img_path, "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result, prob_result = make_prediction(img_path, model)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                }

            except Exception as e:
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if (len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index2.html', error=error)


        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = file.filename

                class_result, prob_result = make_prediction(img_path, classification_model)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if (len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index2.html', error=error)

    else:
        return render_template('index2.html')
# '/predict_calories' as a reference page for '/home'
@app.route('/predict_calories', methods=['GET', 'POST'])
def predict_calories():
    # render the reference page (result.html)
    return render_template('index.html')

# get form data Calories
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # convert string value into numeric value
        gender = 1 if request.args.get('gender') == 'Male' else 0
        age = float(request.args.get('age'))
        duration = float(request.args.get('duration'))
        heart_rate = float(request.args.get('heart_rate'))
        temp = float(request.args.get('temp'))
        height = float(request.args.get('height'))
        weight = float(request.args.get('weight'))

        # store form values into a list
        values = [gender, age, height, weight, duration, heart_rate, temp]

        # turn into array & reshape array for prediction
        input_array = np.asarray(values).reshape(1, -1)

        # scale the inputted reshaped data
        scaled_set = scaler.transform(input_array)

        # predict with inputted values
        predicted = model.predict(scaled_set)

        # display predicted values in result.html file
        return render_template('result.html', predicted_value=predicted[0])

    else:
        return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
