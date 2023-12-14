# import libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request
from joblib import load


# load model Calories Prediction
model = load('Calories_model.pkl')

# load scaler
scalerfile = 'scaler.save'
scaler = pickle.load(open(scalerfile, 'rb'))

# Load model MilkClassification
model_milk_path = 'Milk_classifier_model.pkl'
with open(model_milk_path, 'rb') as model_file_milk:
    model_milk = pickle.load(model_file_milk)

# Load the Milk dataset
filename_milk = 'milknew.csv'
dataframe_milk = pd.read_csv(filename_milk)

# Separate features (X_milk) and target variable (y_milk)
X_milk = dataframe_milk.drop(['Grade', 'pH', 'Colour'], axis=1)
y_milk = dataframe_milk['Grade']

# Convert categorical variables into numerical format using one-hot encoding
X_encoded_milk = pd.get_dummies(X_milk)

def predict_milk_grade(features):
    # Convert the input features into a DataFrame
    input_data = pd.DataFrame([features], columns=X_milk.columns)

    # One-hot encode the input data with the columns used during training
    input_data_encoded = pd.get_dummies(input_data).reindex(columns=X_encoded_milk.columns, fill_value=0)

    # Use the trained model to make predictions
    prediction = model_milk.predict(input_data_encoded)

    return prediction[0]

# Flask constructor
app = Flask(__name__)

@app.route('/')
@app.route('/home', methods=["GET"])
def home():
    # render the main dashboard (Index.html)
    return render_template('home.html')


# Route to predict milk
@app.route('/predict_milk', methods=['POST'])
def predict_milk():
    try:
        # Get input features from the form
        features = [
            float(request.form['Temprature']),
            int(request.form['Taste']),
            int(request.form['Odor']),
            int(request.form['Fat']),
            int(request.form['Turbidity'])
        ]

        # Use the model to make predictions
        predicted_grade = predict_milk_grade(features)

        print("Input features:", features)
        print("Predicted grade:", predicted_grade)

        # Map predicted class to human-readable labels
        grade_labels = {'low': 'Expired', 'medium': 'Nearly Expired', 'high': 'Healthy'}
        predicted_label = grade_labels.get(predicted_grade, 'Unknown')

        print("Predicted label:", predicted_label)

        # Return the numeric class label and the corresponding string label
        return render_template('index2.html', prediction=predicted_grade, label=predicted_label)

    except Exception as e:
        print("Error:", str(e))
        return render_template('error_page.html', error_message='An error occurred')

@app.route('/predict_milk', methods=['GET'])
def get_predict_milk():
    return render_template('index2.html')

#'/predict_calories' as a reference page for '/home'
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
