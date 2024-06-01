import os
import joblib
from flask import Flask, render_template, redirect, url_for
from forms import FlowerForm

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')  # Set this in your environment

# Load the pre-trained KNN model
model = joblib.load('iris_model.pkl')

# Define a dictionary to map class labels to their names
class_label_names = {0: "Iris setosa", 1: "Iris versicolor", 2: "Iris virginica"}

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = FlowerForm()
    if form.validate_on_submit():
        # Process form data here
        sepal_length = float(form.SepalLengthCm.data)
        sepal_width = float(form.SepalWidthCm.data)
        petal_length = float(form.PetalLengthCm.data)
        petal_width = float(form.PetalWidthCm.data)

        # Perform prediction using the KNN model
        prediction = predict_flower(sepal_length, sepal_width, petal_length, petal_width)

        # Get the class name corresponding to the predicted label
        predicted_class_name = class_label_names[prediction]

        return render_template('result.html', prediction=predicted_class_name)
    return render_template('predict.html', form=form)

def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    # Make a prediction using the KNN model
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)
    return prediction[0]

if __name__ == '__main__':
    app.run()
