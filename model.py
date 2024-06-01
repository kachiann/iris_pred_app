from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define a dictionary to map class labels to their names
class_label_names = {0: "Iris setosa", 1: "Iris versicolor", 2: "Iris virginica"}

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'iris_model.pkl')

# Load the model
loaded_model = joblib.load('iris_model.pkl')

# Make predictions
predictions = loaded_model.predict(X_test)

# Print the name of the predicted class for each prediction
for i, prediction in enumerate(predictions):
    predicted_class_name = class_label_names[prediction]
    print(f"Prediction {i + 1}: {predicted_class_name}")
