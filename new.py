import csv
from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
import os
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'

# Load the Keras models
soil_model = load_model("keras_Model.h5", compile=False)
crop_model = load_model("AlexNetModel.hdf5")

# Load class names for the soil type classification
class_names = open("labels.txt", "r").readlines()

# Load crop disease class names
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca', 'Grape___Leaf_blight', 'Grape___healthy',
    'Orange___Haunglongbing', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper___Bacterial_spot', 'Pepper___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites', 'Tomato___Target_Spot', 'Tomato___Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Dictionary of disease solutions
disease_solutions = {
    "Apple___Apple_scab": "Use fungicides like Captan, remove fallen leaves, and ensure proper ventilation.",
    "Apple___Black_rot": "Prune infected branches, apply copper-based fungicides, and improve tree health.",
    "Corn___Common_rust": "Use resistant corn varieties and apply fungicides like Mancozeb if necessary.",
    "Grape___Black_rot": "Apply fungicides before rainy seasons, remove affected grapes, and improve air circulation.",
    "Tomato___Late_blight": "Apply copper fungicides, remove infected leaves, and ensure dry conditions.",
    "Potato___Early_blight": "Use crop rotation, remove infected plants, and apply chlorothalonil-based sprays.",
    "Tomato___Bacterial_spot": "Avoid overhead watering, use copper-based fungicides, and plant disease-free seeds.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants, disinfect tools, and use resistant varieties.",
    "Tomato___healthy": "Your tomato plant is healthy! Maintain proper care and watering."
}

# Function to load soil data from the CSV
def load_soil_data(csv_file):
    soil_data = {}
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            soil_type = row['SoilType']
            crop_name = row['CropName']
            if soil_type not in soil_data:
                soil_data[soil_type] = {"crops": []}
            existing_crops = [crop["name"] for crop in soil_data[soil_type]["crops"]]
            if crop_name not in existing_crops:
                crop_info = {
                    "name": crop_name,
                    "revenue": row['Revenue'],
                    "pesticides": row['Pesticides'],
                    "fertilizers": row['Fertilizers'],
                    "phlevel": float(row['pHLevel']),
                    "moisture": row['Moisture']
                }
                soil_data[soil_type]["crops"].append(crop_info)
    return soil_data

# Load the soil data from CSV
soil_data = load_soil_data('soil_data.csv')

# Function to determine water recommendation
def get_water_recommendation(phlevel, moisture):
    if moisture.lower() == "dry":
        water_amount = "6-8 liters per plant per day"
    elif moisture.lower() == "moist":
        water_amount = "3-5 liters per plant per day"
    else:
        water_amount = "1-2 liters per plant per day"
    if phlevel < 6.5:
        water_comment = "Moderate watering needed due to acidic soil."
    elif 6.5 <= phlevel <= 7.5:
        water_comment = "Optimal watering schedule recommended."
    else:
        water_comment = "Water less frequently due to alkaline soil."
    return f"{water_amount}. {water_comment}"

# Function to predict soil type from an image
def predict_soil(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    img = (img / 127.5) - 1  # Normalize
    prediction = soil_model.predict(img)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    soil_type = class_name.split(" ", 1)[1]
    confidence_score = np.round(prediction[0][index] * 100, 2)
    return soil_type, confidence_score

# Function to predict crop disease and get solutions
def predict_crop_disease(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    preds = crop_model.predict(img)
    predicted_index = np.argmax(preds)
    disease_name = disease_classes[predicted_index]
    solution = disease_solutions.get(disease_name, "No specific solution found. Consult an expert.")
    return disease_name, solution

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    soil_type, confidence = predict_soil(file_path)
    recommended_crops = soil_data.get(soil_type, {"crops": []})["crops"]
    for crop in recommended_crops:
        crop["water_recommendation"] = get_water_recommendation(crop["phlevel"], crop["moisture"])
    return render_template("index.html", class_name=soil_type, confidence=confidence, crops=recommended_crops, image_path=file_path)

@app.route("/predict1", methods=["POST"])
def predict1():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    disease_name, solution = predict_crop_disease(file_path)
    return render_template("index.html", image_path=file_path, disease_name=disease_name, solution=solution)

if __name__ == "__main__":
    app.run(debug=True)