from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('models/saved_models/saved_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        values = [float(x) for x in request.form.values()]
        final_input = np.array(values).reshape(1, -1)

        prediction = model.predict(final_input)[0]

        # Map prediction
        risk_map = {
            0: "Low Risk 🟢",
            1: "Medium Risk 🟠",
            2: "High Risk 🔴"
        }

        result = risk_map.get(prediction, "Unknown")

        return render_template('index.html', prediction_text=f'Predicted Risk: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)# Flask app will go here
