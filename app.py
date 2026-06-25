from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/')
def home():
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuels = car['fuel_type'].unique()

    return render_template(
        'index.html',
        companies=companies,
        years=years,
        fuels=fuels
    )

@app.route('/predict', methods=['POST'])
def predict():

    company = request.form['company']
    name = request.form['name']
    year = int(request.form['year'])
    fuel = request.form['fuel']
    kms = int(request.form['kms'])

    prediction = model.predict(
        pd.DataFrame(
            [[name, company, year, kms, fuel]],
            columns=['name','company','year','kms_driven','fuel_type']
        )
    )

    return str(round(prediction[0],2))

if __name__ == "__main__":
    app.run(debug=True)