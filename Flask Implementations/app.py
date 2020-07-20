from flask import Flask, request, render_template
from joblib import load
import pandas as pd

model = load('randomforestmodel.joblib')

app = Flask('app')

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        try:
            Company = str(request.form['Company'])
            fueltype = str(request.form['fueltype'])
            mpg = float(request.form['mpg'])
            engineSize = float(request.form['engineSize'])
            mileage = int(request.form['mileage'])
            tax = int(request.form['tax'])
            transmission = str(request.form['transmission'])
            year = int(request.form['year'])

            result = model.predict(
                pd.DataFrame([[Company, engineSize, fueltype, mileage, mpg, tax, transmission, year]],
                             columns=['company', 'engineSize', 'fuelType', 'mileage', 'mpg', 'tax',
                                      'transmission', 'year']))

            # result = model.predict([[Company, engineSize, fueltype, mileage, mpg, tax, transmission, year]])
            # print(result)
            return render_template('index.html', result=round(result[0], 2))
        except:
            result = 'Please pass proper input'
            return render_template('index.html', result=result)

    return render_template('index.html')

@app.route('/withOL', methods=['GET', 'POST'])
def with_ol():
    return render_template('your_report.html')

@app.route('/NOL', methods=['GET', 'POST'])
def n_ol():
    return render_template('nol.html')

app.run(host='localhost', port=8080)
