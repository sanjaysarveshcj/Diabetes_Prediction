import joblib
import numpy as np

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

print('\nEnter the required values for the diabetes prediction below.')

preg = float(input('\nEnter Pregnancies: '))

glu = float(input('\nEnter glucose: '))

bp = float(input('\nEnter Blood Pressure: '))

st = float(input('\nEnter Skin Thickness: '))

ins = float(input('\nEnter Insulin: '))

bmi = float(input('\nEnter BMI: '))

dpf = float(input('\nEnter Diabetes Pedigree Function: '))

age = float(input('\nEnter age: '))

data = []

data.extend([preg, glu, bp, st, ins, bmi, dpf, age])

new_data = np.array([data])

new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)

if prediction == 1:
    print("\nThe person is likely to have diabetes.\n")
else:
    print("\nThe person is unlikely to have diabetes.\n")
