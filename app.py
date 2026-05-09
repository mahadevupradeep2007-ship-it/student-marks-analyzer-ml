import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.title("Student Marks Predictor")

data = {
    'study_hours': [1,2,3,4,5,6,7,8,9,10],
    'attendance': [50,55,60,65,70,75,80,85,90,95],
    'assignments': [1,1,2,2,3,3,4,4,5,5],
    'total_marks': [20,25,35,40,50,60,70,78,88,95]
}

df = pd.DataFrame(data)

X = df[['study_hours', 'attendance', 'assignments']]
y = df['total_marks']

model = RandomForestRegressor()
model.fit(X, y)

study = st.slider("Study Hours", 1, 10, 5)
attendance = st.slider("Attendance", 50, 100, 75)
assignments = st.slider("Assignments", 1, 5, 3)

new_data = pd.DataFrame({
    'study_hours': [study],
    'attendance': [attendance],
    'assignments': [assignments]
})

prediction = model.predict(new_data)

st.subheader("Predicted Marks")
st.write(prediction[0])

y_pred = model.predict(X)

plt.scatter(y, y_pred)

plt.plot(
    [min(y), max(y)],
    [min(y), max(y)]
)

plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")

st.pyplot(plt)