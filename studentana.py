import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = {
    'study_hours': [1,2,3,4,5,6,7,8,9,10],
    'attendance': [50,55,60,65,70,75,80,85,90,95],
    'assignments': [1,1,2,2,3,3,4,4,5,5],
    'total_marks': [20,25,35,40,50,60,70,78,88,95]
}
df = pd.DataFrame(data)
X = df[['study_hours', 'attendance', 'assignments']]
y = df['total_marks']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_test.values)
print(y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mse)
print(r2)
new_student = [[7, 82, 4]]
predicted_marks = model.predict(new_student)
print(predicted_marks)
plt.scatter(y_test, y_pred)
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)]
)
plt.plot(y_test.values, y_pred, marker='o')
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.show()
plt.show()