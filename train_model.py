import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

data = {
'Math': [78, 45, 90, 35, 65, 82, 30, 40, 55, 92],
'Science': [69, 40, 95, 30, 70, 75, 20, 45, 60, 98],
'English': [72, 39, 88, 25, 60, 85, 28, 50, 58, 93],
'Result': ['Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Pass', 'Fail', 'Fail',
'Pass', 'Pass']
}
df = pd.DataFrame(data)
df['Result'] = df['Result'].map({'Pass': 1, 'Fail': 0})
print(df)
x = df[['Math', 'Science', 'English']]
y = df['Result']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
# Save model
joblib.dump(model, 'model.pkl')