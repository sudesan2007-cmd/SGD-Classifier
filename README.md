# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load & Split — Load the Iris dataset into a DataFrame and split into train/test sets (80/20).
2. Preprocess — Standardize features using StandardScaler to normalize the data.
3. Train — Fit the SGDClassifier with modified_huber loss on the training data.
4. Evaluate — Predict on test data, print accuracy, classification report, confusion matrix heatmap, and predict for new input.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SUDESAN T 
RegisterNumber: 212225240161
*/
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sgd_clf = SGDClassifier(loss='modified_huber', max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
print("Predicted Y values:", y_pred)
print("Actual Y values:", np.array(y_test))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

xnew = np.array([[5.1, 3.5, 1.4, 0.2]])
xnew = scaler.transform(xnew)
y_prednew = sgd_clf.predict(xnew)
print("Predicted species for new input:", iris.target_names[y_prednew])
```

## Output:

<img width="914" height="688" alt="{86510004-066B-415A-A88C-6B2955D232C3}" src="https://github.com/user-attachments/assets/4f59dd39-11e9-4397-aa7e-cff02a42b95a" />

<img width="1724" height="641" alt="{2FF8E779-8A6E-4FBF-BE65-751E4FB47151}" src="https://github.com/user-attachments/assets/b65db00f-6564-4063-9895-fd87b0897595" />

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
