# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pickle


def train_model():
    # Load and preprocess the data
    data = pd.read_csv("data.csv")  # Replace with your CSV file name
    data = data.fillna(0)  # Fill missing values with 0
    data = pd.get_dummies(data)  # Encode categorical variables
    X = data.drop("target", axis=1)  # Separate the features from the target
    y = data["target"]  # Assign the target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit a decision tree classifier
    clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    filename = "model.pkl"
    pickle.dump(clf, open(filename, "wb"))

    # Evaluate the performance of the model on the testing set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

    # Visualize the decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=X.columns, class_names=["0", "1"], filled=True, rounded=True)
    plt.show()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
