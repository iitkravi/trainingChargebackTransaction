# This is a sample Python script.
import hashlib

import numpy
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Import the libraries
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pickle


class FlattenForEach:
    pass


def update_r(row):
    h = hashlib.sha256()
    h.update(row['merchant_name'].encode())
    row['merchant_name'] = int(h.hexdigest(), 16) % (10 ** 8)

    # h.update(row['merchant_id'].encode())
    # row['merchant_id'] = int(h.hexdigest(), 16)

    h.update(row['merchant_email'].encode())
    row['merchant_email'] = int(h.hexdigest(), 16) % (10 ** 8)

    h.update(row['merchant_phone'].encode())
    row['merchant_phone'] = int(h.hexdigest(), 16) % (10 ** 8)

    h.update(row['customer_name'].encode())
    row['customer_name'] = int(h.hexdigest(), 16) % (10 ** 8)

    # h.update(row['customer_card_number'].encode())
    # row['customer_card_number'] = int(h.hexdigest(), 16)

    h.update(row['customer_billing_address'].encode())
    row['customer_billing_address'] = int(h.hexdigest(), 16) % (10 ** 8)

    h.update(row['city'].encode())
    row['city'] = int(h.hexdigest(), 16) % (10 ** 8)

    h.update(row['zip_code'].encode())
    row['zip_code'] = int(h.hexdigest(), 16) % (10 ** 8)

    h.update(row['order_currency'].encode())
    row['order_currency'] = int(h.hexdigest(), 16) % (10 ** 8)

    h.update(row['order_description'].encode())
    row['order_description'] = int(h.hexdigest(), 16) % (10 ** 8)

    h.update(row['payment_authorization_code'].encode())
    row['payment_authorization_code'] = int(h.hexdigest(), 16) % (10 ** 8)

    # h.update(row['latitude'].encode())
    # row['latitude'] = int(h.hexdigest(), 16)
    #
    # h.update(row['longitude'].encode())
    # row['longitude'] = int(h.hexdigest(), 16)

    h.update(row['ip_address'].encode())
    row['ip_address'] = int(h.hexdigest(), 16) % (10 ** 8)
    return row

def train_model():
    # Load and preprocess the data
    data = pd.read_csv("data.csv")  # Replace with your CSV file name
    data = data.fillna(0)  # Fill missing values with 0
    # data = pd.get_dummies(data)  # Encode categorical variables
    X = data.drop("target", axis=1)  # Separate the features from the target
    X = data.drop("order_date", axis=1)
    X = data.drop("payment_settlement_date", axis=1)
    # X = data.drop("latitude", axis=1)
    # X = data.drop("longitude", axis=1)
    y = data["target"]  # Assign the target variable
    # X = X["merchant_name"]
    # encoder = LabelEncoder()
    # h = FeatureHasher(n_features=7, input_type='string')
    X = X[['merchant_name', 'merchant_id', 'merchant_email', 'merchant_phone', 'customer_name', 'customer_card_number',
           'customer_billing_address', 'city', 'zip_code', 'order_amount', 'order_currency', 'latitude', 'longitude',
           'order_description', 'payment_authorization_code',
           'ip_address']].apply(update_r, axis=1)
    # numpy.save('classes.npy', encoder.classes_, allow_pickle=True)
    y = y.astype("int")
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
