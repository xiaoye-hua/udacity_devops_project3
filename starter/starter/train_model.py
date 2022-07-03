# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import joblib
from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model, compute_model_metrics, inference

# Config
data_dir = '../data/census.csv'
model_dir = '../model/model.pkl'

# Add code to load in the data.
data = pd.read_csv(data_dir)
col_map = dict()
for col in data.columns:
    new_col = col.replace(' ', '')
    col_map[col] = new_col
data = data.rename(columns=col_map)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.

model = train_model(X_train=X_train, y_train=y_train)


X_test, y_test, _, _ = process_data(test, categorical_features=cat_features,
                                    label="salary",
                                    training=False,
                                    encoder=encoder,
                                    lb=lb)
y_pred = inference(model=model, X=X_test)

precision, recall, fbeta = compute_model_metrics(y=y_test, preds=y_pred)

print(f"{precision}, {recall}, {fbeta}")

file_name = joblib.dump(
    value=model,
    filename=model_dir
)[0]