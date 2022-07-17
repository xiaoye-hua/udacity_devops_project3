# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import logging

# Add the necessary imports for the starter code.
import pandas as pd
import joblib
from xgboost import XGBClassifier
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

logging.basicConfig(level='INFO',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    # filename=os.path.join(log_dir, log_file)
                    )
# Config
data_dir = '../data/census.csv'
model_dir = '../model/model.pkl'
label = 'salary'

# Add code to load in the data.
data = pd.read_csv(data_dir, sep=', ')

col_map = dict()
for col in data.columns:
    new_col = col.replace(' ', '') #.replace('-', '_')
    col_map[col] = new_col
data = data.rename(columns=col_map)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
def map_label(row):
    value = row[label]
    if value == ' >50K':
        return 1
    else:
        return 0
        # [' <=50K' ]


data[label] = data.apply(lambda row: map_label(row), axis=1)

feature_col = list(set(data.columns) - set([label]))

X_train, X_test, y_train, y_test = train_test_split(data[feature_col], data[label], test_size=0.20)

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
passthrough_features = [
 'capital-loss', 'capital-gain', 'fnlgt', 'education-num', 'age', 'hours-per-week'
]


data_transformer = ColumnTransformer(
    transformers=[
        ('one_hot', OneHotEncoder(handle_unknown='ignore'), cat_features)
        , ('passthrough', 'passthrough', passthrough_features)
    ],
    remainder='drop'
)

model = XGBClassifier()

pipeline = Pipeline([
    ('transformer', data_transformer),
    ('model', model)
])

logging.info(f"Pipeline info:")
logging.info(pipeline)

# X_train, y_train, encoder, lb = process_data(
#     train, categorical_features=cat_features, label="salary", training=True
# )


# Train and save a model.

# model = train_model(X_train=X_train, y_train=y_train)


# X_test, y_test, _, _ = process_data(test, categorical_features=cat_features,
#                                     label="salary",
#                                     training=False,
#                                     encoder=encoder,
#                                     lb=lb)
logging.info(f"Train data info: ")
logging.info(X_train[cat_features+passthrough_features].info())
for col in cat_features+passthrough_features:
    logging.info(f"{col}:")
    logging.info(X_train[col].unique()[:3])
pipeline.fit(X=X_train, y=y_train)

y_pred = pipeline.predict(X=X_test)
    # inference(model=model, X=X_test)

precision, recall, fbeta = compute_model_metrics(y=y_test, preds=y_pred)

print(f"{precision}, {recall}, {fbeta}")

file_name = joblib.dump(
    value=pipeline,
    filename=model_dir
)[0]