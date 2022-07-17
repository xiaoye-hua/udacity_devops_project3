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
from starter.ml.model import train_model, compute_model_metrics, inference, compute_sliced_model_metrics

logging.basicConfig(level='INFO',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='../slice_output.txt'
                    )
console = logging.StreamHandler()
logging.getLogger().addHandler(console)
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
logging.info(f"Train data info: ")
logging.info(X_train[cat_features+passthrough_features].info())
# for col in cat_features+passthrough_features:
#     logging.info(f"{col}:")
#     logging.info(X_train[col].unique()[:3])
pipeline.fit(X=X_train, y=y_train)

X_test['predict'] = pipeline.predict(X=X_test)
    # inference(model=model, X=X_test)

precision, recall, fbeta = compute_model_metrics(y=y_test, preds=X_test['predict'])
logging.info(f"Metric for all data:")
logging.info(f"{precision}, {recall}, {fbeta}")


for col in cat_features:
    compute_sliced_model_metrics(col=col, df=X_test, y_df=y_test)


file_name = joblib.dump(
    value=pipeline,
    filename=model_dir
)[0]
logging.info(f"Saving pipeline to {file_name}")
