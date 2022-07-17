from typing import Union
import joblib
from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd


# Declare the data object with its components and their type.
class Feature(BaseModel):
    workclass: str='Private'
    education: str='HS-grad'
    marital_status: str=Field(default='Married-civ-spouse', alias='marital-status')
    occupation: str='Adm-clerical'
    relationship: str='Own-child'
    race: str='Asian-Pac-Islander'
    sex: str='Male'
    native_country: str=Field(default='Germany', alias='native-country')
    capital_loss: int=Field(default=0, alias='capital-loss')
    capital_gain: int=Field(default=5013, alias='capital-gain')
    fnlgt: int=37778
    age: int=34
    hours_per_week: int=Field(default=44, alias='hours-per-week')
    education_num: int=Field(default=9, alias='education-num')

app = FastAPI()

model_path = 'model/model.pkl'
pipeline = joblib.load(model_path)


@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


@app.post("/inference/")
async def get_inference(
        feature: Feature
                        ):
    # feature ={'education_num':education_num, 'capital_loss':capital_loss}
    feature_dic = {}
    for key, value in feature.dict().items():
        key = key.replace('_', '-')
        feature_dic[key] = [value]
    feature_df = pd.DataFrame(feature_dic)
    res = pipeline.predict(feature_df).tolist()[0]
    return res

if __name__ == "__main__":
    uvicorn.run('main:app', debug=True, reload=True)