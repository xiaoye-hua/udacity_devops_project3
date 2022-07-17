from typing import Union
import joblib
from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd


# Declare the data object with its components and their type.
class Feature(BaseModel):
    # education_num: int=Field(default=9, alias='education-num')
    # capital_loss: int=Field(default=0, alias='capital-loss')
    a: int=3
    b: int=4


class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list]
    item_id: int

app = FastAPI()

model_path = 'model/model.pkl'
model = joblib.load(model_path)


@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


@app.post("/inference/")
async def get_inference(
        # feature: Feature
        education_num: int=9,
        capital_loss: int=0
                        ):
    feature ={'a':education_num, 'b':capital_loss}
    for key, value in feature.items():
        feature[key] = [value]
    res = model.predict(pd.DataFrame(feature)).tolist()[0]
    return res
    # return feature
# A GET that in this case just returns the item_id we pass,
# but a future iteration may link the item_id here to the one we defined in our TaggedItem.
# @app.get("/items/")
# async def get_items(item_id: int=5, count: int = 1):
#     return {"fetch": f"Fetched {count} of {item_id}"}

# This allows sending of data (our TaggedItem) via POST to the API.
# @app.get("/items/")
# async def create_item(item: TaggedItem):
#     return item

if __name__ == "__main__":
    uvicorn.run('main:app', debug=True, reload=True)