# -*- coding: utf-8 -*-
# @File    : test_fastapi.py
# @Author  : Hua Guo
# @Disc    :
import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_say_hello():
    r = client.get("/")
    # print(r)
    # print(r.json())
    assert r.json() == {'greeting': 'Hello World!'}
    assert r.status_code == 200


def test_inference():
    data = {
      "workclass": "Private",
      "education": "HS-grad",
      "marital-status": "Married-civ-spouse",
      "occupation": "Adm-clerical",
      "relationship": "Own-child",
      "race": "Asian-Pac-Islander",
      "sex": "Male",
      "native-country": "Germany",
      "capital-loss": 0,
      "capital-gain": 5013,
      "fnlgt": 37778,
      "age": 34,
      "hours-per-week": 44,
      "education-num": 9
    }
    r = client.post(url='/inference/', json=data)
    assert r.json() == 0, r.json()
    assert r.status_code == 200
#
#
def test_inference2():
    data = {
      "workclass": "State-gov",
      "education": "7th-8th",
      "marital-status": "Married-civ-spouse",
      "occupation": "Adm-clerical",
      "relationship": "Not-in-family",
      "race": "Asian-Pac-Islander",
      "sex": "Male",
      "native-country": "Germany",
      "capital-loss": 0,
      "capital-gain": 53,
      "fnlgt": 38,
      "age": 20,
      "hours-per-week": 30,
      "education-num": 3
    }
    r = client.post(url='/inference/', json=data)
    assert r.json() == 0, r.json()
    assert r.status_code == 200
