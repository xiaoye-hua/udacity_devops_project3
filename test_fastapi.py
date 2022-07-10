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
    data = {'education_num': 9, 'capital_loss': 0}
    r = client.post(url='/inference/', data=data)
    print(r.json())
    assert r.status_code == 200


def test_inference2():
    data = {'education_num': 40, 'capital_loss': 3}
    r = client.post(url='/inference/', data=data)
    print(r.json())
    assert r.status_code == 200
