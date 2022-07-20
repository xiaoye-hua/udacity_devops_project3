#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    live_post.py
# @Author:      Hua Guo
# @Time:        20/07/2022 21:29
# @Desc:

import requests

url  = 'https://udacity-devlop-project3.herokuapp.com/inference/'
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
response = requests.post(url=url, json=data)

print(response.status_code)
print(response.json())