#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    test_train_model.py
# @Author:      Hua Guo
# @Time:        17/07/2022 16:27
# @Desc:
from starter.ml.model import compute_model_metrics


def test_compute_model_metrics():
    y = [0, 0, 0]
    preds = [0, 0, 0]
    precision, recall, fscore = compute_model_metrics(y=y, preds=preds)
    assert precision == 1
    assert recall == 1


def test_compute_model_metrics2():
    y = [0, 0, 0]
    preds = [1, 1, 1]
    precision, recall, fscore = compute_model_metrics(y=y, preds=preds)
    assert precision == 0
    assert recall == 1
    assert fscore == 0


def test_compute_model_metrics3():
    y = [0, 0, 0]
    preds = [1, 0, 1]
    precision, recall, fscore = compute_model_metrics(y=y, preds=preds)
    assert precision == 0
    assert recall == 1



# def