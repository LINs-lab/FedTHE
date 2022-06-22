# -*- coding: utf-8 -*-
from pcode.local_training.base_worker import BaseWorker


def get_worker_class(conf):
    return BaseWorker(conf)
