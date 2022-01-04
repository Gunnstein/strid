# -*- coding: utf-8 -*-
import unittest
import numpy as np

from .._fdid import *

class TestPolyReferenceLSCF(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.y = np.random.normal(size=(6, 20))
        self.fs = 1.
        self.ix_ref = [0, 3]
