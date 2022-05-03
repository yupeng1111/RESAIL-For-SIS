import os
from collections import namedtuple


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Param = namedtuple("Param", ['test_signal', 'epoch'])
BestModel = namedtuple("BestModel", ["epoch", "fid"])
