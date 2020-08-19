import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import skinematics as skin
import matplotlib.pyplot as plt
import math

#from scipy import signal
from sklearn import preprocessing, metrics
from keras import models, layers, regularizers
from calc_future_orientation import calc_future_orientation

def nop_prediction(euler_o, anticipation_time):
    return euler_o
