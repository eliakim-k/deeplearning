################################################
### Exemple d'utilisation du module keras_facile
### Cas d'une variable
################################################

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


# Importer le module
from keras_facile import *

# Theoreme de l'approximation universelle

# Fonction à approcher
def f(x):
    return np.cos(2*x) + x * np.sin(3*x) + x**0.5

# Intervalle et nombre des divisons
a = 2
b = 10
n = 100

# Architecture du réseau
modele = Sequential()
modele.add(Dense(2*n, input_dim=1, activation=heaviside))
modele.add(Dense(1, activation='linear'))

calcul_approximation(modele, f, a, b, n)
affichage_approximation(modele, f, a, b)
