import numpy as np
import matplotlib.pyplot as plt
import lstm
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow.keras as tf

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import plot_confusion_matrix
import pandas as pd


model = load_model('lstm_train.h5')


while True:
                print(" ")
                print("Real time Testing Started")
                print(" ")
                l=1
                while True:
                    if(l==0):
                        break
                       
                    input_string = input("enter datas separated by space : ")
                    if (input_string=="break"):
                        print("Testing Finished")
                        l=0
                        break
                    print("\n")
                    userList = input_string.split()
                    print("user list is ", userList)

                    list_of_floats = [float(item) for item in userList]

                    print(list_of_floats)

                    ynew=model.predict(np.array(list_of_floats).reshape(1, -1))
                    ynew = ynew.round()
                    print(ynew)
                    if (ynew[0]==0):
                        print(" ")
                        print("Cardiovascular disease status : Not Detected")
                        print("For the given dataset the Predicted Value is Absence of Cardiovascular disease")
                        print(" ")
                    if (ynew[0]==1):
                        print("  ")
                        print("Cardiovascular disease status : Detected")
                        print("For the given dataset the Predicted value is Presence of Cardiovascular disease")    
                        print(" ")
                        
