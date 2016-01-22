import numpy as np

from sklearn.cluster import KMeans

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD



def create_ann(hidden_layer_dimension, input_dimension, output_len):
    '''Implementacija vestacke neuronske mreze sa 784 neurona na uloznom sloju,
        128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    ann.add(Dense(hidden_layer_dimension, input_dim=input_dimension, activation='sigmoid'))
    ann.add(Dense(output_len, activation='sigmoid'))
    return ann
    
def train_ann(ann, input_list, output_list):
    '''Obucavanje vestacke neuronske mreze'''
    input_list = np.array(input_list, np.float32) # dati ulazi
    output_list = np.array(output_list, np.float32) # zeljeni izlazi za date ulaze
   
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(input_list, output_list, nb_epoch=1, batch_size=1, verbose = 1, shuffle=False, show_accuracy = True) 
      
    return ann
    
    
def prepareDataForAnn(games):
    
    input_list = []
    output_list = []
    for row in games:
        temp = []
        for idx,val in enumerate(row):
            if idx==1:
                output_list.append(val)
            else:
                temp.append(val)
        
        input_list.append(temp)
        

    #print input_list
        
    return input_list, output_list
    
    
def convertOutput(output_list):
    
    output = []
    for i in range(0,len(output_list)):
        if output_list[i] == 1:
            output.append([1,0,0])
        elif output_list[i] == 0:
            output.append([0,1,0])
        else:
            output.append([0,0,1])
    
    return output
