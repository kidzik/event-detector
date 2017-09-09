import os
import numpy as np
import math
import keras.backend as K
import matplotlib.pyplot as plt
import pickle
import time
import itertools

from scipy.ndimage.filters import gaussian_filter
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, TimeDistributed, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from itertools import product
from random import randint
from scipy import signal

import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array


# Build the model
def construct_model(hidden = 32, lstm_layers = 2, input_dim = 15, output_dim = 1):
    model = Sequential()
    model.add(LSTM(input_shape = (input_dim,),input_dim=input_dim, output_dim=hidden, return_sequences=True))
    for i in range(lstm_layers-1):
        model.add(LSTM(output_dim = hidden, return_sequences=True))
    model.add(TimeDistributed(Dense(output_dim, activation='sigmoid')))
    model.compile(loss=weighted_binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    return model

def weighted_binary_crossentropy(y_true, y_pred):
    a1 = K.mean(np.multiply(K.binary_crossentropy(y_pred[0:1,:], y_true[0:1,:]),(y_true[0:1,:] + 0.01)), axis=-1)
#    a2 = K.mean(np.multiply(K.binary_crossentropy(y_pred[1:2,:], y_true[1:2,:]),(y_true[1:2,:] + 0.01)), axis=-1)
#    a1 = K.mean(np.multiply(K.binary_crossentropy(y_pred, y_true),(y_true + 0.01)), axis=-1)
    return a1 #+ a2

# Build the model
def construct_model(hidden = 32, lstm_layers = 2, input_dim = 15, output_dim = 2):
    model = Sequential()
    model.add(LSTM(input_shape = (input_dim,),input_dim=input_dim, output_dim=hidden, return_sequences=True))
    for i in range(lstm_layers-1):
        model.add(LSTM(output_dim = hidden / 2**i, return_sequences=True))
    model.add(TimeDistributed(Dense(output_dim, activation='sigmoid')))
    model.compile(loss=weighted_binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    return model

def plot_history(history):
    nepoch = len(history.history['loss'])
    plt.plot(range(nepoch),history.history['loss'],'r')
    plt.plot(range(nepoch),history.history['val_loss'],'b')
    axes = plt.gca()
    axes.set_ylim([0.001,0.005])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

def load_file(filename, input_dim, output_dim, nseqlen = 128):
    try:
        R = np.loadtxt(filename, delimiter=',')
    except:
        return None

    # find first event
    positives1 = np.where(R[:,input_dim] > 0.5)
    positives2 = np.where(R[:,input_dim + 1] > 0.5)
    if len(positives1[0]) == 0 or len(positives2[0]) == 0:
        return None
    nstart = max(positives1[0][0], positives2[0][0])
    nstart = nstart - randint(15,nseqlen / 2)

    if R.shape[0] < (nstart + nseqlen):
        return None

    X = R[nstart:(nstart + nseqlen),0:input_dim]
    Y = R[nstart:(nstart + nseqlen),input_dim:(input_dim + output_dim)]

    # Y = gaussian_filter(Y * 1.0, 1.0)

    if (not Y.any()):
        return None

    if R[0,90] > R[R.shape[1]-1,90]:
        cols = [ i for i in range(30,99) if (i % 3) == 0 or (i%3)==2]
        X[:,cols] = -X[:,cols]

    return X, Y.astype(int)[:,0:output_dim]
    

def load_data(fdir, input_dim, output_dim, nseqlen, nsamples = 100000):
    files = os.listdir(fdir)

    # Merge inputs from different files together
    ids = []
    inputs = np.zeros((len(files), nseqlen, input_dim))
    outputs = np.zeros((len(files), nseqlen, output_dim))

    n = 0
    for i,filename in enumerate(files):
        fname = "%s/%s" % (fdir, filename)

        data = load_file(fname, input_dim, output_dim, nseqlen)
        if not data:
            continue
        X, Y = data
        inputs[n,:,:] = X
        outputs[n,:,:] = Y
        
        ids.append(filename)
        n = n + 1
        
        if n >= nsamples:
            break
    
    return inputs[0:n,:,:], outputs[0:n,:,:], ids

def peak_cmp(annotated, predicted):
    dist = []
    if len(predicted) == 0 or len(annotated) == 0:
        return -1
    if len(predicted) != len(annotated):
        return -1
    
    for a in annotated:
#        if a > 120:
#            continue
        dist = dist + [min(np.abs(predicted - a))]
    if not len(dist):
        return -1
    return min(dist)


def eval_prediction(likelihood, true, patient, plot = True, shift = 0):
    sdist = []
    
    peakind = peakdet(likelihood[:,0],0.5)
    for k,v in peakind[0]:
        if plot:
            plt.axvline(x=k)
    sdist.append(peak_cmp(np.where(true[:,0] > 0.5)[0], [k + shift for k,v in peakind[0]]))
        
#    peakind = peakdet(likelihood[:,1],0.5)
#    for k,v in peakind[0]:
#        if plot:
#            plt.axvline(x=k)
#    sdist.append(peak_cmp(np.where(true[:,1] > 0.5)[0], [k for k,v in peakind[0]]))

    if plot:
        plt.plot(likelihood) # continous likelihood process
        plt.plot(true) # spikes on events
        plt.title(patient)
        axes = plt.gca()
        axes.set_xlim([0,true.shape[0]])
        plt.show()
    return sdist

def plot_stats(sdist):
    plt.hist(sdist,100,[0, 100])
    filtered = [k for k in sdist if k >= 0]
    
    def off_by(threshold, filtered):
        ob = [k for k in filtered if k <= threshold]
        nel = float(len(filtered))
        print("<= %d: %f" % (threshold, len(ob) / float(nel)))
        
    
    print("Error distribution:")
    off_by(1, filtered)
    off_by(3, filtered)
    off_by(5, filtered)
    off_by(10, filtered)
    off_by(60, filtered)
    print("Mean distance: %f" % (np.mean(filtered)))
 
def plot_kinematics(filename, fdir="", ids = None, fromfile=False, input_dim = 15, output_dim = 15, model = None, cols = None):
    if not fromfile:
        ntrial = ids.index(filename)
        X = inputs[ntrial,:,cols]
        Y = outputs[ntrial,:,0:output_dim]
    else:
        R = np.loadtxt("%s/%s" % (fdir, filename), delimiter=',')
        X = R[:,cols]
        Y = R[:,input_dim:(input_dim + output_dim)]        

    likelihood = model.predict(X.reshape((1,-1,len(cols))))[0]

    pylab.rcParams['figure.figsize'] = (5, 4)
    eval_prediction(likelihood, Y, filename)
    pylab.rcParams['figure.figsize'] = (15, 20)

    print("Kinematics of %s" % (filename))
    for i in range(15):
        ax = plt.subplot(5,3,1+i)
        ax.plot(X[:,i])
        ax.set_xlim([0,X.shape[0]])
        for x in np.where(Y[:,0] > 0.5)[0]:
            plt.axvline(x=x, color='g', linewidth=2)
#        for x in np.where(Y[:,1] > 0.5)[0]:
#            plt.axvline(x=x,color="r")

    plt.show()

