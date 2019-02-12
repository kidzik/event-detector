from eventdetector.utils import *
import sys
import urllib

import matplotlib.pyplot as plt
import numpy as np

import sys
#sys.path.append("/usr/local/lib/python2.7/dist-packages")
#import btk
from ezc3d import c3d
import re
import os
import keras
from keras.models import load_model
keras.losses.weighted_binary_crossentropy = weighted_binary_crossentropy

def derivative(traj, nframes):
    traj_der = traj[:,1:nframes] - traj[:,0:(nframes-1)]
    zeros = np.array([[0],[0],[0],[0]])
    
    return np.concatenate((traj_der, zeros),axis=1)

def extract_kinematics(leg, filename_in):
    print("Trying %s" % (filename_in))
    
    # Open c3d and read data
    c = c3d(filename_in)
#    reader = btk.btkAcquisitionFileReader() 
#    reader.SetFilename(filename_in)
#    reader.Update()
#    acq = reader.GetOutput()
    nframes = c['parameters']['POINT']['FRAMES']['value'][0]
    first_frame = c['header']['points']['first_frame']
    rate = int(c['parameters']['POINT']['RATE']['value'][0])
    if rate != 120:
        return

    # We extract only kinematics
    kinematics = ["HipAngles", "KneeAngles", "AnkleAngles", "PelvisAngles", "FootProgressAngles"]
    markers = ["ANK", "TOE", "KNE", "ASI", "HEE"]
    
    # Cols
    # 2 * 5 * 3 = 30  kinematics
    # 2 * 5 * 3 = 30  marker trajectories
    # 2 * 5 * 3 = 30  marker trajectory derivatives
    # 3 * 3 = 9       extra trajectories

    outputs = np.array([[0] * nframes, [0] * nframes]).T

    print(c['data']['points'])
    print(c['data']['points'].shape)
    print(len(c['parameters']['POINT']['LABELS']['value']))

    labels = (c['parameters']['POINT']['LABELS']['value'])
    print(nframes)
    
    # Check if there are any kinematics in the file
    brk = True
    try:
        labels.index("L" + kinematics[0])
    except ValueError:
        print("No kinematics in %s!" % (filename,))
        return

    # Combine kinematics into one big array
    opposite = {'L': 'R', 'R': 'L'}
    angles = [None] * (len(kinematics) * 2)
    for i, v in enumerate(kinematics):
        idx = labels.index(leg + v)
        angles[i] = c['data']['points'][:,idx,:]
        idx = labels.index(opposite[leg] + v)
        angles[len(kinematics) + i] = c['data']['points'][:,idx,:]
    
    # Get the pelvis
    idx = labels.index("LASI")
    LASI = c['data']['points'][:,idx,:]
    idx = labels.index("RASI")
    RASI = c['data']['points'][:,idx,:]
    midASI = (LASI + RASI) / 2
    # incrementX = 1 if midASI[100][0] > midASI[0][0] else -1

    traj = [None] * (len(markers) * 4 + 3)
    for i, v in enumerate(markers):
        try:
            idx = labels.index(leg + v)
            traj[i] = c['data']['points'][:,idx,:] - midASI
            idx = labels.index(opposite[leg] + v)
            traj[len(markers) + i] = c['data']['points'][:,idx,:] - midASI
        except ValueError:
             print("Error while reading marker data %d, %s: No marker %s" % (i,v,v))
             return

        traj[i][0,:] = traj[i][0,:] #* incrementX
        traj[len(markers) + i][0,:] = traj[len(markers) + i][0,:] #* incrementX
        traj[i][2,:] = traj[i][2,:] #* incrementX
        traj[len(markers) + i][2,:] = traj[len(markers) + i][2,:] #* incrementX

    for i in range(len(markers)*2):
        traj[len(markers)*2 + i] = derivative(traj[i], nframes) 
        
    midASI = midASI #* incrementX

    midASIvel = derivative(midASI, nframes)
    midASIacc = derivative(midASIvel, nframes)

    traj[len(markers)*4] = midASI
    traj[len(markers)*4 + 1] = midASIvel
    traj[len(markers)*4 + 2] = midASIacc

    print(angles)
    print(traj)
    curves = np.concatenate(angles + traj, axis=1)

    # Plot each component of the big array
    # for i in range(3 * len(kinematics)):
    #     plt.plot(range(nframes), curves[:,i])

    import pprint
    pprint.pprint(list(c))
    pprint.pprint(c['header']['events']['size'])
    pprint.pprint(list(c['header']['events']['events_time']))
    pprint.pprint(list(c['header']['events']['events_label']))
    pprint.pprint(list(c['data']['points']))
    pprint.pprint(list(c['parameters']))
    pprint.pprint(list(c['parameters']['EVENT']))
    pprint.pprint(list(c['parameters']['EVENT']['LABELS']['value']))
    pprint.pprint(list(c['parameters']['EVENT']['CONTEXTS']['value']))
#    pprint.pprint(list(c))
    
    # Add events as output
    for eventid in range(c['header']['events']['size']):
#        event in btk.Iterate(acq.GetEvents()):
        eventtime = float(c['parameters']['EVENT']['TIMES']['value'][1 + eventid * 2])
        eventframe = int(eventtime * rate)
        if eventframe  >= nframes:
            print("Event happened too far")
            break
        # if len(event.GetContext()) == 0:
        #     print("No events")
        #     break
#        if event.GetContext()[0] == leg:
        label = c['parameters']['EVENT']['LABELS']['value'][eventid]
        if label == "Foot Strike":
            outputs[eventframe, 0] = 1
        elif label == "Foot Off":
            outputs[eventframe, 1] = 1
        print("Event %s in frame %d time %f" % (label, eventframe, eventtime))
            
    if (np.sum(outputs) == 0):
        print("No events in %s!" % (filename,))
        return

    print(curves.shape)
    print(outputs.shape)
    arr = np.concatenate((curves, outputs), axis=1)

    return arr
#    print("Writig %s" % filename_out)
#    np.savetxt(filename_out, arr, delimiter=',')

def convert_data(data):
    # TODO: temporary mess
    def derivative(traj):
        nframes = traj.shape[0]
        traj_der = traj[1:nframes,:] - traj[0:(nframes-1),:]
        return np.append(traj_der, [[0] * traj.shape[1]], axis=0)

    # We assume the data has following sequences
    # joint angles (3 DOF each):
    # - hip
    # - knee
    # - ankle
    # - pelvis
    # - foot progression
    # markers positions (3 DOF each):
    # - ankle
    # - toes
    # - knee
    # - pelvis
    # - heel
    # if there are two extra columns we assume that these are binary
    # sequences of heel strike and foot off events respectively
    if data.shape[1] != 30 and data.shape[1] != 32:
        sys.exit("Wrong data format. There should be 30 columns.")

    # What we actually use are:
    # - joint angles (5 x 3)
    # - velocity of markers (5 x 3)
    # - velocity and acceleration of the pelvis (2 x 3)
    X = np.zeros( (data.shape[0], 15 + 15 + 6) )
    X[:,0:15] = data[:,0:15]
    X[:,15:30] = derivative(data[:,15:30])
    X[:,30:33] = derivative(data[:,24:27])
    X[:,33:36] = derivative(derivative(data[:,24:27]))

    Y = None
    if data.shape[1] != 32:
        Y = data[:,30:32]

    return X, Y

def neural_method(inputs, model):
    cols = range(15) + [15 + i for i in range(13)] + [30 + i for i in range(6)] 
    res = model.predict(inputs[:,cols].reshape((1,inputs.shape[0],len(cols))))
    peakind = peakdet(res[0], 0.7)
    frames = map(int, [k for k,v in peakind[0]])
    return frames

def get_models():
    if not os.path.exists("models/FO.h5"):
        print ("Model not found. Downloading...")
        try:
            os.makedirs("models")
        except:
            pass
        model_path = "https://s3-eu-west-1.amazonaws.com/kidzinski/event-detector/FO.h5"
        urllib.urlretrieve (model_path, "models/FO.h5")
        model_path = "https://s3-eu-west-1.amazonaws.com/kidzinski/event-detector/HS.h5"
        urllib.urlretrieve (model_path, "models/HS.h5")
        print ("Model downloaded!")

get_models()
modelFO = load_model("models/FO.h5")
modelHS = load_model("models/HS.h5")

def process(filename_in, filename_out):
    idxL = [(i / 3) * 3 + i  for i in range(30)]
    idxR = [3 + (i / 3) * 3 + i  for i in range(30)]

    inputs = extract_kinematics('L', filename_in)
    inputsL = inputs[:, idxL]
    inputsR = inputs[:, idxR]
    XL, YL = convert_data(inputsL)
    XR, YR = convert_data(inputsR)

    events = {}
    events[("Foot Strike","Left")] = neural_method(XR, modelFO)
    events[("Foot Strike","Right")] = neural_method(XL, modelFO)
    events[("Foot Off","Left")] = neural_method(XR, modelHS)
    events[("Foot Off","Right")] = neural_method(XL, modelHS)

    import pprint
    pprint.pprint(events)
    # TODO
    # reader = btk.btkAcquisitionFileReader() 
    # reader.SetFilename(filename_in)
    # reader.Update()
    # acq = reader.GetOutput()
    # first_frame = acq.GetFirstFrame()
    # acq.ClearEvents()
    # print(first_frame)

    # for k,v in events.items():
    #     for frame in v:
    #         frame_true = first_frame + frame
    #         fps = 120.0
    #         event = btk.btkEvent()
    #         event.SetLabel(k[0])
    #         event.SetContext(k[1])
    #         event.SetId(2 - (k[0] == "Foot Strike"))
    #         event.SetFrame(np.round(frame_true))
    #         event.SetTime(frame_true/fps)
    #         acq.AppendEvent(event)
    
    # writer = btk.btkAcquisitionFileWriter() 
    # writer.SetInput(acq)
    # writer.SetFilename(filename_out)
    # writer.Update()

    return

