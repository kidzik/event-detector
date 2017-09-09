from utils import *
import argparse
import sys

parser = argparse.ArgumentParser(description='Annotate heel strike (HS) and foot off (FO) events.')
parser.add_argument('event', metavar='event', help="Type of the event to predict: heel strike (HS) or foot-off (FO)", type=str, choices=['HS','FO'])
parser.add_argument('filename', metavar='filename', help="A csv file with observed kinematic sequences", type=str)
# parser.add_argument('--method', '-m',
#                     help="",
#                     type=str, choices=['neural','velocity','coordinate'],default="neural")
args = parser.parse_args()

def derivative(traj):
    nframes = traj.shape[0]
    traj_der = traj[1:nframes,:] - traj[0:(nframes-1),:]
    return np.append(traj_der, [[0] * traj.shape[1]], axis=0)

def convert_data(data):
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

def neural_method(inputs):
    import keras
    from keras.models import load_model

    keras.losses.weighted_binary_crossentropy = weighted_binary_crossentropy
    model = load_model("models/%s.h5" % args.event)
    cols = range(15) + [15 + i for i in range(13)] + [30 + i for i in range(6)] 
    res = model.predict(inputs[:,cols].reshape((1,inputs.shape[0],len(cols))))
    peakind = peakdet(res[0], 0.7)
    print ', '.join(map(str, [k for k,v in peakind[0]]))

inputs = np.loadtxt(args.filename, delimiter=',')
X, Y = convert_data(inputs)

neural_method(X)
