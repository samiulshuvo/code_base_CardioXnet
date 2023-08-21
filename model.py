from keras.models import Model
from keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten, LSTM, Input, concatenate, Reshape
from keras.layers.wrappers import Bidirectional
import keras
# sampling rate
Fs = 2000

def makeConvLayers(inputLayer):
    # two conv-nets in parallel for feature learning, 
    # one with fine resolution another with coarse resolution    
    # network to learn fine features
    convFine = Conv1D(filters=32, kernel_size=int(Fs/2), strides=int(Fs/16), padding='same', activation='relu', name='fConv1')(inputLayer)
    convFine = MaxPool1D(pool_size=8, strides=8, name='fMaxP1')(convFine)
    convFine = Dropout(rate=0.5, name='fDrop1')(convFine)
    convFine = Conv1D(filters=32, kernel_size=8, padding='same', activation='relu', name='fConv2')(convFine)
    convFine = Conv1D(filters=64, kernel_size=8, padding='same', activation='relu', name='fConv3')(convFine)
    convFine = Conv1D(filters=64, kernel_size=8, padding='same', activation='relu', name='fConv4')(convFine)
    convFine = MaxPool1D(pool_size=2, strides=2, name='fMaxP2')(convFine)
    fineShape = convFine.get_shape()
    convFine = Flatten(name='fFlat1')(convFine)
    
    # network to learn coarse features
    convCoarse = Conv1D(filters=16, kernel_size=Fs*4, strides=int(Fs/2), padding='same', activation='relu', name='cConv1')(inputLayer)
    convCoarse = MaxPool1D(pool_size=2, strides=2, name='cMaxP1')(convCoarse)
    convCoarse = Dropout(rate=0.5, name='cDrop1')(convCoarse)
    convCoarse = Conv1D(filters=16, kernel_size=6, padding='same', activation='relu', name='cConv2')(convCoarse)
    convCoarse = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu', name='cConv3')(convCoarse)
    convCoarse = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu', name='cConv4')(convCoarse)
    convCoarse = MaxPool1D(pool_size=2, strides=2, name='cMaxP2')(convCoarse)
    coarseShape = convCoarse.get_shape()
    convCoarse = Flatten(name='cFlat1')(convCoarse)
    
    outLayer = Reshape((113,27,1))(inputLayer)
    #outLayer = Reshape((800,20,3))(inputLayer)
    #conv_5 = Conv2D(96, (5, 5), activation='relu', padding='same')(conv_4)
    conv_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(outLayer)
    x_inputi = BatchNormalization()(conv_1)
    conv_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x_inputi)
    conv_3 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_2)
    batchnorm_3 = BatchNormalization()(conv_3)
    x_inputi= MaxPooling2D(pool_size=(2, 2))(batchnorm_3)
    conv_4 = Conv2D(16, (3, 3), activation='relu', padding='same')(x_inputi)
    #conv_5 = Conv2D(96, (5, 5), activation='relu', padding='same')(conv_4)
    batchnorm_5 = BatchNormalization()(conv_4)
    x_inputi= MaxPooling2D(pool_size=(2, 2))(batchnorm_5)
    x_in1i = Conv2D(filters = int(x_inputi.shape[-1]//4), kernel_size = (1,1), strides = (1,1), padding = 'valid', activation = 'relu')(x_inputi)
    x_in1i=BatchNormalization()(x_in1i)
    x_in2i = Conv2D(filters = int(x_inputi.shape[-1]//2), kernel_size = (1,1), strides = (1,1), padding = 'valid', activation = 'relu')(x_in1i)
    x_in2i=BatchNormalization()(x_in2i)
    x_in3i = Conv2D(filters = int(x_inputi.shape[-1]//2), kernel_size = (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x_in1i)
    x_in3i=BatchNormalization()(x_in3i)
    out = Concatenate() ([x_in2i, x_in3i])
    convCoarse1 = Flatten(name='Flat')(out)
    # concatenate coarse and fine cnns
    mergeLayer = concatenate([convFine, convCoarse ,convCoarse1], name='merge')
    
    return mergeLayer, (coarseShape, fineShape)

def preTrainingNet(n_feats, n_classes):
    inLayer = Input(shape=(n_feats,1), name='inLayer')
    mLayer, (cShape, fShape) = makeConvLayers(inLayer)
    outLayer1 = Dropout(rate=0.5, name='mDrop3')(mLayer)
    
    # this is the network that learns temporal dependencies using LSTM
    # merge the outputs of last layers
    # reshape because LSTM layer needs 3 dims (None, 1, n_feats)

    outLayer = Reshape((1,2816))(outLayer1)
    outLayer = Bidirectional(LSTM(64, activation='relu', dropout=0.5, name='bLstm1'))(outLayer)
    outLayer = Reshape((1, int(outLayer.get_shape()[1])))(outLayer)
    outLayer = Bidirectional(LSTM(32, activation='relu', dropout=0.5, name='bLstm2'))(outLayer)
    out = Dense(64, activation='relu', name='ayer')(outLayer)
    outLayer1 = Dropout(rate=0.5, name='op3')(mLayer)
    outLayer = Dense(n_classes, activation='softmax', name='outLayer')(outLayer1)
    
    network = Model(inLayer, outLayer)
    network.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #network.compile(loss='mean_squared_error', optimizer='adadelta')
    
    return network
