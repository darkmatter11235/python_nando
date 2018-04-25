from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

#from kegra.layers.graph import GraphConvolution
from layers.graph import GraphConvolution
#from kegra.utils import *
from utils import *
from read_activations import get_activations, display_activations


import time
import pdb
import argparse

from tensorflow.python import debug as tf_debug
import keras.backend as K

def dump_checkpoints():
    # Change starts here
    import shutil
    import os
    # delete folder and its content and creates a new one.
    try:
        shutil.rmtree('checkpoints')
    except:
        pass
    os.mkdir('checkpoints')
                                                                                                                           
def main(debug=False, dataset='sch2graph'):
 
    # Define parameters
    DATASET = dataset
    if DATASET == 'sch2graph':
        PATH = 'data/'
        PREFIX = 'dly_cell'
        PREFIX = 'mcdlycellbwcb_psd2x'
    else:
        DATASET = 'cora'
        PATH = 'data/cora/'
        PREFIX = ''
    FILTER = 'localpool'  # 'chebyshev'
    MAX_DEGREE = 2  # maximum polynomial degree
    SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
    NB_EPOCH = 200
    PATIENCE = 20  # early stopping patience

    if debug:
        sess = K.get_session()
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        K.set_session(sess)
    # Get data
    X, A, y = load_data(path=PATH,dataset=DATASET, prefix=PREFIX)
    y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y,DATASET)

    #pdb.set_trace()
    # Normalize X
    X /= X.sum(1).reshape(-1, 1)
    if FILTER == 'localpool':
        """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
        print('Using local pooling filters...')
        A_ = preprocess_adj(A, SYM_NORM)
        support = 1
        graph = [X, A_]
        #G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]
        G = [Input(shape=(None, None), batch_shape=(None, None), sparse=False)]

    elif FILTER == 'chebyshev':
        """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
        print('Using Chebyshev polynomial basis filters...')
        L = normalized_laplacian(A, SYM_NORM)
        L_scaled = rescale_laplacian(L)
        T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
        support = MAX_DEGREE + 1
        graph = [X]+T_k
        G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

    else:
        raise Exception('Invalid filter type.')

    X_in = Input(shape=(X.shape[1],))

    # Define model architecture
    # NOTE: We pass arguments for graph convolutional layers as a list of tensors.
    # This is somewhat hacky, more elegant options would require rewriting the Layer base class.
    H = Dropout(0.5)(X_in)
    H = GraphConvolution(12, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
    H = Dropout(0.5)(H)
    Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)

    # Compile model
    model = Model(inputs=[X_in]+G, outputs=Y)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

    # Helper variables for main training loop
    wait = 0
    preds = None
    best_val_loss = 99999

    #dump_checkpoints()
    #checkpoint = ModelCheckpoint(monitor='val_acc', filepath='checkpoints/model_gcn.txt',save_best_only=False) 
    # Fit
    for epoch in range(1, NB_EPOCH+1):

        # Log wall-clock time
        t = time.time()
        #pdb.set_trace()
        # Single training iteration (we mask nodes without labels for loss calculation)
        model.fit(graph, y_train, sample_weight=train_mask,
                  batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

        # Predict on full dataset
        preds = model.predict(graph, batch_size=A.shape[0])

        #print(preds)
        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                       [idx_train, idx_val])
        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_val_loss[0]),
              "train_acc= {:.4f}".format(train_val_acc[0]),
              "val_loss= {:.4f}".format(train_val_loss[1]),
              "val_acc= {:.4f}".format(train_val_acc[1]),
              "time= {:.4f}".format(time.time() - t))
        #print(len(graph))
        #X = graph[0]
        #print(X.shape)


        # Early stopping
        if train_val_loss[1] < best_val_loss:
            best_val_loss = train_val_loss[1]
            wait = 0
        else:
            if wait >= PATIENCE:
                print('Epoch {}: early stopping'.format(epoch))
                break
            wait += 1

    # Testing
    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(test_loss[0]),
          "accuracy= {:.4f}".format(test_acc[0]))
    #a = get_activations(model, graph, print_shape_only=True)  # with just one sample.
    #display_activations(a)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", type=str, default='sch2graph', help="Dataset string ('cora', 'sch2graph')")
    ap.add_argument("-d", "--debug", type=bool, default=False, help="launch tfdbg")
    args = vars(ap.parse_args())
    DATASET = args['dataset']
    DEBUG = args['debug']
    main(debug=DEBUG, dataset=DATASET)
