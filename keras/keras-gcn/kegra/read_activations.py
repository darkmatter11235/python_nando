import keras.backend as K
import scipy.sparse as sp
import numpy as np
import pdb

def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    for layer in model.layers:
        print(layer.name)
    #print(len(inp))
    #print(inp[0].shape)
    #inp += [K.learning_phase()]
    #funcs = [K.function(inp[0] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    #funcs = [K.function(inp, [out]) for out in outputs]  # evaluation functions
    funcs = []
    index = 0
    for out in outputs:
        print( "processing outputs for layer {}".format(model.layers[index].name))
        index += 1
        print(out.shape)
        funcs.append(K.function(inp + [K.learning_phase()], [out]))
    #funcs = [K.function(inp, [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    #list_inputs[1] = K.sparse_to_dense(list_inputs[1].indices, list_inputs[1].dense_shape, list_inputs[1].values)
    list_inputs[1] = list_inputs[1].todense().astype(np.float32)
    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    index = 0
    layer_outputs = []
    print(list_inputs)
    for func in funcs:
        layer_name = model.layers[index].name
        print(model.layers[index].name)
        index += 1
        #print len(list_inputs)
        print(func)
        #layer_outputs.append(func(list_inputs)[0])
        #layer_outputs.append(func([list_inputs[0], list_inputs[1]]))
        if layer_name == 'input_2' or layer_name == 'dropout_1':
            layer_outputs.append(func([list_inputs[0]])[0])
        elif layer_name == 'input_1':
            print("Skipping input_1")
            #pass
            #layer_outputs.append(func([list_inputs[0],list_inputs[1]]))
        else:
            layer_outputs.append(func([list_inputs[0], list_inputs[1]])[0])

    #layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def display_activations(activation_maps):
    import numpy as np
    import matplotlib.pyplot as plt
    """
    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """
    batch_size = activation_maps[0].shape[0]
    #pdb.set_trace()
    print(batch_size)
    #assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        activations = activation_map
        if True:
            print("")
        elif len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.imshow(activations, interpolation='None', cmap='jet')
        plt.show()
