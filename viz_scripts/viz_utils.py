from vis.utils import utils
from keras import activations
from keras.models import  Model
from vis.visualization import visualize_saliency,visualize_activation, visualize_cam
from keras.layers import Lambda, Reshape

# take a model1 generating function new_model_function, a position (i,j) and input, calcaluate the gradient for viz
# necessory if trained model has variable input length and can be set with parameter L
# modifier: [None, 'guided', 'relu']
def get_grad(model1,model_function,pos,seed_input,viz='saliency',modifier = 'guided', input_range=(0., 1.)):
    def new_model_function():
        return model_function(L=seed_input.shape[0])
    model2 = new_model_function()
    assert len(model1.layers) == len(model2.layers)
    for idx,layer in enumerate(model1.layers):
        assert type(model1.layers[idx]) == type(model2.layers[idx])
        model2.layers[idx].set_weights(layer.get_weights())
    model_input = model2.get_input_at(0)
#    model2.layers[-1].activation = activations.linear
#    model2 = utils.apply_modifications(model2)    
    model_output = model2.get_output_at(0)
    if len(pos) == 2:
        model_output2 = Lambda(lambda x:x[:,pos[0],pos[1],0])(model_output)
    elif len(pos)==1:
        model_output2 = Lambda(lambda x:x[:,pos[0],0])(model_output)
    model_output3 = Reshape([1])(model_output2)
    model3 = Model(model_input,model_output3)
    if viz == 'saliency':
        grads = visualize_saliency(model3, -1, filter_indices=0, seed_input=seed_input,backprop_modifier=modifier)
    elif viz == 'activation':
        grads = visualize_activation(model3, -1, filter_indices=0, input_range = input_range)
    elif viz == 'cam':
        grads = visualize_cam(model3, -1, filter_indices=0, seed_input=seed_input,backprop_modifier=modifier)
    return grads