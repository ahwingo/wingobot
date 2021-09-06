#import tensorflow as tf
#import tensorflow.keras.backend as K
#from tensorflow.python.framework import graph_io
from tensorflow import keras
import tensorflow as tf


def keras_to_frozen_pb(model_in_path,
                       model_out_path,
                       custom_object_dict=None,
                       tensor_out_name=None,
                       tensorboard_dir=None):
    """
    Converter that transforms keras model to frozen pb model

    Args:
        model_in_path (str): Input model path (.h5)
        model_out_path (str): Output model path (dir)
        tensor_out_name (str, optional): Specified name of output tensor.
                                         If None, it will get default tensor name from keras model.
                                         Defaults to None.
        tensorboard_dir (str, optional): Output tensorboard dir path for inspecting output model graph.
                                         If None, it doesn't generate.
                                         Defaults to None.
    """

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session()
        K.set_session(sess)
        K.set_learning_phase(0)

        # load the model to graph and sess
        model = tf.keras.models.load_model(model_in_path, custom_objects=custom_object_dict)

        # get the tensor_out_name
        if tensor_out_name is None:
            if len(model.outputs) > 1:
                print(model.outputs[0].name, model.outputs[1].name)
                #raise NameError("the model has multiple output tensor. Need to specify output tensor name.")
                tensor_out_names = ["value/Tanh", "policy/Softmax"]
            else:
                tensor_out_names = [model.outputs[0].name.split(":")[0]]

        # freeze the graph
        graphdef = tf.compat.v1.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), tensor_out_names)
        graphdef = tf.compat.v1.graph_util.remove_training_nodes(graphdef)
        graph_io.write_graph(graphdef, './', model_out_path, as_text=False)

    # output tensorboard graph
    if not tensorboard_dir is None:
        tf.compat.v1.summary.FileWriter(logdir=tensorboard_dir, graph_def=graphdef)

    return tensor_out_names

def h5_to_pb(h5_file, new_output):
    model = keras.models.load_model(h5_file)
    model.save(new_output)
    print("saved...")


def h5_to_savedmodel_pb(h5_file, new_output):
    """
    Following the instructions here: https://www.tensorflow.org/guide/saved_model#saving_a_custom_model
    """
    model = keras.models.load_model(h5_file)
    tf.saved_model.save(model, new_output)


if __name__ == "__main__":
    input_keras_model = "shodan_fossa_92.h5"
    output_pb_model = "shodan_fossa.pb"
    pb_dir = "testing_pb_export_savedmodel"
    #node_out_name = keras_to_frozen_pb(input_keras_model, output_pb_model)
    #print("the output node name is:", node_out_name)
    h5_to_savedmodel_pb(input_keras_model, pb_dir)
