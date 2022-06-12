"""
This module defines the PolicyValue neural network, modeled after AlphaGo Zero.
"""
# Third Party
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, BatchNormalization, Reshape
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import metrics


class PolicyValueNetwork:
    """
    Instances of this class represent a WinGoBot neural network, which can be trained or used to play games.
    """
    def __init__(self, l2_const,
                 board_size=13,
                 history_length=8,
                 bot_name=None,
                 starting_network_file=None,
                 train_supervised=False,
                 train_reinforcement=False,
                 trt_mode=False):
        """
        Construct an instance of the the policy-value network used to eveluate board positions and select moves.
        :param l2_const:
        :param board_size: the size of the board (e.g. 9, 13, 19)
        :param history_length: the number of historical board states that will be presented to the network.
        :param starting_network_file:
        :param train_supervised:
        :param train_reinforcement:
        """
        self.name = bot_name
        self.board_size = board_size
        self.history_length = history_length

        self.l2_const = l2_const
        self.trt_mode = trt_mode

        self.trt_model = None  # Create this when you are in the self play process. Use for faster inference.
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
        conversion_params = conversion_params._replace(max_workspace_size_bytes=(1 << 32))
        self.conversion_params = conversion_params._replace(precision_mode="FP16")
        self.trt_inference_func = None

        # If a model is already provided, load it from the file.
        if starting_network_file:
            if train_supervised:
                self.model = load_model(starting_network_file, compile=False)
                self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                                   loss={"value": tf.keras.losses.mean_squared_error,
                                         "policy": tf.keras.losses.categorical_crossentropy},
                                   loss_weights=[0.1, 1.0],
                                   metrics=[metrics.mse, metrics.categorical_accuracy])
            elif train_reinforcement:
                self.model = load_model(starting_network_file, compile=False)
                self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9),
                                   loss={"value": tf.keras.losses.mean_squared_error,
                                         "policy": tf.keras.losses.categorical_crossentropy},
                                   loss_weights=[1.0, 1.0],
                                   metrics=[metrics.mse, metrics.categorical_accuracy])
            else:
                self.model = load_model(starting_network_file)
            # Initialize the TRT model.
            if trt_mode:
                self.save_model_to_pb_dir()
                self.optimize_w_trt()
            return

        # Otherwise, build the model. It will have 19 layers and process a board of size 13x13.
        inputs = Input(shape=(19, 13, 13))  # Used to be listed as 13 x 13 x 19...

        # Define the input convolutional block that will be used by the network.
        conv_block = Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=1,
                            padding="same",
                            data_format="channels_first",
                            kernel_initializer='random_normal',
                            bias_initializer='random_normal',
                            kernel_regularizer=regularizers.l2(l2_const),
                            bias_regularizer=regularizers.l2(l2_const))(inputs)
        conv_block = BatchNormalization()(conv_block)
        conv_block = Activation("relu")(conv_block)

        # The policy and value blocks will connect to a stack of 11 residual connection blocks.
        res = conv_block
        for i in range(11):
            res = self.create_res_block(res)

        # The policy block will have an output size of 170 (169 board positions, and one option to pass).
        policy = Conv2D(filters=2,
                        kernel_size=(1, 1),
                        strides=1,
                        padding="same",
                        data_format="channels_first",
                        kernel_initializer='random_normal',
                        bias_initializer='random_normal',
                        kernel_regularizer=regularizers.l2(l2_const),
                        bias_regularizer=regularizers.l2(l2_const))(res)
        policy = BatchNormalization()(policy)
        policy = Activation("relu")(policy)
        policy = Reshape((13*13*2,))(policy)
        policy = Dense(170, activation="softmax", name="policy")(policy)

        # The value block will have an output size of 1, indicating the outcome of the game (win or loss).
        value = Conv2D(filters=1,
                       kernel_size=(1, 1),
                       strides=1,
                       padding="same",
                       data_format="channels_first",
                       kernel_initializer='random_normal',
                       bias_initializer='random_normal',
                       kernel_regularizer=regularizers.l2(l2_const),
                       bias_regularizer=regularizers.l2(l2_const))(res)
        value = BatchNormalization()(value)
        value = Activation("relu")(value)
        value = Reshape((13*13,))(value)
        value = Dense(256, activation="relu")(value)
        value = Dense(1, activation="tanh", name="value")(value)

        # Construct the model by combining the inputs and the value / policy outputs.
        self.model = Model(inputs=inputs, outputs=[value, policy])

        # Compile the model. If training with supervised learning, give greater weight to the policy network.
        if train_supervised:
            self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9),
                               loss={"value": tf.keras.losses.mean_squared_error,
                                     "policy": tf.keras.losses.categorical_crossentropy},
                               loss_weights=[0.01, 1.0],
                               metrics=[metrics.mae, metrics.categorical_accuracy])
        else:
            self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9),
                               loss={"value": tf.keras.losses.mean_squared_error,
                                     "policy": tf.keras.losses.categorical_crossentropy},
                               loss_weights=[1.0, 1.0],
                               metrics=[metrics.mse, metrics.categorical_accuracy])
        print(self.model.summary())

    def load_latest_model(self):
        self.model = load_model("young_saigon.h5")

    def load_model_from_file(self, model_file):
        self.model = load_model(model_file)
        if self.trt_mode:
            self.optimize_w_trt()

    def create_res_block(self, input_layer):
        res_conn = input_layer
        res_block = Conv2D(filters=256,
                           kernel_size=(3, 3),
                           strides=1,
                           padding="same",
                           data_format="channels_first",
                           kernel_initializer='random_normal',
                           bias_initializer='random_normal',
                           kernel_regularizer=regularizers.l2(self.l2_const),
                           bias_regularizer=regularizers.l2(self.l2_const))(input_layer)
        res_block = BatchNormalization()(res_block)
        res_block = Activation("relu")(res_block)
        res_block = Conv2D(filters=256,
                           kernel_size=(3, 3),
                           strides=1,
                           padding="same",
                           data_format="channels_first",
                           kernel_initializer='random_normal',
                           bias_initializer='random_normal',
                           kernel_regularizer=regularizers.l2(self.l2_const),
                           bias_regularizer=regularizers.l2(self.l2_const))(res_block)
        res_block = BatchNormalization()(res_block)
        res_block = tf.keras.layers.add([res_conn, res_block])
        res_block = Activation("relu")(res_block)
        return res_block

    def predict_given_state(self, model_inputs, batch_size=1):
        if self.trt_mode:
            return self.predict_w_trt(model_inputs)
        pred_value, prior_probs = self.model.predict(model_inputs, batch_size=batch_size)
        return prior_probs, pred_value

    def save_model_to_pb_dir(self):
        if not self.trt_mode:
            return
        tf.saved_model.save(self.model, "tmp_pb_model_dir")

    def optimize_w_trt(self):
        """
        Optimize the model using TensorRT.
        Load the latest saved network, convert it to an FP16 TensorRT model, and load that.
        """
        if not self.trt_mode:
            return
        converter = trt.TrtGraphConverterV2(input_saved_model_dir="tmp_pb_model_dir",
                                            conversion_params=self.conversion_params)
        converter.convert()
        converter.save("tmp_trt_model_dir")
        self.trt_model = tf.saved_model.load("tmp_trt_model_dir", tags=[tag_constants.SERVING])
        self.trt_inference_func = self.trt_model.signatures["serving_default"]

    def predict_w_trt(self, inputs, batch_size=1):
        """
        Perform inference using the TRT graph.
        """
        if not self.trt_mode:
            return self.predict_given_state(inputs, batch_size=batch_size)
        if not self.trt_model:
            print("The TensorRT model has not been created. Therefore inference cannot be called.")
            raise Exception
        inputs = tf.constant(inputs, dtype=float)
        output = self.trt_inference_func(inputs)
        return output["policy"].numpy(), output["value"].numpy()

    def train(self, training_data_input, training_data_gt_value, training_data_gt_policy, save_file):
        """
        This function creates a copy of the model, trains it, and then replaces the model when training is done.
        :param training_data: a numpy array of
        """
        # Load a copy of the model for training.
        training_model = load_model(save_file)
        history = training_model.fit(x=training_data_input,
                                     y={"value": training_data_gt_value, "policy": training_data_gt_policy},
                                     batch_size=32,
                                     epochs=3,
                                     verbose=2)

        # TODO: Watch out for race conditions!!! Obtain a lock to update self.model.
        # Save the newly trained version of this model to the young_saigon.h5 file.
        training_model.save(save_file)
        print("Done training!")

    def train_supervised_gen(self, dataset, epochs=10000, batch_size=32, callbacks=[]):
        history = self.model.fit(x=dataset,
                                 epochs=epochs, verbose=2,
                                 steps_per_epoch=32,
                                 callbacks=callbacks)


    def train_supervised(self, training_data_input, training_data_gt_value, training_data_gt_policy,
                         batch_size=32, callbacks=[]):
        """
        This function creates a copy of the model, trains it, and then replaces the model when training is done.
        :param training_data_input:
        :param training_data_gt_value:
        :param training_data_gt_policy:
        :param batch_size:
        :return: the training history object.
        """
        # Load a copy of the model for training.
        history = self.model.fit(x=training_data_input,
                                 y={"value": training_data_gt_value, "policy": training_data_gt_policy},
                                 epochs=1, verbose=2,
                                 batch_size=batch_size,
                                 callbacks=callbacks)
        return history

    def train_on_self_play_data(self, h5_files, batch_size, num_batches, weights_outfile=None):
        """
        Given a list of h5 files, train on a set of randomly sampled game positions.
        Use the Goban module to apply flips and rotations to increase variation.
        :param h5_files: a list of paths to the h5_files that contain self play data
        :param batch_size: how many moves to train on for a single batch (e.g. 32)
        :param num_batches: how many batches to train on in total (e.g. 64)
        :param weights_outfile: if provided, save the new weights to this file.
        """
        print("TODO")

    def save_checkpoint(self, ckpt_file):
        self.model.save(ckpt_file)
        self.save_model_to_pb_dir()


