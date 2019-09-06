import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Reshape
from keras import regularizers
from keras.models import Model


class PolicyValueNetwork:

    def __init__(self, l2_const, starting_network_file=None):

        self.l2_const =l2_const

        # If a model is already provided, load it from the file.
        if starting_network_file:
            self.model = keras.models.load_model(starting_network_file)
            return

        inputs = Input(shape=(13, 13, 17))

        conv_block = Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=1,
                            padding="same",
                            kernel_initializer='random_uniform',
                            bias_initializer='random_uniform',
                            kernel_regularizer=regularizers.l2(l2_const),
                            bias_regularizer=regularizers.l2(l2_const))(inputs)
        conv_block = BatchNormalization()(conv_block)
        conv_block = Activation("relu")(conv_block)

        res = conv_block
        for i in range(19):
            res = self.create_res_block(res)

        policy = Conv2D(filters=2,
                        kernel_size=(1, 1),
                        strides=1,
                        padding="same",
                        kernel_initializer='random_uniform',
                        bias_initializer='random_uniform',
                        kernel_regularizer=regularizers.l2(l2_const),
                        bias_regularizer=regularizers.l2(l2_const))(res)
        policy = BatchNormalization()(policy)
        policy = Activation("relu")(policy)
        policy = Reshape((13*13*2,))(policy)
        policy = Dense(170, activation='softmax', name="policy")(policy)

        value = Conv2D(filters=1,
                       kernel_size=(1, 1),
                       strides=1,
                       padding="same",
                       kernel_initializer='random_uniform',
                       bias_initializer='random_uniform',
                       kernel_regularizer=regularizers.l2(l2_const),
                       bias_regularizer=regularizers.l2(l2_const))(res)
        value = BatchNormalization()(value)
        value = Activation("relu")(value)
        value = Reshape((13*13,))(value)
        value = Dense(256, activation="relu")(value)
        value = Dense(1, activation="tanh", name="value")(value)

        self.model = Model(inputs=inputs, outputs=[value, policy])
        self.model.compile(optimizer="adam",
                           loss={"value": keras.losses.mean_squared_error, "policy": keras.losses.binary_crossentropy},
                           loss_weights=[1.0, 1.0])

        self.model._make_predict_function()

        # Save the model so we can load it in a different thread!!
        self.save_model_to_file("young_saigon.h5")

    def load_latest_model():
        self.model = keras.models.load_model("young_saigon.h5")

    def create_res_block(self, input_layer):
        res_conn = input_layer
        res_block = Conv2D(filters=256,
                           kernel_size=(3, 3),
                           strides=1,
                           padding="same",
                           kernel_initializer='random_uniform',
                           bias_initializer='random_uniform',
                           kernel_regularizer=regularizers.l2(self.l2_const),
                           bias_regularizer=regularizers.l2(self.l2_const))(input_layer)
        res_block = BatchNormalization()(res_block)
        res_block = Activation("relu")(res_block)
        res_block = Conv2D(filters=256,
                           kernel_size=(3, 3),
                           strides=1,
                           padding="same",
                           kernel_initializer='random_uniform',
                           bias_initializer='random_uniform',
                           kernel_regularizer=regularizers.l2(self.l2_const),
                           bias_regularizer=regularizers.l2(self.l2_const))(res_block)
        res_block = BatchNormalization()(res_block)
        res_block = keras.layers.add([res_conn, res_block])
        res_block = Activation("relu")(res_block)
        return res_block

    def predict_given_state(self, model_inputs, batch_size=1):
        pred_value, prior_probs = self.model.predict(model_inputs, batch_size=batch_size)
        return prior_probs, pred_value

    def print_model(self):
        from keras.utils import plot_model
        plot_model(self.model, to_file='model.png')
        keras.utils.print_summary(self.model)

    def save_model_to_file(self, model_filename):
        self.model.save(model_filename)

    def train(self, training_data_input, training_data_gt_value, training_data_gt_policy):
        """
        This function creates a copy of the model, trains it, and then replaces the model when training is done.
        :param training_data: a numpy array of
        """
        # Load a copy of the model for training.
        training_model = keras.models.load_model("young_saigon.h5")
        #training_model = self.model
        training_model.fit(x=training_data_input,
                          y={"value": training_data_gt_value, "policy": training_data_gt_policy},
                          batch_size=32)

        # TODO: Watch out for race conditions!!! Obtain a lock to update self.model.
        # Save the newly trained version of this model to the young_saigon.h5 file.
        training_model.save("young_saigon.h5")
        print("Done training!")



