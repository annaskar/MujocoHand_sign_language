import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from models.tf.nn import NeuralNetwork
from utils import dataset
from controllers.model import ModelController
from models.tf.nn import NeuralNetwork
from pyopengl import GLFWSimulator
from utils import control
from utils import dataset


dataset_filepath = 'data/expert_dataset.csv'
checkpoint_directory = 'checkpoints/nn'
one_hot_signs = True
num_outputs = dataset.NUM_ACTUATORS
learning_rate = 0.001
loss_fn = 'MAE'
epochs = 100
batch_size = 10
seed = 20

model_checkpoint_directory = 'checkpoints/nn'
shadow_hand_xml_filepath = 'objects/shadow_hand/scene_left.xml'
ctrl_limits_filepath = 'data/ctrl_limits.csv'
trajectory_steps = 10
cam_verbose = False
sim_verbose = True



summary = False
print_test_predictions = False
plot_performance = True


def main():
    print("this is num of out")
    print(num_outputs)
    tf.random.set_seed(seed=seed)
    np.random.seed(seed=seed)
    random.seed(seed)

    x, y = dataset.read_dataset(dataset_filepath=dataset_filepath, one_hot=one_hot_signs)

    print('Train Dataset:', y.shape, x['sign'].shape, x['order'].shape)



    model = NeuralNetwork(
        input_shapes={name: input_data.shape[1:] for name, input_data in x.items()},
        num_outputs=num_outputs,
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        epochs=epochs,
        batch_size=batch_size
    )
    model.build(summary=summary)
    loss = model.train(x=x, y=y)
    model.save(checkpoint_directory=checkpoint_directory+"saved_model.keras")

    if print_test_predictions:
        y_pred = model.predict_next_control(x['sign'], x['order'])

        for i in range(y.shape[0]):
            print(f'i: {i + 1}, y_pred: {y_pred[i]}\tactual: {y[i]}')

    if plot_performance:
        plt.plot(loss, label='Loss')
        plt.title('Neural Network Performance')
        plt.xlabel('Epochs')
        plt.ylabel(loss_fn)
        plt.legend()
        plt.show()

    model = NeuralNetwork(input_shapes={'sign': (), 'order': ()}, num_outputs=-1)
    model.load(checkpoint_directory=model_checkpoint_directory + "saved_model.keras")

    ctrl_limits = control.read_ctrl_limits(csv_filepath=ctrl_limits_filepath)
    hand_controller = ModelController(
            model=model,
            ctrl_limits=ctrl_limits,
            num_actuators=dataset.NUM_ACTUATORS,
            one_hot_signs=dataset.ONE_HOT_SIGNS,
            one_hot_orders=dataset.ONE_HOT_ORDERS
        )
    simulator = GLFWSimulator(
            shadow_hand_xml_filepath=shadow_hand_xml_filepath,
            hand_controller=hand_controller,
            trajectory_steps=trajectory_steps,
            cam_verbose=cam_verbose,
            sim_verbose=sim_verbose
        )
    simulator.run()


if __name__ == '__main__':
    main()
