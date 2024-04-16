# MujocoHand_sign_language


# Requirements
* **python==3.10 || 3.12** https://www.python.org/downloads/release/python-390/
* **mujoco=2.3.7** https://github.com/deepmind/mujoco
* **tensorflow==2.9.1** https://www.tensorflow.org/install
* **ray[rllib]==2.3.1** https://docs.ray.io/en/latest/rllib/index.html (to install this easy to your editor first get python 3.8)
* **gymnasium==0.26.1** https://gymnasium.farama.org/
* **matplotlib==3.7.2** https://matplotlib.org/

# File 
Data &#8594; ctr_limits.csv ## limits of simulator <br>
Data &#8594; expert_dataset.csv ## the dataset       <br>




# Neural network
The proposal of this Neural Network is to predict the 20 actuators by using the sign(final state of sign language) and the order(keys from keyboard)
![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/2febac40-23a9-4c0a-babd-33b54b16e587)
For that the neural network uses Adam optimizer - which is a improvemnt of Gradient Descent algorithm -, and Mean Absolute Error function in order to evaluate prediction error.
 ![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/727d1a58-630e-4afe-9d5e-f7e7deece557)
After some experimentation the ideal values ​​for batch size is 10 and the seed for this network .
![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/1394591e-7d5a-4841-98d8-7d5e6039f8a2)


