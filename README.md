# MujocoHand_sign_language
  ![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/b1ac5f53-3876-4027-91db-dde79d5997ab)
 ![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/10eb5d7a-0bd1-46ca-9c66-1e689a2f0a91)


The MujocoHand Sign Language project aims to develop a framework for predicting movements in sign language. <br>
This document describes the software requirements, file structure, data specifications, and the neural network architecture used in the project.

# Requirements
The project utilizes several libraries and tools to facilitate its implementation:
* **python==3.10 || 3.12** https://www.python.org/downloads/release/python-390/
* **mujoco=2.3.7** https://github.com/deepmind/mujoco and https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand 
* **tensorflow==2.9.1** https://www.tensorflow.org/install
* **ray[rllib]==2.3.1** https://docs.ray.io/en/latest/rllib/index.html (to install this easy to your editor first get python 3.8)
* **gymnasium==0.26.1** https://gymnasium.farama.org/
* **matplotlib==3.7.2** https://matplotlib.org/

# Files 
The project is organized into several directories, each serving a specific purpose:
Data &#8594; ctr_limits.csv [ limits of simulator] <br>
Data &#8594; expert_dataset.csv [ the dataset ]      <br>
environments &#8594; shadowhand.py [ control the simulator ]. <br>
models &#8594; tf  &#8594; nn.py [ the neural network structure]  <br>
objects  &#8594; shadow_hand [ there are all the scenes of mujoco.] <br>
utils  &#8594; dataset.py [ the dataset in with hot encoding . ]<br>
train_nn.py [ run neural network and after run the simulator .] <br>
pyopengl.py [ controlling  between simulator and the user .] <br>

# Data
Inputs are : <br>
* Dataset random movements .The final state of each movement which is represent at the table below <br>
* key from 1 to 9 which is represent a movement
Both of them are encoding with one hot encoding .<br>
<br>

| Movement | Sign Set                                |Image                    |
|----------|--------------------------------------------------------|--------------------------|
| rest     | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] | ![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/9d4a957d-2f21-42c5-96d1-25e070966e71)|
| one      | [0, 0, 0, 1.22, 0.2, 0.69, 1.57, 0, 0, 0, 0, 1.57, 3.14, 0, 1.57, 3.14, 0, 0, 1.57, 3.14] | ![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/2f175643-1e86-4922-96fe-e6c80b89eb13)|
| two      | [0, 0, 0, 0, 0.2, 0.69, 1.57, 0, 0, 0, 0, 0, 0, 0, 1.57, 3.14, 0, 0, 1.57, 3.14] |![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/ca708aba-45f3-4e81-bef6-cc2bc773b4cb)|
| three    | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.57, 3.14, 0, 0, 1.57, 3.14] |  ![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/4308b40d-ee43-4d88-9bac-805794883523) |
| four     | [0, 0, 0, 0, 0.2, 0.69, 1.57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] | ![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/e0d8be8e-5606-44eb-9af6-87c461c8380e)|
| five     | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | ![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/444a13f7-6fc4-47ef-a4c7-1d6c04ef84ad) |
| six      | [0, 0, 0, 0.8, 0.2, 0.69, 1.57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7, 0, 1.57, 1.29] |![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/4765c3a9-2e97-4fdb-a611-1c897b215983)|
| seven    | [0, 0, 0, 1.22, 0.2, 0.69, 0.84, 0, 0, 0, 0, 0, 0, 0, 1.57, 1.6, 0, 0, 0, 0] | ![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/3062fd97-8835-469f-b875-1e2e20770951)|
| eight    | [0, 0, 0, 0.7, 0.2, 0.69, 1.29, 0, 0, 0, 0.349066, 1.25, 2, 0, 0, 0, 0, 0, 0, 0] | ![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/47a360db-5f35-4024-8d02-9e7398535f23)|
| nine     | [0, 0, 0, 1.22, 0.2, 0.69, 1.57, 0, 1.57, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/ce045f04-33ed-44cc-8f30-df13efba333d)|


</br>

| Order | One-Hot Encoding                      |
|-------|---------------------------------------|
| 0     | [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]        |
| 1     | [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]        |
| 2     | [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]        |
| 3     | [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]        |
| 4     | [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]        |
| 5     | [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]        |
| 6     | [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]        |
| 7     | [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]        |
| 8     | [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]        |
| 9     | [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]        |





 <br> </br>
# Neural network
The aim of this Neural Network is to predict the 20 actuators by using the sign(final state of sign language) and the order(keys from keyboard)
This neural network utilizes 2 hidden layers and concatenates .  Each prediction corresponds to a 20-value vector representing the edge of the actuators.
 <br> </br>
![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/2febac40-23a9-4c0a-babd-33b54b16e587)<br>
For that the neural network uses Adam optimizer - which is a improvemnt of Gradient Descent algorithm -, and Mean Absolute Error function in order to evaluate prediction error.
 <br> </br>

![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/50133716-fbc8-44a2-8e74-c2178439e193) <br>

After experimentation, the ideal values determined for training over 1000 epochs are as follows:

 * Batch size: 10
* Learning rate: 0.001 

These values were found to optimize the convergence and performance of the neural network model in the current application which is  under consideration.

 <br> </br>

!![image](https://github.com/annaskar/MujocoHand_sign_language/assets/69804667/6ae5fc82-6f39-4d7b-8ff6-c95e1834c891)



