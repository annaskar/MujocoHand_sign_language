# MujocoHand_sign_language

This project focuses on sign language for the numbers one through nine, using MuJoCo.

![image](https://github.com/user-attachments/assets/d7c96f39-a380-4b8d-aded-b40506ca8395)




# Requirements
* **python==3.10 || 3.12** https://www.python.org/downloads/release/python-390/
* **mujoco=2.3.7** https://github.com/deepmind/mujoco
* **tensorflow==2.9.1** https://www.tensorflow.org/install
* **ray[rllib]==2.3.1** https://docs.ray.io/en/latest/rllib/index.html (to install this easy to your editor first get python 3.8)
* **gymnasium==0.26.1** https://gymnasium.farama.org/
* **matplotlib==3.7.2** https://matplotlib.org/

# ΑΡΧΕΙΑ 
Data &#8594; ctr_limits.csv ## limits of simulator <br>
Data &#8594; expert_dataset.csv ## the dataset       <br>




# Diagram for Dnn
The application of a Deep Neural Network (DNN) in the Shadow Hand system aims to control the extremities of a robotic arm to perform sign language gestures. This DNN is trained and applied to generate predictions for the robotic hand’s actuators, based on input data such as gestures and command sequences. 
![image](https://github.com/user-attachments/assets/7e6012cc-b020-48d7-994f-c48b66d9e3fe)


