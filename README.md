# upo_fcn_learning
ROS Package that contains the scripts needed to perform training of a Fully Convolutional Network (FCN) for the task of robot path planning.

Training data is provided in the folder *data*:

	- **RRT_dataset**. 10500 pairs of images with the nework inputs and respective labels with robot trajectories recorded using a RRT* planner with a set of pre-defined set of features specially designed for robot navigation. 
	- **Real_traj_dataset**. 300 pairs of images (network inputs and labels) sorted in 3 sets with trajectories recorded by a human expert controlling the robot remotely.


Scripts and ROS nodes:

	- **fcn_path_planning.py**. ROS node written in Python for network training. See the launch file "/launch/fcn_training.launch".
	- **prediction_test.py**. Python script for testing a trained network and store the output images.


Other tools:

	- **record_bag_trajectories**. ROS node for recording robot trajectories in rosbag files. This node uses other nodes includes in the metapackage **upo_robot_navigation**. See the launch files in "/launch/record_trajs/" and read the INFO.txt file included.
	- **capture_node**. ROS node that transforms the sensors data and people detection information stored in rosbag files into inputs images and labels images required for training the FCN proposed. The rosbag files are not provide. Please, ask the authors if your are interested in this data. See the launch file "/launch/capture_nav_imgs.launch".


## Dependences

* The instalation of **Keras** (and **Theano**) is required.

The package is a **work in progress** used in research prototyping. Pull requests and/or issues are highly encouraged.


