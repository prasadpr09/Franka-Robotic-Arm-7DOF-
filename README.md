# Programmed the Franka Emika Arm (part of coursework MEAM520: Introduction to Robotics)
### Date Modified: 08/29/2024

This repo contains code for Forward and Inverse Kinematics and Path planning algorithms for the Franka Emika Robotic Arm.


# Subdirectory Structure:
- `core`: contains helper functions and interfaces for interacting with the robot (either simulated or real!)
- `lib`:  contains functions implementing algorithms to solve computational problems, such as forward and inverse kinematics and trajectory planning
- `labs`: contains test scripts that use the algorithms I implement in `lib` and the tools in `core` to control the robot and achieve tasks
- `ros`: contains code necessary to launch the simulator. You won't need to work in this directory.


# Native Install Instructions (NOT REQUIRED FOR VIRTUAL MACHINE)
---
**NOTE**

These installation instructions are for the TAs when setting up the Virtual Machine, and can also be followed by experienced users to set up the lab environment on a native Ubuntu Linux installation. These steps do not need to be followed by students using the Virtual Machine. For all other users, skip to the section on forking this repo.

---

## Operating System

The simulator must be run on Ubuntu 20.04 with ROS noetic installed. You can follow the standard installation instructions for [Ubuntu 20.04](https://phoenixnap.com/kb/install-ubuntu-20-04) and [ROS noetic](http://wiki.ros.org/noetic/Installation).

## panda_simulator installation

To get started using this development environment, you must first follow the instructions to install [panda_simulator](https://github.com/justagist/panda_simulator/tree/noetic-devel), a Gazebo-based simulator for the Franka Emika Panda robot. The only difference is that you will name the catkin workspace `meam520_ws` to avoid conflicts with other projects.

The instructions specify to use `catkin build`, but we recommend building with `catkin_make_isolated` instead.

Once you've completed that installation, add

```
source ~/meam520_ws/devel_isolated/setup.bash
```

to your `~/.bashrc` to source the workspace. If all has been done correctly, you should be able to run

```
roslaunch panda_gazebo panda_world.launch
```

to launch the Gazebo simulation.

Note known dependencies that are not installed from the above instructions are:

```
pip3 install numba scipy future
```

### Speed up Gazebo shutdown

Since you may be launching and killing Gazebo many times this semester, we recommend reducing the clean shutdown wait time for a better experience. Edit the file:
```
cd /opt/ros/noetic/lib/python3/dist-packages/roslaunch
sudo vim nodeprocess.py
```
and change the lines that say
```
...
_TIMEOUT_SIGINT = 15.0 #seconds

_TIMEOUT_SIGTERM = 2.0 #seconds
...
```
to `2.0` and `1.0` respectively.

## Update Python path to find core module

Add the following line to your ~/.bashrc to allow your python scripts to find the meam520 core modules.

```
export PYTHONPATH="${PYTHONPATH}:/home/${USER}/meam520_ws/src/meam520_labs"
```


To check that the installation is successful, run a demo script:

```
cd ~/meam520_ws/src/meam520_labs/labs/lab0
python demo.py
```

You should see the robot in Gazebo moving.


