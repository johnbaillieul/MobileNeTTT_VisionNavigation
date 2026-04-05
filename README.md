# MobileNeTTT: DNN based Time-To-Transit Estimator for Visual Navigation

[![ROS 2 Jazzy](https://img.shields.io/badge/ROS2-Jazzy-blueviolet)](https://docs.ros.org/en/jazzy/)
[![Ubuntu 24.04](https://img.shields.io/badge/Ubuntu-24.04-E95420?logo=ubuntu)](https://releases.ubuntu.com/24.04/)
[![License: BSD-2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-blue.svg)](./LICENSE)


This repository contains code that allows navigation in unknown environments using only a monocular camera. The  work  takes  inspiration  from a  classic  paper  by  Lee  and  Reddish  (Nature,  1981, https://doi.org/10.1038/293293a0) in which they outline a behavioral strategy pursued by diving sea birds based on a visual cue called time-to-contact. The algorithm we introduce here is based on the estimation of a closely related quantity called time-to-transit (TTT).

The main code is represented by three nodes: the `/OpticalFlow node` is  responsible  for  the  Optical Flow (OF) estimation.  It  acquires  a  sequence  of  images  from the  camera  mounted  on  the  robot  and  it  extracts the  relevant  features  to  finally  compute  the  OF vectors. The `/TauComputation node` analyzes the array of keypoints with their velocities packed in the OF message, it computes tau values and it creates input signals for the controller. The `/Controller` selects the right control action to be used depending on the distribution of TTT values and it implements the Sense-Act cycle. For full details see our ICRA2022 paper, available on [arXiv](https://arxiv.org/abs/2111.09669).

<div align="center">
    <h4>Past and Current Architecture</h4>
    <img src="assets/architecture.gif" width = 700 alt="BU Robotics Lab U Shaped Corridor"/>

</div>

<div align="center">
    <h4>U Shaped Corridor at BU Robotics Lab</h4>
    <img src="assets/u_shaped_readme.gif" alt="BU Robotics Lab U Shaped Corridor"/>

</div>
<!--Use 480fps find a way such that we can make gifs not so heavy and easy diagrams, currently 480p cropped out gif-->

##  License and attribution

If you use the code in this repository, please cite [our paper](https://arxiv.org/abs/2111.09669). The code is available under the [BSD-2-Clause License](./LICENSE).

```bibtex
@inproceedings{bbzbaillieul2021,
      title={Visual Navigation Using Sparse Optical Flow and Time-to-Transit},
      author={Boretti, Chiara and Bich, Philippe and Zhang, Yanyu and Baillieul, John},
      booktitle={IEEE International Conference on Robotics and Automation},
      year={2022}
}
```
## Prerequisites
- Python 3.12
- PyTorch with CUDA
- torchvision
- Custom `cbam.py` module (Provided in repo)
- Ubuntu 24.04.
- ROS2 Jazzy (EOL 2029) and Gazebo Harmonic
- ROS-2 Control `sudo apt install ros-jazzy-ros2-control ros-jazzy-ros2-controllers`

**If not using local jackal urdf provided**
- Clearpath package to simulate Jackal UGV, use the following instruction: `sudo apt-get install ros-<distro>-jackal-simulator ros-<distro>-jackal-desktop ros-<distro>-jackal-navigation`. 

> [!Important]
> In simulation, it is important to include as many distinct features for optimal estimation performance of the trained model

## Setup and use

Our navigation strategy can be used with any mobile robot equipped with a monocular camera.

First of all, run `nodes/compute_ttt.py` to compute Time-To-Transit values from a live camera stream. The node subscribes to a camera image topic and processes consecutive frame pairs - cropping each frame to the region of interest (rows 70-340), normalizing and concatenating them into a 6-channel input tensor. This tensor is passed through a MobileNetV3-Small backbone, optinoally enahcne with CBAM attention module (channel+spatial attention), to produce a 2-channel output: predicted TTT values and a confidence map. Cells with confidence below 0.5 are masked out, and the resulting TTT grid is published as a TTTValues.msg for downstream navigation nodes. An optional grid overlay can be displayed in real time for debugging. 

Run the node with: `ros2 run mobilenettt_visionnavigation ttt_values`
<div align="center">
  <img src="assets/estimator.png" width="810" alt="Dynamic_Env_w_moving_obstacles"/>
</div>


Dynamic Environment with moving agents.

Next step is run nodes/controller.py to compute and publish velocity commands based on the TTT values received from the previous node. The node subscribes to the TTTValues.msg topic and reshapes the incoming data into a 10×20 grid, computing column-wise mean TTT values while ignoring masked-out (zero) cells. The column means are then analyzed to identify the peak TTT region (closest obstacle) and its left/right boundaries using a neighbor-search algorithm with a configurable threshold (x_percent). Based on the detected obstacle layout, the controller applies a proportional control law to generate an angular velocity command — steering the robot away from the nearest obstacle while maintaining forward motion. Three control cases are handled: obstacles on both sides, obstacle only on the left, and obstacle only on the right. The resulting TwistStamped message is published to the /cmd_vel topic. An optional histogram overlay can be enabled for debugging by setting plot_histogram = True. Run the node with:

<div align="center">
  <img src="assets/sl_corridor.gif" width="410" alt="Dynamic_Env_w_moving_obstacles"/>
  <img src="assets/l_corridor.gif" width="410" alt="Dynamic_Env_w_moving_obstacles"/>
</div>


Essential parameters for these three nodes are shown below. Other parameters exist and their default values are good for a large amount of environments, but better performances can be achieved by fine-tuning them to adapt the algorithm to the particular environment in which the robot has to move.

### Parameters for `optical_flow.py`

| Parameter            | Description                                                                                                   |    Example Value  |
| -------------------- | ------------------------------------------------------------------------------------------------------------- | :---------------: |
| ~image_sub_name      | name of the Image topic to subscribe.                                                                         | "front/image_raw" |
| ~num_ext_features    | max number of features to be detected in the two outer parts of the image.                                    |        250        |
| ~num_cen_features    | max number of features to be detected in the central part of the image.                                       |        150        |
| ~min_feat_threshold  | minimum % of tracked features that must still be in the image to avoid the reusage of the detector. Parameter must stay in range (0.0-1.0].      |        0.7        |

### Parameters for `tau_computation.py`

| Parameter            | Description                                                                                                   |        Example Value      |
| -------------------- | ------------------------------------------------------------------------------------------------------------- | :-----------------------: |
| ~image_sub_name      | name of Image topic to subscribe to allow visual representation of results of the node. The same used in `optical_flow.py` | "front/image_raw": |
| ~min_TTT_number      | minimum number of features needed to compute the average TTT for each ROI                                     |             10            |
| ~x_init_*ROI*        | x coordinate of the top left corner of ROI                                                                    |              0            |
| ~y_init_*ROI*        | y coordinate of the top left corner of ROI                                                                    |              0            |
| ~x_end_*ROI*         | x coordinate of the bottom right corner of ROI                                                                | int(3\*img_width/12)      |
| ~y_end_*ROI*         | y coordinate of the bottom right corner of ROI                                                                | int(7.5\*img_width/12)    |

where *ROI*={*el*: far left ROI, *er*: far right ROI, *l*: left ROI, *r*: right ROI, *c*: central ROI}

### Parameters for `controller.py`

| Parameter            | Description                                                                                                   |        Example Value      |
| -------------------- | ------------------------------------------------------------------------------------------------------------- | :-----------------------: |
| ~robot_publisher     | name of topic on which the control action generated by the node has to be published.                          | "jackal_velocity_controller/cmd_vel" 
| ~percentage          | percentage of discarded time-to-transit values for each ROI                                                   |             0.25           
| ~max_u               | saturation value for the control input. It depends on the specifics of the selected robot.                    |              1            |

Note that it is also possible to tune the duration of the *Sense* and *Act* cycles.

## Virtual environments
To simulate the behavior of the algorithm in artificial and realistic environments, many scenarios are created in Gazebo. In this repository you will find the code to recreate them in the **GazeboWorlds** folder.  

<p align="center">
  <img src="assets/virtual_l_shaped.png" width="200" alt="Dynamic_Env_w_moving_obstacles"/>
  &nbsp;&nbsp;&nbsp;
  <img src="assets/virtual_zig_zag.png" width="200" alt="Zig Zag Environment"/>
  &nbsp;&nbsp;&nbsp;
  <img src="assets/aerial_view.png" width="250" alt="Dynamic_Env_w_moving_obstacles"/>
</p>



We also developed patterns to be put on the walls with fixed feature density (Bernoulli distributions of features). The pattern can be created using the `Bernoulli_Textures.py` python scripts in which the density of the features can be set manually. Then the new pattern must be added to the Gazebo materials, the *bernoullixM.png* (where x is the percentage indicating the feature density chosen) generated by the python script has to be saved in `/usr/share/gazebo<y>/media/materials/textures` (where y is the Gazebo version). To use the new material in Gazebo, a *bernoullixM.material* file must be created and saved in `/usr/share/gazebo<y>/media/materials/scripts`. An example of *bernoullixM.material* is:
```
material BernoulliMix/<X>
 {
    technique
    {
       pass
        {
          texture_unit
          {
            texture bernoullixM.png
          }
       }  
    }
 }
 ```
Dynamic Environment with moving agents.
<div align="center">
  <h4>Dynamic Environment with moving agents</h4>
  <img src="assets/dynamic_env_readme.gif" alt="Dynamic_Env_w_moving_obstacles"/>
</div>



<p align="right">
    <img src="assets/torino_logo.png" alt="Features density" width="150"/>
  &nbsp;&nbsp;&nbsp;
  <img src="assets/bu_logo.svg" alt="Features density" width="150"/>
</p>
