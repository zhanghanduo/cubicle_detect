.. _chapter-userguide:

USER GUIDE
==========

.. contents:: Contents
   :local:
   :backlinks: none


0. Source Code Structure
------------------------
::

    sslam/
            ├── README.md
            ├── package.xml
            ├── CMakeLists.txt
            ├── cmake
            │   ├── FindEigen.cmake
            │   └── FindSphinx.cmake
            ├── Config                  # (1) Configuration files including parameter set
            │   ├── bus                       and calibration.
            │   ├── bus2
            │   ├── euroc
            │   ├── honda
            │   ├── kitti_odom
            │   ├── kitti_raw
            │   └── zed
            ├── launch                  # (2) ROS launch file, the entry point of the code.
            │   ├── bus_core.launch
            │   ├── bus_global_core.launch
            │   ├── bus_global.launch
            │   └── bus.launch
            ├── docs
            │   ├── html
            │   ├── latex
            │   └── xml
            ├── camera_models           # (3) Camera model module, with pinhole and
            │   ├── include                   other camera definition and modeling.
            │   ├── readme.md
            │   └── src
            ├── slam_estimator          # (4) SLAM estimator module, the core library
            │   ├── cmake-build-debug         for real-time visual (inertial) odometry.
            │   ├── CMakeLists.txt
            │   ├── package.xml
            │   └── src
            ├── pose_graph              # (5) Pose graph module, the loop closure and
            │   ├── CMakeLists.txt            map management library.
            │   ├── package.xml
            │   └── src
            ├── global_fusion           # (6) Global fusion module, the library that
            │   ├── CMakeLists.txt            integrates slam_estimator with GPS message.
            │   ├── models
            │   ├── package.xml
            │   ├── src
            │   └── ThirdParty
            └── support_files           # (7) Feature descriptor training files and SSLAM
                ├── brief_k10L6.bin           related documentation module.
                ├── brief_pattern.yml
                ├── doc
                └── image

1. Configuration
----------------

Configuration file lies in ``config/`` directory, stored in ``yaml`` format. Before you run the code, make sure you have correctly set the parameters.

#. Sensor

    - ``imu``: set to ``0`` means ignores **IMU input**, ``1`` means uses **IMU input**.

    - ``ins``: set to ``0`` means ignores **INSPVA input**, ``1`` means uses **INSPVA input**.

    - ``num_of_cam``: set to ``1`` means uses **monocular camera**, ``2`` means uses **stereo camera**.

    - ``cubicle``: set to ``0`` means ignores **object detection** input, ``1`` means uses **object detection** input.

        .. NOTE ::

            ``cubicle`` requires both `obstacle msgs <https://gitlab.com/ugv_stereo/obstacle_msgs.git>`_ package and `cubicle detect <https://gitlab.com/ugv_stereo/cubicle_detect.git>`_ package.


    - ``gps_initial``: set to 1, then the initial robot body pose is aligned with **GPS pose** at the initial moment, for visualization and evaluation; otherwise SLAM starts at local frame.

    - ``imu_topic``: set IMU topic name if use IMU data. IMU message is: ``sensor_msgs/Imu``.

    - ``ins_topic``: set INS topic name if use INS data. INSPVA message is: ``msg_novatel_inspva``.

    - ``image0_topic``: set left image topic name.

    - ``image1_topic``: set right image topic name, if use stereo camera.

    - ``cubicle_topic``: set object detection topic name, if enable  ``cubicle`` flag. Cubicle message is: ``obstacle_msgs/MapInfo``.

    - ``gps_topic``: set GPS pose topic name, if enable ``gps_initial`` flag. GPS message has format ``geometry_msgs/PoseWithCovarianceStamped``.

#. Camera Calibration

    - ``estimate_extrinsic``: set extrinsic parameter between IMU/INS and camera.

    - ``body_T_cam0``: set extrinsic matrix from **left camera** to IMU/INS.

    - ``body_T_cam1``: set extrinsic matrix from **right camera** to IMU/INS.


#.  Feature Extraction

        - ``use_gpu``: set to ``0`` means use CPU to extract features; set to ``1`` means use GPU to accelerate.

        - ``use_gpu_acc_flow``: set to ``0`` means use CPU to calculate optical flow; set to ``1`` means use GPU to accelerate.

#. SLAM System Setting

    - ``multiple_thread``: set to ``1`` means use multiple threading. ``0`` mode is useful for debugging.

    - ``show_track``: set to ``1`` means publish SLAM feature tracking image as a ROS image topic.

#. SLAM Algorithm Setting

    - ``flow_back``: set to ``1`` to perform forward and backward optical flow to improve feature tracking accuracy; set to ``0`` to save more time during tracking.


2. Launch File
--------------

ROS Launch file lies in ``launch/`` directory.


3. Demo on bus
--------------

.. code-block:: bash

    roslaunch sslam_estimator bus.launch


3. EuRoC Example
----------------

.. _section-euroc:

Download `EuRoC MAV Dataset <http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets>`_ to ``YOUR_DATASET_FOLDER``. Take ``MH_01`` for example, you can run SSLAM with three sensor types (monocular camera + IMU, stereo cameras + IMU and stereo cameras).
Open 3 terminals, run sslam_estimator, rviz and play the bag file respectively.
Green path is VIO odometry; yellow path is odometry under visual loop closure.


3.1 Monocualr camera + IMU

.. code-block:: bash

    roslaunch sslam_estimator bus_imu.launch
    rosbag play YOUR_DATASET_FOLDER/bus_03_with_imu.bag


3.2 Stereo cameras + IMU

.. code-block:: bash

    roslaunch sslam_estimator bus_imu.launch
    rosbag play YOUR_DATASET_FOLDER/bus_03_with_imu.bag

3.3 Stereo cameras

.. code-block:: bash

    roslaunch sslam_estimator bus.launch
    rosbag play YOUR_DATASET_FOLDER/bus_01.bag


4. KITTI Example
----------------

.. _section-kitti:

4.1 KITTI Odometry (Stereo)
Download `KITTI Odometry dataset <http://www.cvlibs.net/datasets/kitti/eval_odometry.php>`_ to YOUR_DATASET_FOLDER. Take sequences 00 for example,
Open two terminals, run sslam_estimator and rviz respectively.

.. code-block:: bash

    rosrun sslam_estimator kitti_odom_test ~/catkin_ws/src/sslam_resuse/slam_estimator/config/kitti_odom/kitti_config00-02.yaml YOUR_DATASET_FOLDER/sequences/00/

4.2 KITTI GPS Fusion (Stereo + GPS)
Download `KITTI raw dataset <http://www.cvlibs.net/datasets/kitti/raw_data.php>`_ to ``YOUR_DATASET_FOLDER``.
Open three terminals, run sslam_estimator, global fusion and rviz respectively.
Green path is VIO odometry; blue path is odometry under GPS global fusion.

.. code-block:: bash

    roslaunch sslam_estimator bus_global.launch


5. SSLAM on car demonstration with GPS
--------------------------------------
.. _section-sslam:

Run SSLAM odometry, rviz and play the bag file respectively.
Green path is VIO odometry; red path is odometry under visual loop closure.

.. code-block:: bash

    roslaunch sslam_estimator car.launch
    rosbag play YOUR_DATASET_FOLDER/car.bag


