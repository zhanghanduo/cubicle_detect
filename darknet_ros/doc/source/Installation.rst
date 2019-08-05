.. _chapter-installation:

============
Installation
============

Getting the source code
=======================
.. _section-source:

If you want the latest version, you can clone the git repository

.. code-block:: bash

       git clone https://gitlab.com/ugv_stereo/sslam_resuse.git -b devel

.. _section-dependencies:

Dependencies
============

  .. NOTE ::

    SSLAM package requires ROS environment and a **fully C++11-compliant**
    compiler. Our code has been fully tested under Ubuntu 16.04, ROS Kinetic.

1. Ceres Solver
---------------

Follow `Ceres Installation
<http://ceres-solver.org/installation.html>`_.

2. OpenCV 3.4
---------

Follow `Opencv Installation
<https://zhanghanduo.github.io/post/new_system>`_.

3. Cereal Serialization
-----------------------

It is used to serialize / deserialize map points swiftly and lightly.
Download it from `github repository <https://github.com/USCiLab/cereal.git>`_.

  .. NOTE ::
    Under Ubuntu 18.04, you have to edit CMakeLists.txt and change line 4 from ``OFF`` to ``ON`` to avoid compiling error.

.. code-block:: bash

        mkdir build && cd build
        cmake ..
        make -j8
        sudo make install

4. Obstacle_msg package
------------------------

5. Cubicle_detect package
-------------------------

6. Build SSLAM package
----------------------
Clone the repository and catkin build

.. code-block:: bash

        catkin build sslam

