# Stella-VSLAM-Python-bindings
Python bindings for Stella Visual SLAM system.

Compiling the only cpp file makes a module for using Stella VSLAM from Python.
Two Python illustration programs are provided:

- test.py tests if the module is working.  No fireworks, only console messages, if it goes until "shutdown SLAM system", congratulations: Python is bound to Stella VSLAM.
- example.py is an example of how to use the module in your Python program.  No nice graphics, only console messages with the 4x4 pose matrix returned by Stella VSLAM and a view of the video.


# Installation for Linux
These instructions were tested on Ubuntu 22.04 with stella_vslam 0.3.3 from May 10, 2022.  Should work on any Ubuntu and Debian derivative, and any stella_vslam version from 0.22.  Won't work on OpenVSLAM.

You need to have installed:

- Stella VSLAM: [installation document](https://stella-cv.readthedocs.io/en/latest/installation.html)
- Python 3: is usually already installed on your Linux
- PyBind11: you can install it by terminal:

    sudo apt install pybind11-dev

The only file required to compile is `stella_vslam_bindings.cpp` .  You can download it or the whole repository, so you get the two test Python programs.

    g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -I/usr/local/include/stella_vslam/3rd/json/include -I/usr/local/include/eigen3 -I/usr/local/include/opencv4 ./stella_vslam_bindings.cpp -o stellavslam$(python3-config --extension-suffix) -lstellavslam

The output is the Python module bound to stella_vslam: one file with name like **stellavslam.cpython-310-x86_64-linux-gnu.so** you should move or copy to your Python project.

# Test
In order to test your module you need two files in your working folder:

- [equirectangular.yaml](https://github.com/stella-cv/stella_vslam/blob/main/example/aist/equirectangular.yaml)
- [orb_vocab.fbow](https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow)

Simply run test.py on a terminal:

    python3 test.py

You will see messages in the terminal:

- module details
- configuration content
- confirmation on vocabulary loaded
- stella_vslam starting and shutting down

If the test doesn't end in error, congratulations, your module works well.

# Example
The example shows how to use stella_vslam by feeding the system with images from a video.  You can easyly modify it to feed images from your webcam.  Keep in mind the configuration file must correspond to the feeding camera.

The example need the vocabulary file.  For a quick demo, you can use the same configuration file used in testing (equirectangular.yaml), and get a short video with that exact omnidirectional camera [from this zip file](https://drive.google.com/uc?export=download&id=1d8kADKWBptEqTF7jEVhKatBEdN7g0ikY).

Placing video.mp4 in the same folder, you can run on a terminal:

    python3 example.py

You will see de video on a window, and the pose matrix returned by stella_vslam on the terminal.

# License
Please refer to LICENSE.
In brief, no special requirements but mention the authors in your distributions.  So little, so easy, there are no excuses not to do it.
Keep in mind that Stella VSLAM may be considered a derivative work from ORB-SLAM2, so GPL licenses can apply to Stella VSLAM - but not to these bindings.

# Thanks

To [Squiro](https://github.com/Squiro), who found the incompatibility between [openvslam_bindings](https://github.com/AlejandroSilvestri/OpenVSLAM-Python-bindings) and stella_vslam, warned me and quickly elaborated and tested a [solution](https://github.com/Squiro/StellaVSLAM-Python-bindings) I'm using here.