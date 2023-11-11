# README

# INTRODUCTION

This codebase is very expansive, and only part of the development work has been completed by our team. The files this team has updated / created are as follows:

- version_2/brlcad/src/rt/train_neural.cpp
- version_2/brlcad/src/rt/neural.py
- version_2/brlcad/src/rt/CMakeLists.txt

The purpose of these files is to develop training data for a neural network by performing a ray trace around the object from different perspectives and storing it into a file. The neural network is then trained on this input data. The work to gather the training data can be found in train_neural.cpp, whereas the actual implementation of the model can be found in neural.py. 

# External dependencies

- Python
- PyTorch
- TorchVision

#Installation:

Download the code using the following:

"git clone https://github.com/FA23-CSCE482-capstone-classroom/github-setup-capstone-brl-cad-neural-rendering.git"

# Compilation

Follow the BRL-CAD compilation instructions to compile the code. This is a very extensive process and the instructions can be found here: https://brlcad.org/wiki/Compiling

#Execution

There is a directory called "testing_database at the root of this GitHub repository. Copy and paste the "tank.g" file into the version_2/brlcad/build directory once you have successfully compiled BRL-CAD. 


To execute the code, remain in the version_2/brlcad/build directory and run "bin/rt_trainneural tank.g default.1.4.b.o.s"

This will begin the process of developing training data and train the model on the BRL-CAD database you have specified (which we have provided as example data). 
