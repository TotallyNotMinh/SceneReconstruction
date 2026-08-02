#!/bin/bash
cd SceneReconstruction/PartC
git clone https://github.com/TencentARC/InstantMesh.git
cd InstantMesh
pip install requirement.txt
pip uninstall -y flax   #this can cause problem in InstantMesh
