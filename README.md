# pbdlib


Pbdlib is a python library for robot programming by demonstration. The goal is to provide users with an easy to use library of algorithms that can be used to learn movement primitives using probabilistic tools.

This Python version of Pbdlib is maintained by Emmanuel Pignat at the Idiap Research Institute. Pbdlib is a collection of codes available in different languages and developed as a joint effort between the Idiap Research Institute and the Italian Institute of Technology (IIT). 

For more information see http://www.idiap.ch/software/pbdlib/.

## References

If you find these codes useful for your research, please acknowledge the authors by citing:

Pignat, E. and Calinon, S. (2017). [Learning adaptive dressing assistance from human demonstration](http://doi.org/10.1016/j.robot.2017.03.017). Robotics and Autonomous Systems 93, 61-75.


# Installation

Following these instructions install the library without copying the files, allowing to edit the sources files without reinstalling.

    git clone git@gitlab.idiap.ch:epignat/pbdlib.git
    cd pbdlib
    pip install -e .

If pip is not install, you can get it that way:

    sudo apt-get install python-pip

## Notebooks-tutorials

### Requirement
This requires to [install Jupyter](http://jupyter.org/install.html), which you should already have if you [installed Anaconda](https://www.continuum.io/downloads). Please follow the links for installation instructions.

### Launching the notebooks

Launch jupyter server with:

    jupyter notebook tutorial/

Then navigate through folders and click on desired notebook.

| Filename | Description |
|----------|-------------|
| pbdlib overview.ipynb| Overview of the main functionalities of pbdlib.|
| hmm for regression.ipynb| Demonstrate the advantages of encoding a time dependence in a regression problem, through an HMM structure.|

