# pbdlib


Pbdlib is a python library for robot programming by demonstration. The goal is to provide users with an easy to use library of algorithms that can be used to learn movement primitives using probabilistic tools.

The algorithms implemented in the library are developed by the Learning and Interaction group at the Italian Institute of Technology (IIT) and the IDIAP Research Insitute in Switzerland. 

For more information see http://www.programming-by-demonstration.org.

## References

If you find these codes useful for your research, please acknowledge the authors by citing this academic publication.

[Learning adaptive dressing assistance from human demonstration](http://doi.org/10.1016/j.robot.2017.03.017)

Emmanuel Pignat, March 2016

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
    
    cd pbdlib/tutorial
    jupyter notebook

Then navigate through folders and click on desired notebook.

| Filename | Description |
|----------|-------------|
| pbdlib overview.ipynb| Overview of the main functionalities of pbdlib.|
| hmm for regression.ipynb| Demonstrate the advantages of encoding a time dependence in a regression problem, through an HMM structure.|

