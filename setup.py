from setuptools import setup, find_packages
import os

# Pybdlib can be installed with demonstration data. The demonstration data should be 
# located in the following folder:
data_path = '%s/pbdlib/data/'%(os.curdir)
# The script below will search this folder recursively for files that have the
# following extensions:
data_ext  = ['mat']
# The folder structure is copied when installing the data with the pybdlib module.

# Gather data files:
def gatherfiles(path,ext_list,fname):
    dlist = []
    # gather files located direclty in path: 
    print('path: ', path)
    file_names = ['%s%s'%(path,fn) for fn in os.listdir((path))
            if any(fn.endswith(ext) for ext in ext_list)]

    if len(file_names)>0:
        dlist.append((fname,file_names))

    # collect files located in subdirectories of path:
    for fn in os.listdir(path):
        if os.path.isdir('%s%s'%(path,fn)):
            # select first element in the list 
            tmp = gatherfiles('%s%s/'%(path,fn),ext_list,'%s/%s'%(fname,fn))
            if len(tmp) >0:
                dlist.append(tmp[0])
    return dlist

dlist = gatherfiles(data_path,data_ext,'pybdlib/data')

# Setup: 
setup(name='pbdlib',
      version='0.1',
      description='Programming by Demonstration module for Python',
      url='',
      author='Emmanuel Pignat',
      author_email='emmanuel.pignat@idiap.ch',
      license='MIT',
      packages=find_packages(),
      data_files = dlist,
      install_requires = ['numpy','scipy','matplotlib','sklearn', 'dtw', 'jupyter', 'enum', 'termcolor'],
      zip_safe=False)
