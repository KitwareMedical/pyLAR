pyLAR
=================

pyLAR features Python implementations of a low-rank atlas-to-image
registration(LAR) framework and its applications in medical image analysis
and computer vision. The core machine learning technique is Robust PCA.

Subdirectory content:
* core --- two implementations of RPCA
* examples -- a couple of ipython notebook examples of running RPCA
* tests -- testing scripts of the core functionalities
* low_rank_atlas -- the study of using  RPCA in a low-rank atlas buildling framework
* eval_utils -- the utilitiy scripts to evaluate the results on the low rank atlas building framework

pyLAR contains implementation of the following paper:
```bibtex
@article{Liu14,
    author = {X.~Liu and M.~Niethammer and R.~Kwitt and M.~McCormick and S.~Aylward},
    title = {Low-Rank to the Rescue – Atlas-based Analyses in the Presence of Pathologies},
    year = 2014,
    journal = {MICCAI},
```

The implementations of two recent proposals for *robust PCA* can be found in the "core" subdirectory:
```bibtex
@article{Candes11a,
    author = {E.J.~Cand\'es and X.~Li and Y.~Ma and J.~Wright},
    title = {Robust Principal Component Analysis?},
    year = 2011,
    volume = 58,
    number = 3,
    journal = {J. ACM},
    pages = {1-37}}
```
and
```bibtex
@article{Xu12a,
    author = {H.~Xu and C.~Caramanis and S.~Sanghavi},
    title = {Robust {PCA} via Outlier Pursuit},
    journal = {IEEE Trans. Inf. Theory},
    volume = 59,
    number = 5,
    pages = {3047-3064},
    year = 2012}
```
Please cite these articles in case you use the code in "core". Note that the original
authors of those articles also provide MATLAB code. Further, the objectives
of the two works are different: Candes et al.'s approach assumes randomly
distributed corruptions throughout the dataset, while Xu et al.'s approach
assumes that full observations (i.e., column vectors of the data matrix) and
not just single entries are corrupted.

Requirements
------------

* [**numpy**](http://www.numpy.org/)
* [**SimpleITK**](http://www.simpleitk.org) [Optional]

Problem Statement(s)
--------------------

See references (above) for the exact problem formulations of Candes
et al. and Xu et al.

Example
-------

An illustrative example for Candes et al.'s RPCA approach is to use a
checkerboard image (provided under the `examples` directory) which is,
by definition, low-rank and corrupt that image with randomly distributed
outliers. The task is then to recover the low-rank part and thus obtain
a *clean* version of the checkerboard image (as well as the sparsity
pattern).

The `examples` directory contains an example (`ex1.py`) that demonstrates
exactly this scenario.  (**Note:** The example requires
[SimpleITK](http://www.simpleitk.org)'s python wrapping for image loading and
image writing; it should be easy to replace these parts with your favorite
image handling library, though).

Run the code with
```bash
python ex1.py checkerboard.png 0.3 /tmp/outlierImage.png /tmp/lowRank.png
```
Two images will be written: `/tmp/outlierImage.png` (i.e., the image *with*
outliers) and `/tmp/lowRank.png` (i.e., the *low-rank* recovered part).

Using the IPython Notebook
--------------------------

We provide an IPython notebook, ```pyrpca-Tutorial.ipynb``` which can be found
in the top-level directory of ```pyrpca```. It basically walks a new user through
the example implemented in ```ex1.py```.

The following instructions were tested on a Linux machine running
Ubuntu 12.04. We assume that you have ```virtualenv``` installed,
e.g., using ```apt-get install python-virtualenv```. Basically, we
create a virtual environment, install all the required packages
and eventually run the IPython notebook.

To vizualize the matplolib plots without using IPython Notebook, you 
need to make sure that matplotlib is built with a backend that allows 
plotting. By default, it is usually set to use 'Agg' as a backend, 
which only allows to save the plot, not to visualize it.

On Ubuntu 15.10, To build matplotlib with 'tkAgg', which allows 
interactive plotting, you need to install the following packages:

```bash
sudo apt-get install tcl-dev tk-dev python-tk python3-tk
```

Install matplotlib with pip only after installing these packages.

```bash
cd ~
mkdir tutorial-env
virtualenv ~/tutorial-env --no-site-packages
~/tutorial-env/bin/pip install ipython
~/tutorial-env/bin/pip install ipython[zmq]
~/tutorial-env/bin/pip install tornado
~/tutorial-env/bin/pip install numpy
~/tutorial-env/bin/pip install matplotlib
~/tutorial-env/bin/easy_install SimpleITK
```
Next, launch the IPython notebook:
```bash
~/tutorial-env/bin/ipython notebook --pylab=inline
```

If you get the following error message, just follow the instructions and 
write `%pylab inline` at the top of your notebook.

```bash
[E 15:41:01.741 NotebookApp] Support for specifying --pylab on the command line has been removed.
[E 15:41:01.741 NotebookApp] Please use `%pylab inline` or `%matplotlib inline` in the notebook itself.
```

License
=======

pyLAR is distributed under the Apache License Version 2.0 (see LICENSE.md)
