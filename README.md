# overfitting-experiment

Requirements

Python = 2.7
Python dependencies:
  Numpy
  Sklearn
  Matplotlib

The easiest way to install them is via pip. Pip can be installed on Ubuntu with the default package manager:

sudo apt-get install pip

The dependencies can be further installed with:

sudo pip install numpy
sudo pip install sklearn
sudo pip install matplotlib

The project is composed of 4 files.

generate.py: constains helper functions to generate datasets
fit.py, fit2.py: files that run each experiment (stochastic noise and deterministic noise, respectively)
plot.py: plots the results for both experiments

fit.py and fit2.py can be run via

python <fit.py, fit2.py>

Each of them will run a set of experiments and dump the results in a pickle file with a randomly generated name. Results from the second experiment will have an underline (_) before the random string of numbers.
Next, plot.py can be run via

python plot.py

It will scan for all .pkl files in its folder and plot the two graphs.
