# Online-EM-Algo
Implementation of "Online EM Algorithm for Latent Data Models" (Cappé O., Moulines E., 2007).

## Run experiments
Run the following Python programs:
* `exp/cv_rate_experience.py` for comparing convergence trajectories between OL1 and OL06.
* `exp/methods_experiments.py` for comparing online EM with several step size and batch EM.

## Structure
* Jupyter notebook for testing: ``test.ipynb``
* Basic experiments are in ``exp/``
* Methods for Online EM implementation for Poisson Mixture model: ``src/online_EM.py``, ``src/mixture_poisson.py``
* Classical batch EM: ``src/batch_EM.py``

## Slides
Slides for this project work is available in this repository.