# Online-EM-Algo
Implementation of "Online EM Algorithm for Latent Data Models" (Cappé O., Moulines E., 2007).

## Run experiments
Run the following Python programs:
* `cv_rate_experience.py` for comparing convergence trajectories between OL1 and OL06.
* `methods_experiments.py` for comparing online EM with several step size and batch EM.

## Structure
* Jupyter notebook for testing: ``test.ipynb``
* Basic experiments: ``experiments.py``
* Methods for Online EM implementation for Poisson Mixture model: ``online_EM.py``, ``mixture_poisson.py``
* Classical batch EM: ``batch_EM.py``
