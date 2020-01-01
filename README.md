# Online-EM-Algo
Implementation of "Online EM Algorithm for Latent Data Models" (Cappé O., Moulines E., 2007).

## Run experiments
Run the following Python programs:
* `cv_rate_experience.py` for comparing convergence trajectories between OL1 and OL06.
* `methods_experiments.py` for comparing online EM with several step size and batch EM.

## ToDo List

* Figure 4: plot the convergence of parameters using Online EM Algo.
* Check the sensitivity of Online EM Algo on the initialization.

## Remarks

### Poisson Mixture experiment
* To see convergence, need to choose very different lambdas for the true parameters, like [1, 10, 100].
* The Online EM Algo seems to be sensitive to initalization. Sometimes it converges toward the truth, sometimes it doesn't. Check this

