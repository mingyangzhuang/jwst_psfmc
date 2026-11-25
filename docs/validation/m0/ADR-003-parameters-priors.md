# ADR 003: parameter coordinates, priors and reproducibility

Status: accepted design; implementation belongs to M1 and later inference stages.

Legacy theta is [flux,dx,dy,bkg] with open uniform prior intervals.
prepare_for_fitting derives the flux upper bound from the data maximum, uses
an absolute lower bound (default -3), and generates walkers with an unseeded
default_rng. M0 records deterministic prior bounds and fixed-theta likelihoods;
walker positions and stochastic chains are deliberately excluded from goldens.
The signed-flux fixture explicitly widens the flux prior to -100. Default
legacy priors must not silently become priors for a new physical model.

New Parameter/ParameterSet must separate physical prior, sampling coordinate,
transform and Jacobian, free/fixed state and simple equality ties. Sampling
log(re) with a physical uniform-re prior requires a Jacobian in the transformed
log density. A log-uniform prior is a different choice, not the same transform.
Nautilus's unit-cube prior transform and emcee's log prior must induce the same
physical distribution. Fixed/tied parameters must not duplicate prior factors.
Round-trip, Jacobian and prior predictive tests precede generic inference.

New preparation and sampling must both accept explicit random state. A seed
for Nautilus alone does not seed legacy walker initialization or emcee.
Compare stochastic results by Monte Carlo uncertainty and convergence
diagnostics, not chain equality. Notebook execution alone establishes neither
posterior calibration nor convergence. M0 has no cross-run posterior comparison.
Persist actual prior bounds, parameter labels, transforms and initialization
configuration with results. Science-image host flux is nonnegative by declared
model choice; a difference-image point source can have genuinely signed flux.
