# MPPI

This subdirectory contains the code for the MPPI model predictive control
algorithm. Once each epoch, `train_step` is called, which takes in an optional
`critic` network and outputs a list of tuples contianing experience replay
data. These data are collected by calling `do_rollouts`, which loads the
environment at its current state, and rolls out Gaussianly-perturbed policies.
Once these policies are run, they are collected together and used in MPPI's
`update` equation.
