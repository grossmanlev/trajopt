# Test Scripts

This subdirectory contains most test and example scripts for our CS249r final project.

Installation instructions can be found in the main README located in the ``trajopt`` folder.

## Examples

### ac_reacher_7dof_mppi.py

This file contains methods to run the MPPI trajopt agent as well as train a Critic value network in conjunction (online learning).

Run the MPPI algorithm on the 7DOF arm:

```
python3 ac_reacher_7dof_mppi.py
```

Run the MPPI algorithm on the 7DOF arm and have a Critic value network learn online, using a replay buffer
```
python3 ac_reacher_7dof_mppi.py --train True
```

Run the MPPI algorithm, and have it learn using a learned Critic network as its cost function
```
python3 ac_reacher_7dof_mppi.py --critic data/initial_critic_converged/reacher_7dof_critic.pt
```

### offline_critic_reacher_7dof.py

This file trains a Critic value network using batch data collected previously from MPPI trajectories.

Run the Critic learning scheme using a pre-collected, pickled ReplayBuffer
```
python3 offline_critic_reacher_7dof.py data/good_replay/reacher_7dof_replay.pickle
```

### play_mppi_agent.py

This file takes in a pretrained, pickled MPPI agent file and displays the solution trajectory
```
python3 play_mppi_agent.py data/offline_critic_converged_agent/reacher_7dof_mppi.pickle
```