# Reference-guided, Value-based MPC

Installation instructions can be found in the main README located in the ``trajopt`` folder.

## Examples

### ac_reacher_7dof_mppi.py

This file runs the RVMPC algorithm on the 7DOF robotic reacher environment.

```
python3.7 ac_reacher_7dof_mppi.py --train --save
```

### humanoid_mppi.py

This file runs the MPPI algorithm on the humanoid environment.
NOTE: Still a work in progress getting an MPPI controller working on a
humanoid environment (defined in `envs/humanoid_env.py`).
```
python3.7 humanoid_mppi.py
```


### play_mppi_agent.py

This file takes in a pretrained, pickled MPPI agent file and displays the solution trajectory
```
python3 play_mppi_agent.py <path_to_agent_file>