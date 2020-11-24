# Environments

This subdirectory contains all code pertinent to the generating and defining
of project-specific, MuJoCo environments.

### reacher_env.py
This file contains a 7DOF point-to-point robotic arm reaching environment.
There are four separate reward structures (`dist` denotes the distance between
the robot's fingertip and the goal position):

* `dense`: reward equals `- 10.0 * dist - 0.25 * np.linalg.norm(self.data.qvel)`.
* `sparse`: `100` if `dist < 0.025`, `-100` if `dist > 0.8`, and `-10` otherwise.
* `tracking`: `-10.0 * dist - 0.25 * np.linalg.norm(self.data.qvel)`, where
`dist` is the l2 distance between the robot's fingertip and the fingertip position
of the reference trajectory at that given timestep.
* `cooling`: this is a combination of `sparse` and `tracking`, specifically
`(1 - alpha)*sparse + alpha*tracking`.

### humanoid_env.py
This file contains a number of Bullet-physics powered humanoid environments based
on the DeepMimic paper (e.g., `HumanoidDeepMimicWalkEnv`, `HumanoidDeepMimicSpinkickEnv`).
Unfortunately, I have not been able to get a tracking controller (with the same
general structure as the Reacher environment) to work. I have tried a number of
different tweaks to the tracking controller:

* `feetTrackingReward`, an l2 norm-based reward proportional to the distance
between the robot's feet and the reference foot position.
* `COMTrackingReward`, an l2 norm-based reward proportional to the distance
between the robot's chest (center-of-mass) and the reference COM.
* `AllTrackingReward`, an l2 norm-based reward proportional to the distance
between all robot joint positions and reference joint positions.
* `calculateGoalReward`, used just for spinkick. Implementation of the DeepMimic
"Strike" reward, which gives a positive reward for all states after the goal
location has been "struck" by the robot's foot.