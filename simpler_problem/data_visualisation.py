import torch 
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

from wrapper_panda import PandaWrapper
from param_parsers import ParamParser

data = torch.load('trajectories_10.pt', weights_only=True)

### Filling the lists of trajectories
targ_list = []
traj_list = []
for d in data:
    targ = d[0][:3]
    q0 = d[0][3:].reshape(1, 7)
    traj =torch.cat((q0, d[1]))
    targ_list.append(targ)
    traj_list.append(traj)
    
    
### Creating the robot models and scenes

robot_wrapper = PandaWrapper(capsule=False)
rmodel, cmodel, vmodel = robot_wrapper()

yaml_path = "scenes.yaml"
pp = ParamParser(yaml_path, 1)
cmodel = pp.add_collisions(rmodel, cmodel)

rdata = rmodel.createData()
cdata = cmodel.createData()

### Plotting the position of the end effector for each trajectory

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, traj in enumerate(traj_list):
    for iter, q in enumerate(traj):
        pin.forwardKinematics(rmodel, rdata, q.numpy())
        pin.updateFramePlacements(rmodel, rdata)
        pos = rdata.oMf[rmodel.getFrameId("panda2_hand_tcp")].translation
        if i == 0 and iter == 0:
            legend_start = 'Start of trajectory'
        elif i ==0 and iter == 1:
                legend_traj = 'Nodes of trajectory'
        else:
            legend_start = ''
            legend_traj =  ''
        if iter == 0:
            ax.scatter(pos[0], pos[1], pos[2], c='g', marker='o', label=legend_start)
        else:
            ax.scatter(pos[0], pos[1], pos[2], c='r', marker='o', label=legend_traj)
    if i == 0:
        legend_targ = 'Target'
    else:
        legend_targ = ''
    ax.scatter(targ_list[i][0], targ_list[i][1], targ_list[i][2], c='b', marker='x', label=legend_targ)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('End effector position for each trajectory')
ax.legend()
plt.show()