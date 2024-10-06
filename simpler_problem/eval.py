import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch

import time 
import pinocchio as pin
from training_v3 import MLP, TorchDataset
from training import CustomDataset
# from model import Net
from wrapper_panda import PandaWrapper
from param_parsers import ParamParser
from visualizer import create_viewer, add_sphere_to_viewer

data = torch.load('data_3000.pt', weights_only=True)
# Paths to the model and data
model_path = "trained_model_box_.pth"

net = MLP(input_size=14,output_size=98)


# Load the model state
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")
else:
    print("Model file does not exist.")
    exit()

# Create dataset and dataloader
dataset = TorchDataset(data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Set the model to evaluation mode
net.eval()

# Initialize lists to store predictions and actual values
predictions = []
actuals = []
diff = []

# Make predictions and compare with actual values
with torch.no_grad():
    for inputs, actual in dataloader:
        output = net(inputs)
        predictions.append(output)
        actuals.append(actual)



robot_wrapper = PandaWrapper(capsule=False)
rmodel, cmodel, vmodel = robot_wrapper()

yaml_path = "scenes.yaml"
pp = ParamParser(yaml_path, 1)
cmodel = pp.add_collisions(rmodel, cmodel)

vis = create_viewer(rmodel, cmodel, cmodel)  
add_sphere_to_viewer(vis, "goal", 5e-2, pp.get_target_pose().translation, color=0x006400)
### INITIAL X0
qgoal = np.array([-8.32558671e-01,  1.95560355e+00, -6.67370548e-01,  7.99628366e-01,
        6.28067728e-01,  1.31252482e+00,  4.27264806e-01])
q0 = pin.randomConfiguration(rmodel)
targ = pp.get_target_pose().translation
vis.display(q0)

inputs = np.concatenate((q0,qgoal ))
with torch.no_grad():
    output = net(torch.tensor(inputs, dtype=torch.float32))

print("visualisation of the input given trajectory")


while True:
    print("visualisation of the NN given trajectory")
    vis.display(q0)
    input()
    for xs in output.split(rmodel.nq):
        vis.display(xs.numpy())
        input()
    print("replay")
