import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (torch.Tensor): Tensor of tuples containing the data to be used in the dataset. The tuple should contain the following elements:
                - input (torch.Tensor): Input tensor of shape (input_size)
                - output (torch.Tensor): Output tensor of shape (nq + nv + nu, T - 1)
                where: nq, nv and nu are the number of joint positions, velocities and control inputs, respectively and T is the number of time steps.
        """
        self.data = data
        
    def __len__(self):
        # Return the total number of elements in the dataset (i.e., length of the tensor)
        return len(self.data)



    def __getitem__(self, idx):
        # Get list A at the given index
        d = self.data[idx]

        # Return the input and output tensors
        input_data = d[0]
        output_data = torch.flatten(d[1])
        return input_data, output_data
    

if __name__ == "__main__":
    import os

    from visualizer import create_viewer, add_sphere_to_viewer
    from wrapper_panda import PandaWrapper
    from param_parsers import ParamParser


    # Creating the robot
    robot_wrapper = PandaWrapper(capsule=False)
    rmodel, cmodel, vmodel = robot_wrapper()

    yaml_path = os.path.join(os.path.dirname(__file__), "scenes.yaml")
    pp = ParamParser(yaml_path, 1)

    cmodel = pp.add_collisions(rmodel, cmodel)
    # Generating the meshcat visualizer
    vis = create_viewer(rmodel, cmodel, cmodel)    

    # Example usage
    # Simulated example data (same as the previous one, but now tensors instead of NumPy arrays)
    data = torch.load('trajectories_test.pt')
    # Create the dataset and dataloader
    dataset = CustomDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    nu = 7
    # Example loop to see inputs and outputs
    for iter, (inputs, outputs) in enumerate(dataloader):     
        inputs = inputs.squeeze()
        outputs = outputs.squeeze()  
        targ = inputs[:3]
        x0 = inputs[3:]
        q0 = x0[:7]
        T = pp.get_T()
        xs = []
        us = []
        for i in range(T-1):
            x = outputs[i * (rmodel.nq + rmodel.nv + nu) : (i) * (rmodel.nq + rmodel.nv + nu) + rmodel.nq + rmodel.nv]
            u = outputs[(i) * (rmodel.nq + rmodel.nv + nu) + rmodel.nq + rmodel.nv: (i + 1) * (rmodel.nq + rmodel.nv + nu)]
            xs.append(x)
            us.append(u)
        if iter > 0:
            vis.viewer["goal" + str(iter-1)].delete()
            add_sphere_to_viewer(
                vis, "goal" + str(iter), 5e-2, targ.numpy(), color=0x006400
            )
            for x in xs:
                vis.display(x[:7].numpy())
                input()

