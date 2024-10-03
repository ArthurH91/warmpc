import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, nq, nv, nu):
        """
        Args:
            data (list): List A, where each element contains:
                - A tensor of 3 elements
                - A list containing T tensors of size (nq + nv)
                - A list of T - 1 tensors of size nq
            nq (int): Size of the second list elements (nq).
            nv (int): Size of the remaining elements in the (nq + nv) list.
        """
        self.data = data
        self.nq = nq
        self.nv = nv
        self.nu = nu

    def __len__(self):
        # Return the total number of elements in the dataset (i.e., length of list A)
        return len(self.data)

    def __getitem__(self, idx):
        # Get list A at the given index
        A = self.data[idx]
        
        # Unpack the elements
        target_pose = A[0]  # Tensor of 3 elements
        XS = A[1]   # List of T tensors of size (nq + nv)
        US = A[2]  # List of T - 1 tensors of size nq
        
        # Ensure XS has at least one tensor
        if len(XS) < 1:
            raise ValueError(f"Expected at least 1 tensor in XS, but got {len(XS)}.")
        
        # Input to the NN: 1D tensor of 3 elements and the first tensor of XS
        input_data = torch.cat((target_pose, XS[0]), dim=0).squeeze()

        # Use zip to pair XS[1:] and US directly, then concatenate each pair
        output_list = [torch.cat((x, u), dim=0) for x, u in zip(XS[1:], US)]

        # Perform a single concatenation of all tensors in the list
        output_data = torch.cat(output_list, dim=0).squeeze()
        return input_data, output_data
    
    def get_us_from_output(self, output):
        # Split the output tensor into T - 1 tensors of size nq
        out = torch.split(output, self.nq + self.nv + self.nu)
        us = [o[-self.nu:] for o in out]
        return torch.stack(us)

    def get_xs_from_input_output(self,input, output):
        # Split the output tensor into T - 1 tensors of size nq
        inp = torch.split(input, self.nq + self.nv)
        out = torch.split(output, self.nq + self.nv + self.nu)
        x0 = input[3:].reshape(1, len(input[3:]))
        xs_out = torch.stack([o[:self.nq + self.nv] for o in out])
        return torch.cat((x0, xs_out))

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
    nq, nv, nu = 7, 7, 7
    # Simulated example data (same as the previous one, but now tensors instead of NumPy arrays)
    data = torch.load('trajectories_test.pt')
    # Create the dataset and dataloader
    dataset = CustomDataset(data, nq=nq, nv=nv, nu=nu)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Example loop to see inputs and outputs
    for iter, (inputs, outputs) in enumerate(dataloader):
        inputs = inputs.squeeze()
        outputs = outputs.squeeze()
        
        targ = inputs[:3]
        x0 = inputs[3:]
        xs = dataset.get_xs_from_input_output(input=inputs, output=outputs)    
        if iter > 0:
            vis.viewer["goal" + str(iter-1)].delete()
            add_sphere_to_viewer(
                vis, "goal" + str(iter), 5e-2, targ.numpy(), color=0x006400
            )
            for x in xs:
                vis.display(x[:7].numpy())
                input()

