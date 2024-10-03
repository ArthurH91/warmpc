import os
import time
import torch
import pinocchio as pin
import numpy as np
from visualizer import create_viewer, add_sphere_to_viewer
from wrapper_panda import PandaWrapper
from create_ocp import OCPPandaReachingColWithMultipleCol
from param_parsers import ParamParser
from plan_and_optimize import PlanAndOptimize


class TrajGeneration():
    
    def __init__(self, rmodel: pin.Model, cmodel: pin.GeometryModel, pp: ParamParser):
        self.rmodel = rmodel
        self.cmodel = cmodel
        self.pp = pp
        self.PaO = PlanAndOptimize(self.rmodel, self.cmodel, "panda2_hand_tcp", self.pp.get_T())

    def generate_trajs_random_target_fixed_initial_config(self, num_trajs= 10):
                
        results = []
        q0 = self.pp.get_initial_config()
        for i in range(num_trajs):
            targ = self.PaO.get_random_reachable_target()    
            OCP_CREATOR = OCPPandaReachingColWithMultipleCol(rmodel, cmodel,TARGET_POSE=targ, x0=pp.get_X0(),pp= pp)
            OCP = OCP_CREATOR.create_OCP()    
            xs, us = self.PaO.compute_traj(q0,targ, OCP)
            
            # Convert to PyTorch tensors and store
            target_tensor = torch.tensor(pin.SE3ToXYZQUAT(targ)[:3], dtype=torch.float32)  # 1D tensor (3 elements)
            xs_tensor = torch.tensor(xs, dtype=torch.float32)  # List of T 1D arrays of size (nq + nv)
            us_tensor = torch.tensor(us, dtype=torch.float32)  # List of T - 1 1D arrays of size nq

            results.append((target_tensor, xs_tensor, us_tensor))
        return results
    
    def generate_trajs_fixed_target_random_initial_config(self, num_trajs=10):
        """_summary_

        Args:
            num_trajs (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        results = []
        targ = self.pp.get_target_pose()    
        for i in range(num_trajs):
            q0 = pin.randomConfiguration(self.rmodel)  # Random configuration
            x0 = np.concatenate((q0, np.zeros(self.rmodel.nv)))
            OCP_CREATOR = OCPPandaReachingColWithMultipleCol(self.rmodel, self.cmodel, TARGET_POSE=targ, x0=x0, pp=self.pp)
            OCP = OCP_CREATOR.create_OCP()    
            xs, us = self.PaO.compute_traj(q0, targ, OCP)
            xs_array = np.vstack(xs)
            us_array = np.vstack(us)
            # Convert to PyTorch tensors and store
            target_tensor = torch.tensor(pin.SE3ToXYZQUAT(targ)[:3], dtype=torch.float32)  # 1D tensor (3 elements)
            xs_tensor = torch.tensor(xs_array, dtype=torch.float32)  # List of T 1D arrays of size (nq + nv)
            us_tensor = torch.tensor(us_array, dtype=torch.float32)  # List of T - 1 1D arrays of size nq

            results.append((target_tensor, xs_tensor, us_tensor))

        return results
    
    def save_trajs_as_tensors(self, trajs, filename='trajectories.pt'):
        """
        Store the generated trajectories as a tensor file using torch.save.
        
        Args:
            trajs (list): List of tuples (target_tensor, xs_tensor, us_tensor)
            filename (str): Name of the file to store the tensors.
        """
        torch.save(trajs, filename)
        print(f"Trajectories stored as {filename}.")
    
if __name__ == "__main__":
    
    import os
    import pinocchio as pin

    from visualizer import create_viewer, add_sphere_to_viewer
    from wrapper_panda import PandaWrapper
    from param_parsers import ParamParser

    # Creating the robot
    robot_wrapper = PandaWrapper(capsule=False)
    rmodel, cmodel, vmodel = robot_wrapper()

    yaml_path = os.path.join(os.path.dirname(__file__), "scenes.yaml")
    pp = ParamParser(yaml_path, 1)

    cmodel = pp.add_collisions(rmodel, cmodel)

    cdata = cmodel.createData()
    rdata = rmodel.createData()

    # Generating the meshcat visualizer
    vis = create_viewer(rmodel, cmodel, cmodel)    
    TG = TrajGeneration(rmodel, cmodel, pp)
    results = TG.generate_trajs_fixed_target_random_initial_config(num_trajs = 5)
    TG.save_trajs_as_tensors(results, 'trajectories_test.pt')
    for i,result in enumerate(results):
        if i > 0:
            vis.viewer["goal" + str(i-1)].delete()
        add_sphere_to_viewer(
            vis, "goal" + str(i), 5e-2, result[0].numpy(), color=0x006400
        )
        
        for x in result[1]:
            vis.display(x[:7].numpy())
            input()
