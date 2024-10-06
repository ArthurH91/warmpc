import os
import torch
import pinocchio as pin
import numpy as np
from visualizer import create_viewer, add_sphere_to_viewer
from wrapper_panda import PandaWrapper
from create_ocp import OCPPandaReachingColWithMultipleCol
from param_parsers import ParamParser
from plan_and_optimize import PlanAndOptimize

from typing import Tuple

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Define custom progress bar
progress_bar = Progress(
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)

class TrajGeneration():
    
    def __init__(self, rmodel: pin.Model, cmodel: pin.GeometryModel, pp: ParamParser):
        self.rmodel = rmodel
        self.cmodel = cmodel
        self.pp = pp
        self.PaO = PlanAndOptimize(self.rmodel, self.cmodel, "panda2_hand_tcp", self.pp.get_T())


    def generate_traj(self, X0, targ):
        """Helper function to generate a single trajectory"""
        try:
            OCP_CREATOR = OCPPandaReachingColWithMultipleCol(self.rmodel, self.cmodel, TARGET_POSE=targ, x0=X0, pp=self.pp)
            OCP = OCP_CREATOR.create_OCP()
            xs, _ = self.PaO.compute_traj(X0[:7], targ, OCP) # Here the q0 is for the initial config. The velocity is encompassed in the OCP.
            # Convert to PyTorch tensors and store
            target = pin.SE3ToXYZQUAT(targ)[:3]
            inputs, outputs = self.from_targ_xs_to_input_output(target, xs)
            return inputs, outputs

        except Exception as e:
            print(f"Failed to generate trajectory. Error: {str(e)}")
            return None, None

    def generate_trajs_random_target_fixed_initial_config(self, num_trajs= 10):
        results = []
        q0 = self.pp.get_initial_config()
        with progress_bar as p:
            for i in p.track(range(num_trajs)):
                targ = self.PaO.get_random_reachable_target()  
                X0 = np.concatenate((q0, np.zeros(self.rmodel.nv)))
                input, output = self.generate_traj(X0, targ)
                if output is not None:
                    results.append((input, output))
                else:
                    i -= 1
        return results
    
    def generate_trajs_fixed_target_random_initial_config(self, num_trajs=10):
        """_summary_

        Args:
            num_trajs (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        results = []      
        targ =  self.pp.get_target_pose()   
        with progress_bar as p:
            for i in p.track(range(num_trajs)):
                q0 = pin.randomConfiguration(self.rmodel)
                X0 = np.concatenate((q0, np.zeros(self.rmodel.nv)))
                input, output = self.generate_traj(X0, targ)
                if output is not None:
                    results.append((input, output))
                else:
                    i -= 1
        return results
    
    def from_targ_xs_to_input_output(self, targ: np.ndarray, xs: list) -> Tuple[torch.Tensor, torch.Tensor]:
        
        X0 = xs[0]
        inputs_tensor = torch.tensor(np.concatenate((targ,X0[:7])), dtype=torch.float32)
        outputs = np.zeros((self.pp.get_T() - 1, self.rmodel.nq)) #7 is dim of Nu
        for iter, X in enumerate(xs[1:]):
            outputs[iter] = X[:7]
        outputs_tensor = torch.tensor(outputs, dtype=torch.float32)
        return inputs_tensor, torch.flatten(outputs_tensor)
    
    
    def save_results_as_tensors(self, results, filename='trajectories.pt'):
        """
        Store the generated results as a tensor file using torch.save.
        
        Args:
            results (list): List of tuples (target_tensor, xs_tensor, us_tensor)
            filename (str): Name of the file to store the tensors.
        """
        torch.save(results, filename)
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
    results = TG.generate_trajs_fixed_target_random_initial_config(num_trajs = 3000)
    TG.save_results_as_tensors(results, 'trajectories_3000.pt')
    for i,result in enumerate(results):
        if i > 0:
            vis.viewer["goal" + str(i-1)].delete()
        add_sphere_to_viewer(
            vis, "goal" + str(i), 5e-2, result[0].numpy(), color=0x006400
        )
        
        for x in torch.split(result[1], rmodel.nq):
            vis.display(x[:7].numpy())
            input()
