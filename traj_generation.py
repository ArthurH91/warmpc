import os
import time
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
            results.append((pin.SE3ToXYZQUATtuple(targ),xs.tolist(),us.tolist()))
        return results
    
    def generate_trajs_fixed_target_random_initial_config(self, num_trajs= 10):
        
        results = []
        targ = self.pp.get_target_pose()    
        for i in range(num_trajs):
            q0 = pin.randomConfiguration(rmodel)
            x0 = np.concatenate((q0, np.zeros(self.rmodel.nv)))
            OCP_CREATOR = OCPPandaReachingColWithMultipleCol(rmodel, cmodel,TARGET_POSE=targ, x0=x0,pp= pp)
            OCP = OCP_CREATOR.create_OCP()    
            xs, us = self.PaO.compute_traj(q0,targ, OCP)
            results.append((pin.SE3ToXYZQUATtuple(targ),xs.tolist(),us.tolist()))
        return results
    
    
    
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
        
    for i,result in enumerate(results):
        print(result[0])
        add_sphere_to_viewer(
            vis, "goal" + str(i), 5e-2, result[0][:3], color=0x006400
        )
        for x in result[1]:
            vis.display(x[:7])
            input()
