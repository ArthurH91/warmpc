# Import all packages
from typing import List
import warnings
import numpy as np
import pydynorrt as pyrrt
import pinocchio as pin 


class PlanAndOptimize:
    """This class takes a robot model and a collision model, an initial configuration and a SE3 pose and a name of the frame that is wanted to reach the SE3
    in inputs, and returns a trajectory and its dynamics, linking the initial configuration to the SE3.
    """

    def __init__(
        self, rmodel: pin.Model, cmodel: pin.GeometryModel, ee_name: str, T: int
    ) -> None:
        """This class takes a robot model and a collision model, an initial configuration and a SE3 pose and a name of the frame that is wanted to reach the SE3
    in inputs, and returns a trajectory and its dynamics, linking the initial configuration to the SE3.

        Args:
            rmodel (pin.Model): pinocchio model of the robot.
            cmodel (pin.GeometryModel): collision model of the robot.
            ee_name (str): end-effector frame name.
            T (int): number of nodes of the trajectory.
        """

        # Models of the robot
        self._rmodel = rmodel
        self._cmodel = cmodel

        # Setting up the planning problem
        self._ee_name = ee_name

        # Number of nodes in the OCP at the end
        self._T = T 

        # Booleans describing the state of the planner
        self._set_IK = False
        self._solved_IK = False
        self._set_collision_planner = False
        self._set_lim = False
        self._set_planner = False
        self._shortcut_done = False

    def _set_limits(self, q_upper_bounds=None, q_lower_bounds=None) -> None:
        """Setting the bounds for the joints.

        Args:
            q_upper_bounds (np.ndarray, optional): Upper bounds of the joints. Defaults to None.
            q_lower_bounds (np.ndarray, optional): Lower bounds of the joints. Defaults to None.

        Raises:
            Warning: No upper bounds specified, will use the default ones specified in the URDF.
            Warning: No lower bounds specified, will use the default ones specified in the URDF.
        """
        if q_upper_bounds is None:
            self._q_upper_bounds = self._rmodel.upperPositionLimit
            warnings.warn(
                "No upper bounds specified, will use the default ones specified in the URDF."
            )
        else:
            assert np.all(
                q_upper_bounds < self._rmodel.upperPositionLimit
            ), "The upper bound is higher than the position limits set in the URDF."
            self._q_upper_bounds = q_upper_bounds
        if q_lower_bounds is None:
            self._q_lower_bounds = self._rmodel.lowerPositionLimit
            warnings.warn(
                "No lower bounds specified, will use the default ones specified in the URDF."
            )
        else:
            assert np.all(
                q_lower_bounds < self._rmodel.lowerPositionLimit
            ), "The lower bound is higher than the position limits set in the URDF."
            self._q_lower_bounds = q_lower_bounds

        self._set_lim = True

    def set_ik_solver(
        self,
        q_upper_bounds=None,
        q_lower_bounds=None,
        oMgoal=pin.SE3.Identity(),
        max_num_attempts=1000,
        max_time_ms=3000,
        max_solutions=20,
        max_it=1000,
        use_gradient_descent=False,
        use_finite_diff=False,
    ) -> "pyrrt.Pin_ik_solver":
        """Set the IK solver.

        Args:
            q_upper_bounds (np.ndarray, optional): Bounds of the robot. Defaults to None.
            q_lower_bounds (np.ndarray, optional): Bounds of the robot. Defaults to None.
            oMgoal (pin.SE3, optional): SE3 describing the postion to be reached for the target. Defaults to pin.SE3.Identity().
            max_num_attempts (int, optional): Number max of attemps for the solver. Defaults to 1000.
            max_time_ms (int, optional): Maximum time of solving. Defaults to 3000.
            max_solutions (int, optional): Number max of configurations found as solution. Defaults to 20.
            max_it (int, optional): Maximum iterations of the solver. Defaults to 1000.
            use_gradient_descent (bool, optional): Use gradient descent or 2nd order method. Defaults to False.
            use_finite_diff (bool, optional): Use finite differencing. Defaults to False.

        Returns:
            pyrrt.Pin_ik_solver: IK solver.
        """
        self._set_IK = True
        self._solver = pyrrt.Pin_ik_solver()
        pyrrt.set_pin_model_ik(self._solver, self._rmodel, self._cmodel)

        # Setting the bounds
        self._set_limits(q_upper_bounds=q_upper_bounds, q_lower_bounds=q_lower_bounds)

        self._solver.set_frame_positions([oMgoal.translation])
        self._solver.set_bounds(self._q_lower_bounds, self._q_upper_bounds)
        self._solver.set_max_num_attempts(max_num_attempts)
        self._solver.set_frame_names([self._ee_name])
        self._solver.set_max_time_ms(max_time_ms)
        self._solver.set_max_solutions(max_solutions)
        self._solver.set_max_it(max_it)
        self._solver.set_use_gradient_descent(use_gradient_descent)
        self._solver.set_use_finite_diff(use_finite_diff)

        return self._solver

    def solve_IK(self) -> List[np.ndarray]:
        """Solves the constrained IK problem.
        This IK problem is simply finding a or some configuration(s) that satisfy the collisions constraints & that has the end effector reaching a SE3 defined target.

        Raises:
            Warning: The IK problem hasn't been set. The defaults parameters will be used. To change them, call the method set_ik_solver first.

        Returns:
            list: list of configurations that are solutions to the IK problem
        """

        if not self._set_IK:
            self.set_ik_solver()
            warnings.warn(
                "The IK problem hasn't been set. The defaults parameters will be used. To change them, call the method set_ik_solver first."
            )
        if not self._set_collision_planner:
            self.set_collision_planner()

        if not self._set_lim:
            self._set_limits()

        out = self._solver.solve_ik()
        ik_solutions = self._solver.get_ik_solutions()

        self._solved_IK = True

        # NOTE: depending on the tolerances of the IK self._solver, some
        # configs might have very small collisions, and the RRT planner
        # will complain. Let's filter them.
        self._ik_solutions = [
            s
            for s in ik_solutions
            if self.cm.is_collision_free(s)
            and np.sum(s < self._q_lower_bounds) == 0
            and np.sum(s > self._q_upper_bounds) == 0
        ]
        return self._ik_solutions

    def set_collision_planner(self):
        """Set the collision planner.
        """
        self.cm = pyrrt.Collision_manager_pinocchio()
        pyrrt.set_pin_model(self.cm, self._rmodel, self._cmodel)
        self.cm.reset_counters()
        self._set_collision_planner = True

    def _generate_random_collision_free_configuration(self) -> np.ndarray:
        """Generate random feasible configurations.

        Returns:
            np.ndarray: feasible configuration.
        """
        if not self._set_collision_planner:
            self.set_collision_planner()
        if not self._set_lim:
            self._set_limits()

        valid_start = False
        while not valid_start:
            s = np.random.uniform(self._q_lower_bounds, self._q_upper_bounds)
            if self.cm.is_collision_free(s):
                valid_start = True
        return s
    
    def get_random_reachable_target(self) -> pin.SE3:
        """Generate a random reachable target.

        Returns:
            pin.SE3: Reachable target.
        """
        
        random_config = self._generate_random_collision_free_configuration()
        rdata = self._rmodel.createData()
        pin.forwardKinematics(self._rmodel, rdata, random_config)
        pin.updateFramePlacements(self._rmodel, rdata)
        oMee = rdata.oMf[self._rmodel.getFrameId(self._ee_name)]
        
        return oMee

    def init_planner(
        self,
        start=None,
        ik_solutions=[],
        q_upper_bounds=None,
        q_lower_bounds=None,
    ):
        """_summary_

        Args:
            start (_type_, optional): _description_. Defaults to None.
            ik_solutions (list, optional): _description_. Defaults to [].
            q_upper_bounds (_type_, optional): _description_. Defaults to None.
            q_lower_bounds (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        # Configuration of the RRT
        self._rrt = pyrrt.PlannerRRT_Rn()
        config_str = '''
        [RRT_options]
        max_it = 20000
        max_num_configs = 20000
        max_step = 1.0
        goal_tolerance = 0.001
        collision_resolution = 0.05
        goal_bias = 0.1
        store_all = false
        '''

        # Setting up start & goal
        if start is None:
            start = self._generate_random_collision_free_configuration()
        self._rrt.set_start(start)
        self._rrt.set_goal_list(ik_solutions)
        self._rrt.init(self._rmodel.nq)

        # If the bounds were not defined before
        if not self._set_lim:
            self._set_limits(
                q_upper_bounds=q_upper_bounds, q_lower_bounds=q_lower_bounds
            )

        self._rrt.set_bounds_to_state(self._q_lower_bounds, self._q_upper_bounds)
        self._rrt.set_is_collision_free_fun_from_manager(self.cm)
        self._rrt.read_cfg_string(config_str)

        self._set_planner = True
        return self._rrt

    def plan(self) -> List[np.ndarray]:
        """_summary_

        Returns:
            List[np.ndarray]: _description_
        """
        assert self._set_planner == True, "Set up the planner first"
        out = self._rrt.plan()
        assert out == pyrrt.TerminationCondition.GOAL_REACHED
        self._fine_path = self._rrt.get_fine_path(0.5)
        self._shortcut()
        self._sol = self._ressample_path()
        return self._sol

    def _shortcut(self) -> List[np.ndarray]:
        """_summary_

        Returns:
            List[np.ndarray]: _description_
        """
        self._shortcut_done = True
        path_shortcut = pyrrt.PathShortCut_RX()
        path_shortcut.init(self._rmodel.nq)
        path_shortcut.set_bounds_to_state(self._q_lower_bounds, self._q_upper_bounds)
        self.cm.reset_counters()
        path_shortcut.set_is_collision_free_fun_from_manager(self.cm)
        path_shortcut.set_initial_path(self._fine_path)
        path_shortcut.shortcut()
        self._new_path_fine = path_shortcut.get_fine_path(0.01)
        return self._new_path_fine

    def _ressample_path(self) -> List[np.ndarray]:
        """_summary_

        Raises:
            ValueError: _description_

        Returns:
            List[np.ndarray]: _description_
        """
        if not self._shortcut_done:
            raise ValueError("Call the short and the plan method before.")
        start = self._new_path_fine[0]
        end = self._new_path_fine[-1]

        T = self._T  # Number of nodes in the optimized trajectory
        T_remaining = T - 2  # T - start - end number of nodes

        T_new_path = len(self._new_path_fine) - 2

        step = T_new_path // T_remaining

        ressample_path = [start]
        for t in range(T_remaining):
            ressample_path.append(self._new_path_fine[step * (t + 1)])
        ressample_path.append(end)

        return ressample_path
    
    def optimize(self, OCP):
        """_summary_

        Args:
            OCP (_type_): _description_

        Returns:
            _type_: _description_
        """
        X_init = []
        
        for q in self._sol:
            X_init.append(np.concatenate((q, np.zeros(self._rmodel.nv))))
        U_init = OCP.problem.quasiStatic(X_init[:-1])
        OCP.solve(X_init, U_init)
        
        return OCP.xs, OCP.us
    
    def compute_traj(self, q0: np.ndarray, oMgoal: pin.SE3, OCP):
          
        self.set_ik_solver(oMgoal=oMgoal)
        sol = self.solve_IK()
        self.init_planner(start=q0, ik_solutions=sol)
        
        fine_path = self.plan()
        t = self._ressample_path()
        xs, us = self.optimize(OCP)
        
        return xs, us

        
if __name__ == "__main__":

    import os
    import pinocchio as pin

    from visualizer import create_viewer, add_sphere_to_viewer
    from wrapper_panda import PandaWrapper
    from create_ocp import OCPPandaReachingColWithMultipleCol
    from param_parsers import ParamParser

    # Creating the robot
    robot_wrapper = PandaWrapper(capsule=False)
    rmodel, cmodel, vmodel = robot_wrapper()

    yaml_path = os.path.join(os.path.dirname(__file__), "scenes.yaml")
    pp = ParamParser(yaml_path, 1)

    cmodel = pp.add_collisions(rmodel, cmodel)

    cdata = cmodel.createData()
    rdata = rmodel.createData()
    ocp_creation = OCPPandaReachingColWithMultipleCol(
        rmodel, cmodel, pp.get_target_pose(), pp.get_X0(), pp
    )
    OCP = ocp_creation.create_OCP()
    # Generating the meshcat visualizer
    vis = create_viewer(rmodel, cmodel, cmodel)
    add_sphere_to_viewer(
        vis, "goal", 5e-2, pp.get_target_pose().translation, color=0x006400
    )
    PaO = PlanAndOptimize(rmodel, cmodel, "panda2_hand_tcp", pp.get_T())
    xs, us = PaO.compute_traj(pp.get_initial_config(),pp.get_target_pose(), OCP)
    print("ready to visualize")
    while True:
        vis.display(pp.get_initial_config())
        input()
        for x in xs:
            vis.display(np.array(x[:7].tolist()))
            # time.sleep(1e-1)
            input()
        print("replay")
