# BSD 3-Clause License
#
# Copyright (C) 2024, LAAS-CNRS.
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.

from typing import Any
import numpy as np
import crocoddyl
import pinocchio as pin
import mim_solvers

from param_parsers import ParamParser
import colmpc as col


class OCPPandaReachingColWithMultipleCol:
    """This class is creating a optimal control problem of a panda robot reaching for a target while taking a collision between a given previously given shape of the robot and an obstacle into consideration"""

    def __init__(
        self,
        rmodel: pin.Model,
        cmodel: pin.GeometryModel,
        TARGET_POSE: pin.SE3,
        x0: np.ndarray,
        pp: ParamParser,
    ) -> None:
        """Creating the class for optimal control problem of a panda robot reaching for a target while taking a collision between a given previously given shape of the robot and an obstacle into consideration.

        Args:
            rmodel (pin.Model): pinocchio Model of the robot
            cmodel (pin.GeometryModel): Collision model of the robot
            TARGET_POSE (pin.SE3): Pose of the target in WOLRD ref
            OBSTACLE_POSE (pin.SE3): Pose of the obstacle in the universe ref
            OBSTACLE_RADIUS (float): Radius of the obstacle
            T (int): Number of nodes in the trajectory
            dt (float): Time step between each node
            x0 (np.ndarray): Initial state of the problem
            WEIGHT_xREG (float, optional): State regulation weight. Defaults to 1e-1.
            WEIGHT_uREG (float, optional): Command regulation weight. Defaults to 1e-4.
            WEIGHT_GRIPPER_POSE (float, optional): End effector pose weight. Defaults to 10.
            SAFETY_THRESHOLD (float, optional): Safety threshold of collision avoidance. Defaults to 1e-2.
        """
        # Models of the robot
        self._rmodel = rmodel
        self._cmodel = cmodel

        self.pp = pp
        # Poses & dimensions of the target & obstacle
        self._TARGET_POSE = TARGET_POSE
        self._SAFETY_THRESHOLD = self.pp.get_safety_threshold()

        # Params of the problem
        self._T = self.pp.get_T()
        self._dt = self.pp.get_dt()
        self._x0 = x0

        # Weights
        self._WEIGHT_xREG = self.pp.get_W_xREG()
        self._WEIGHT_uREG = self.pp.get_W_uREG()
        self._WEIGHT_GRIPPER_POSE = self.pp.get_W_gripper_pose()
        self._WEIGHT_GRIPPER_POSE_TERM = self.pp.get_W_gripper_pose_term()

        # Data models
        self._rdata = rmodel.createData()
        self._cdata = cmodel.createData()

        # Frames
        self._endeff_frame = self._rmodel.getFrameId("panda2_leftfinger")

        # Making sure that the frame exists
        assert self._endeff_frame <= len(self._rmodel.frames)

    def create_OCP(self) -> "OCP":
        "Setting up croccodyl OCP"

        # Stat and actuation model
        self._state = crocoddyl.StateMultibody(self._rmodel)
        self._actuation = crocoddyl.ActuationModelFull(self._state)

        # Running & terminal cost models
        self._runningCostModel = crocoddyl.CostModelSum(self._state)
        self._terminalCostModel = crocoddyl.CostModelSum(self._state)

        ### Creation of cost terms

        # State Regularization cost
        xResidual = crocoddyl.ResidualModelState(self._state, self._x0)
        xRegCost = crocoddyl.CostModelResidual(self._state, xResidual)

        # Control Regularization cost
        uResidual = crocoddyl.ResidualModelControl(self._state)
        uRegCost = crocoddyl.CostModelResidual(self._state, uResidual)

        # End effector frame cost
        framePlacementResidual = crocoddyl.ResidualModelFrameTranslation(
            self._state,
            self._endeff_frame,
            self._TARGET_POSE.translation,
        )

        goalTrackingCost = crocoddyl.CostModelResidual(
            self._state, framePlacementResidual
        )

        # Obstacle cost with hard constraint
        self._runningConstraintModelManager = crocoddyl.ConstraintModelManager(
            self._state, self._actuation.nu
        )
        self._terminalConstraintModelManager = crocoddyl.ConstraintModelManager(
            self._state, self._actuation.nu
        )
        # Creating the residual

        for col_idx in range(len(self._cmodel.collisionPairs)):
            obstacleDistanceResidual = col.ResidualDistanceCollision(
                self._state, 7, self._cmodel, col_idx
            )
            # Creating the inequality constraint
            constraint = crocoddyl.ConstraintModelResidual(
                self._state,
                obstacleDistanceResidual,
                np.array([self._SAFETY_THRESHOLD]),
                np.array([np.inf]),
            )

            # Adding the constraint to the constraint manager
            self._runningConstraintModelManager.addConstraint(
                "col_" + str(col_idx), constraint
            )
            self._terminalConstraintModelManager.addConstraint(
                "col_term_" + str(col_idx), constraint
            )

        # Adding costs to the models
        self._runningCostModel.addCost("stateReg", xRegCost, self._WEIGHT_xREG)
        self._runningCostModel.addCost("ctrlRegGrav", uRegCost, self._WEIGHT_uREG)
        self._runningCostModel.addCost(
            "gripperPoseRM", goalTrackingCost, self._WEIGHT_GRIPPER_POSE
        )
        self._terminalCostModel.addCost("stateReg", xRegCost, self._WEIGHT_xREG)
        self._terminalCostModel.addCost(
            "gripperPose", goalTrackingCost, self._WEIGHT_GRIPPER_POSE_TERM
        )

        # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
        self._running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self._state,
            self._actuation,
            self._runningCostModel,
            self._runningConstraintModelManager,
        )
        self._terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self._state,
            self._actuation,
            self._terminalCostModel,
            self._terminalConstraintModelManager,
        )

        # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
        self._runningModel = crocoddyl.IntegratedActionModelEuler(
            self._running_DAM, self._dt
        )
        self._terminalModel = crocoddyl.IntegratedActionModelEuler(
            self._terminal_DAM, 0.0
        )

        self._runningModel.differential.armature = np.array(
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]
        )
        self._terminalModel.differential.armature = np.array(
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]
        )

        problem = crocoddyl.ShootingProblem(
            self._x0, [self._runningModel] * (self._T-1), self._terminalModel
        )
        # Create solver + callbacks
        # Define mim solver with inequalities constraints
        ocp = mim_solvers.SolverCSQP(problem)

        # Merit function
        ocp.use_filter_line_search = False

        # Parameters of the solver
        ocp.termination_tolerance = 1e-3
        ocp.max_qp_iters = 1000
        ocp.eps_abs = 1e-6
        ocp.eps_rel = 0

        ocp.with_callbacks = False

        return ocp
