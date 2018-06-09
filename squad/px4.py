import logging

import numpy as np
import numba as nb

from . import pid, quat
from .utis import cross, clip_norm, clip_all, clip_one
from .dyn.mc import Quad

log = logging.getLogger(__name__)

model = Quad()

class Controller():
    xy_p = np.r_[0.95, 0.95]
    vel_xy_max = 2.0  # m/s

    vel_xy_p = np.r_[0.09, 0.09]
    vel_xy_i = np.r_[5e-3, 5e-3]
    vel_xy_d = np.r_[5e-4, 5e-4]
    thrust_xy_max = 2  # m/s^2
    thrust_xy_max_int = 0.2  # m
    tilt_max = np.pi/4  # rad

    z_p = 0.95
    vel_z_max = 5.0  # m/s

    vel_z_p = 0.4
    vel_z_i = 0.02
    vel_z_d = 0.0
    thrust_z_max = 20  # # m/s^2
    thrust_z_max_int = 0.3  # m
    thrust_z_hover = 0.5

    # TODO Does not use feedforward rates: MC_{YAW,PITCH,ROLL}{RATE,}_FF

    roll_tc,    pitch_tc               =   0.2,   0.2
    roll_p,     pitch_p,     yaw_p     =   6.5,   6.5, 2.8
    rollrate_p, pitchrate_p, yawrate_p = 15e-2, 15e-2, 0.2
    rollrate_i, pitchrate_i, yawrate_i =  1e-2,  1e-2, 0.01
    rollrate_d, pitchrate_d, yawrate_d =  0e-3,  0e-3, 0.0
    att_tc = np.r_[0.2/roll_tc, 0.2/pitch_tc, 1.0]
    att_p = att_tc*np.r_[roll_p, pitch_p, yaw_p]
    rates_p = att_tc*np.r_[rollrate_p, pitchrate_p, yawrate_p]
    rates_i = np.r_[rollrate_i, pitchrate_i, yawrate_i]
    rates_d = np.r_[rollrate_d, pitchrate_d, yawrate_d]
    alpha_max = np.r_[np.pi/8, np.pi/8, np.pi/4]
    alpha_int_max = np.r_[np.pi/16, np.pi/16, np.pi/8]

    def __init__(ctrl, model=model):
        ctrl.xy_ctrl = pid.new(ctrl.xy_p, max_norm=ctrl.vel_xy_max)
        ctrl.z_ctrl = pid.new(ctrl.z_p, max_abs=ctrl.vel_z_max)
        ctrl.thrustf_xy_ctrl = pid.new(ctrl.vel_xy_p, ctrl.vel_xy_i, ctrl.vel_xy_d,
                                       max_norm=ctrl.thrust_xy_max,
                                       max_integral_norm=ctrl.thrust_xy_max_int)
        ctrl.thrustf_z_ctrl = pid.new(ctrl.vel_z_p, ctrl.vel_z_i, ctrl.vel_z_d,
                                      max_norm=ctrl.thrust_z_max,
                                      max_integral_norm=ctrl.thrust_z_max_int)
        ctrl.rates_ctrl = pid.new(ctrl.rates_p, ctrl.rates_i, ctrl.rates_d,
                                  max_abs=ctrl.alpha_max,
                                  max_integral_abs=ctrl.alpha_int_max)
        ctrl._init_mixer(model=model)

    def reset(ctrl, *a, **k):
        ctrl.xy_ctrl.reset()
        ctrl.z_ctrl.reset()
        ctrl.thrustf_xy_ctrl.reset()
        ctrl.thrustf_z_ctrl.reset()
        ctrl.rates_ctrl.reset()

    def state_ctrl(ctrl, sp, x):
        return ctrl(sp, (x[0:3], x[3:7], x[7:10], x[10:13]))

    def __call__(ctrl, sp, pv):
        sp_pos, sp_yaw = sp
        pv_pos, pv_attq, pv_linvel, pv_angvel = pv
        sp_vel = np.r_[ctrl.xy_ctrl.feed(sp_pos[0:2], pv_pos[0:2]),
                       ctrl.z_ctrl.feed(sp_pos[2], pv_pos[2])]
        sp_force = ctrl.vel_ctrl(sp_vel, pv_linvel)
        sp_attq_thrust = ctrl.thrust_ctrl(sp_force, sp_yaw, pv_linvel, pv_attq)
        sp_attq, sp_thrust = sp_attq_thrust[0:4], sp_attq_thrust[4]
        sp_rates = ctrl.attq_ctrl(sp_attq, pv_attq)
        sp_alpha = ctrl.rates_ctrl.feed(sp_rates, pv_angvel)
        controls = np.r_[sp_alpha, sp_thrust]
        pwms = ctrl.mixer(controls)
        return np.clip(pwms, 0.0, 1.0)

    def vel_ctrl(ctrl, sp_vel, pv_vel):
        # Calculate thrust force setpoint
        sp_thrustf  = np.r_[ctrl.thrustf_xy_ctrl.feed(sp_vel[0:2], pv_vel[0:2]),
                            ctrl.thrustf_z_ctrl.feed(sp_vel[2],    pv_vel[2])]
        sp_thrustf += np.r_[0, 0, ctrl.thrust_z_hover]

        # Limit max tilt
        thrust_xy_max = sp_thrustf[2] * np.tan(ctrl.tilt_max)
        sp_thrustf[0:2] = clip_norm(sp_thrustf[0:2], thrust_xy_max)

        return sp_thrustf

    @staticmethod
    @nb.njit('f8[:](f8[:], f8, f8[:], f8[:])', cache=True, nogil=True)
    def thrust_ctrl(sp_thrustf, sp_yaw, pv_vel, pv_attq):
        e_z = np.array((0.0, 0.0, 1.0))
        # Calculate desired total thrust amount in body z direction by
        # projecting the thrust force vector onto Z axis of body frame.
        R_z = quat.vecdot(pv_attq, e_z)
        sp_thrust = sp_thrustf @ R_z

        # Construct basis vectors of setpoint body frame.

        # Z axis from thrust setpoint
        sp_thrustf_norm = np.linalg.norm(sp_thrustf)
        if sp_thrustf_norm > 1e-10:
            body_z = sp_thrustf/sp_thrustf_norm
        else:
            body_z = e_z

        # X axis from yaw setpoint
        # Desired yaw direction in XY plane, rotated by 90°
        y_C = np.array((-np.sin(sp_yaw), np.cos(sp_yaw), 0.0))
        if abs(body_z[2]) > 1e-10:
            body_x = cross(y_C, body_z)
            if body_z[2] < 0:
                body_x = -body_x
            body_x /= np.linalg.norm(body_x)
        else:
            # Desired thrust is in XY plane, set X downside to construct
            # correct matrix, but yaw component will not be used actually
            body_x = e_z

        # Y axis is simply cross product of Z and X.
        body_y = cross(body_z, body_x)

        # Calculate setpoint rotation matrix and its quaternion
        sp_R = np.vstack((body_x, body_y, body_z)).T
        sp_attq_thrust = np.empty(5)
        sp_attq_thrust[0:4] = quat.from_rotmat(sp_R)
        sp_attq_thrust[4] = sp_thrust
        return sp_attq_thrust

    @staticmethod
    @nb.njit(cache=True, nogil=True)
    def attq_ctrl(sp_attq, pv_attq, att_p=att_p):
        e_z = np.array((0.0, 0.0, 1.0))
        # Z axis of current attitude and setpoint attitude
        R_z    = quat.vecdot(pv_attq, e_z)
        R_sp_z = quat.vecdot(sp_attq, e_z)

        # Error in roll and pitch as a vector, e_R(2) = 0.
        e_R = quat.vecdot(quat.inv(pv_attq), cross(R_z, R_sp_z))
        e_R_z_sin = np.linalg.norm(e_R)
        e_R_z_cos = R_z @ R_sp_z

        if e_R_z_sin > 1e-15:
            e_R_z_angle = np.arctan2(e_R_z_sin, e_R_z_cos)
            e_R_z_axis = e_R / e_R_z_sin
            e_R = e_R_z_axis * e_R_z_angle
            # Cross product matrix for e_R_z_axis.
            e_cp = np.array(((             0, -e_R_z_axis[2],  e_R_z_axis[1]),
                             ( e_R_z_axis[2],              0, -e_R_z_axis[0]),
                             (-e_R_z_axis[1],  e_R_z_axis[0],              0)))
            R = np.eye(3) + e_cp*e_R_z_sin + e_cp@e_cp*(1 - e_R_z_cos)
            rp_attq = quat.dot(pv_attq, quat.from_rotmat(R))
        else:
            # No roll/pitch
            rp_attq = pv_attq

        # Attitude roll/pitch error
        err_rp_q   = quat.dot(sp_attq, quat.inv(rp_attq))

        # Weight for yaw control. Higher roll/pitch error, less yaw correction.
        yaw_w = e_R_z_cos**2

        # Set yaw error from angle between R_rp(e_x) and R_sp(e_x).
        e_x    = quat.vecdot(err_rp_q, np.array((1.0, 0.0, 0.0)))
        e_R[2] = np.arctan2(e_x[1], e_x[0]) * yaw_w

        if e_R_z_cos < 0:
            # For large thrust vector rotations use another rotation method:
            # calculate angle and axis for R -> R_sp rotation directly.
            errq = quat.dot(quat.inv(pv_attq), sp_attq)
            if errq[0] >= 0:
                e_R_d = 2.0*errq[0:3]
            else:
                e_R_d = -2.0*errq[0:3]

            # Unclear why but direct_w = e_R_z_cos**4.
            direct_w = yaw_w*(e_R_z_cos**2)
            e_R = (1 - direct_w)*e_R + direct_w*e_R_d

        return att_p*e_R

    def _init_mixer(ctrl, model=model):
        # Simplified version of px4's mixing logic. Each rotor *i* is assumed
        # linear in frequency *f* to torque *t* and force *f*:
        #   T f = f (a(e_z x r) + (-1)^i b e_z)
        # where a = 1, b = 0.05 are weights for roll/pitch torque and yaw torque,
        # and *r* is the position of the rotor, and thrust force
        #   F f = f a e_z.
        # Consider the matrix a where each column is f_i and t_i. This matrix
        # projects rotor frequencies to thrust forces and torques. Taking the
        # pseudo-inverse of this matrix allows us to calculate frequencies from
        # thrust forces and torques:
        #   f = a^-1 [f, t].
        Ct, Cm = 1.0, 0.05
        Ct, Cm = 0.5, 0.25
        Am  = Ct*np.cross(model.rotor_positions, model.rotor_axes)
        Am += Cm*model.rotor_directions[:, None]*model.rotor_axes
        At  = Ct*model.rotor_axes
        A = np.hstack((Am, At)).T
        B = np.linalg.pinv(A)

        @nb.njit('f8[:](f8[::1])', cache=True, nogil=True)
        def mixer(controls):
            roll, pitch, yaw = clip_all(controls[0:3], -1, +1)
            thrust           = clip_one(controls[3],    0, +1)
            outputs = B @ np.array((roll, pitch, yaw, 0, 0, thrust))
            #print('controls:', np.r_[roll, pitch, yaw, thrust])
            #print('outputs: ', outputs)

            # Odd boosting scheme I don't understand
            boost = 0
            roll_pitch_scale = 1
            thrust_increase_factor = 1.5
            thrust_decrease_factor = 0.6
            min_out, max_out = outputs.min(), outputs.max()

            if min_out < 0 and max_out < 1 and -min_out <= 1 - max_out:
                max_thrust_diff = thrust*(thrust_increase_factor - 1)
                if max_thrust_diff >= -min_out:
                    boost = -min_out
                else:
                    boost = max_thrust_diff
                    roll_pitch_scale = (thrust + boost) / (thrust - min_out)

            elif max_out > 1 and min_out > 0 and min_out >= max_out - 1:
                max_thrust_diff = thrust*(1 - thrust_decrease_factor)
                if max_thrust_diff >= max_out - 1:
                    boost = -(max_out - 1)
                else:
                    boost = -max_thrust_diff
                    roll_pitch_scale = (1 - (thrust + boost)) / (max_out - thrust)

            elif min_out < 0 and max_out < 1 and -min_out > 1 - max_out:
                max_thrust_diff = thrust*(thrust_increase_factor - 1)
                boost = clip_one(-min_out - (1 - max_out) / 2, 0, max_thrust_diff)
                roll_pitch_scale = (thrust + boost) / (thrust - min_out)

            elif max_out > 1 and min_out > 0 and min_out < max_out - 1:
                max_thrust_diff = thrust*(1 - thrust_decrease_factor)
                boost = clip_one(-(max_out - 1 - min_out) / 2, -max_thrust_diff, 0)
                roll_pitch_scale = (1 - (thrust + boost)) / (max_out - thrust)

            elif min_out < 0 and max_out > 1:
                max_thrust_diff = thrust*(thrust_increase_factor - 1)
                boost = clip_one(-(max_out - 1 + min_out) / 2,
                                 thrust*(thrust_decrease_factor - 1),
                                 thrust*(thrust_increase_factor - 1))
                roll_pitch_scale = (thrust + boost) / (thrust - min_out)

            # NOTE There is some yaw mixing logic in the PX4 that we skipped.

            # Recompute with boosted and scaled controls
            controls = np.array((roll*roll_pitch_scale, pitch*roll_pitch_scale,
                                 yaw, 0, 0, thrust + boost))
            outputs = B @ controls
            #print('output2: ', outputs)
            return clip_all(outputs, 0.0, 1.0)

        ctrl.mixer = mixer

class Agent():
    target_pos = np.r_[0.0, 0.0, 0.0]
    target_yaw = 0.0
    ctrl = Controller(model)

    def __init__(agent, **kw):
        if kw:
            log.warn('reinstantiating controller')
            agent.ctrl = Controller(**kw)

    def __call__(agent, state):
        sp = agent.target_pos, agent.target_yaw
        return agent.ctrl.state_ctrl(sp, state)

    def reset(agent):
        agent.ctrl.reset()

class MixerAgent(Agent):
    def __call__(agent, x):
        ctrl = agent.ctrl
        sp_pos, sp_yaw = agent.target_pos, agent.target_yaw
        pv_pos, pv_attq, pv_linvel, pv_angvel = x[0:3], x[3:7], x[7:10], x[10:13]
        sp_vel = np.r_[ctrl.xy_ctrl(sp_pos[0:2], pv_pos[0:2]),
                       ctrl.z_ctrl(sp_pos[2], pv_pos[2])]
        sp_force = ctrl.vel_ctrl(sp_vel, pv_linvel)
        sp_attq_thrust = ctrl.thrust_ctrl(sp_force, sp_yaw, pv_linvel, pv_attq)
        sp_attq, sp_thrust = sp_attq_thrust[0:4], sp_attq_thrust[4]
        sp_rates = ctrl.attq_ctrl(sp_attq, pv_attq)
        sp_alpha = ctrl.rates_ctrl(sp_rates, pv_angvel)
        return np.r_[sp_alpha, sp_thrust]
