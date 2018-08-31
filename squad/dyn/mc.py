import numpy as np
from scipy.stats import norm as normal, uniform

from .. import quat
from ..utis import env_param
from .base import ODEModel

ivp_method = env_param('ivp_method', default='euler', cast=str)

class Quad(ODEModel):
    radius = 0.2  # m
    mass = 1.3  # kg
    volume = 4.0/3.0*np.pi*(radius**3)  # m^3
    density = mass/volume  # kg/m^3
    # I = mk^2 where k is the radius of gyration
    inertia = mass*(0.275)**2

    # Multicopter characteristics
    num_rotors = 4
    arm_length = radius
    rotor_angles = 2*np.pi*(np.r_[0:num_rotors] - 1/2)/num_rotors
    rotor_positions = arm_length*np.c_[np.cos(rotor_angles), np.sin(rotor_angles), np.zeros(num_rotors)]
    rotor_axes = np.repeat(np.c_[0, 0, 1], num_rotors, axis=0)
    rotor_rpm_max = 20000
    rotor_directions = (-1)**np.r_[0:num_rotors]
    # Thrust coefficients from https://www.bitcraze.io/2015/02/measuring-propeller-rpm-part-3/
    thrust_rpm_max = (mass + 5)*9.82
    thrust_coeffs = np.r_[+1.0942e-07, -2.1059e-04, +0.15417]
    thrust_coeffs *= thrust_rpm_max/np.polyval(thrust_coeffs, rotor_rpm_max)
    z_torque_per_rpm = 5e-4/rotor_rpm_max
    rotor_rpm_coeff = 6e1

    state_shape = (3 + 4 + 3 + 3 + num_rotors,)
    action_shape = (num_rotors,)

    friction_force = 1e-2
    friction_torque = 1e-5
    gravity = np.r_[0.0, 0.0, 9.82]

    #                     x  y  z qi qj qk qr vx vy vz wx wy wz w0 w1 w2 w3
    #x_sample_mean = np.r_[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #x_sample_len  = np.r_[4, 4, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 4, 0, 0, 0, 0]
    x_sample_mean = np.r_[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x_sample_std  = np.r_[1, 1, 1, 0, 0,.1,.1,.5,.5,.5,.1,.1,.2, 0, 0, 0, 0]
    state_dist = normal(x_sample_mean, x_sample_std)

    u_lower = np.zeros(num_rotors)
    u_upper = np.ones(num_rotors)
    action_dist = uniform(u_lower, u_upper)

    def sample_states(model, *a, **kw):
        X = super().sample_states(*a, **kw)
        # Sample orientation by making an axis rotation in the XY plane, then
        # rotate about z axis by heading. Remove Z axis from rotation.
        X[:, 3:7] /= np.linalg.norm(X[:, 3:7])
        return X

    from . import quadfast
    x_dot = staticmethod(quadfast.x_dot)
    step_eul = staticmethod(quadfast.step_eul)
    step_array = staticmethod(quadfast.step_array)
    derivatives = staticmethod(quadfast.derivatives)

    def step(m, x, u, t=0, dt=1/400):
        x = x.astype(dtype=np.float64)
        attqnorm = np.linalg.norm(x[3:7])
        assert attqnorm > 0.0, (x, u)
        x[3:7] /= attqnorm
        u = u.astype(np.float64, copy=False)
        u = np.clip(u, m.u_lower, m.u_upper)
        if ivp_method == 'scipy':
            return m.step_ivp(x, u, t=t, dt=dt)
        elif ivp_method == 'euler':
            x_ = m.step_eul(x, u, dt=dt)
            return t + dt, x_, 0.0
        else:
            raise ValueError(ivp_method)

    def state(self, *, position=(0, 0, 0), pitch=0.0, roll=0.0, yaw=0.0,
              velocity=(0, 0, 0), pitchrate=0.0, rollrate=0.0, yawrate=0.0,
              rpms=None, dtype=np.float64):
        position = np.asarray(position, dtype=dtype)
        attq = quat.from_rpy(roll, pitch, yaw).astype(dtype)
        linvel = np.asarray(velocity, dtype=dtype)
        angvel = np.asarray((rollrate, pitchrate, yawrate), dtype=dtype)
        rpms = np.asarray((0,)*self.num_rotors if rpms is None else rpms, dtype=dtype)
        return np.r_[position, attq, linvel, angvel, rpms]

    def state_rep(self, state):
        if state.ndim == 2:
            return '\n'.join(map(self.state_rep, state))
        px, py, pz, qi, qj, qk, qr, vx, vy, vz, wx, wy, wz, *_ = state
        Rr, Rp, Ry = quat.euler_rpy((qi, qj, qk, qr))
        return (f'[p: {px:+6.2e} {py:+6.2e} {pz:+6.2e}; R: {Rr:+6.2e} {Rp:+6.2e} {Ry:+6.2e}\n'
                f' v: {vx:+6.2e} {vy:+6.2e} {vz:+6.2e}; w: {wx:+6.2e} {wy:+6.2e} {wz:+6.2e}]')

class Quad2D(Quad):
    from . import qf2d
    x_dot = staticmethod(qf2d.x_dot)
    step_eul = staticmethod(qf2d.step_eul)
    step_array = staticmethod(qf2d.step_array)

    #                     x  y  z qi qj qk qr vx vy vz wx wy wz w0 w1 w2 w3
    x_sample_mean = np.r_[0, 0, 0, 0,.2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x_sample_std  = np.r_[1, 0, 1, 0,.1, 0,.1,.5, 0,.5, 0,.1, 0, 0, 0, 0, 0]
    state_dist = normal(x_sample_mean, x_sample_std)

class QuadMix(Quad):
    action_shape = (4,)
    u_lower = np.r_[-1, -1, -1,  0]
    u_upper = np.r_[+1, +1, +1, +1]
    step_array = None

    def __init__(model):
        from agents import px4
        model.ctrl = px4.Controller(model=model)

    def step(m, x, u, **kw):
        return super().step(x, m.ctrl.mixer(u), **kw)

class PointMassQuad(Quad):
    """Quadcopter model formulated as 3D accelerations for actions, where the
    accelerations are actualized using PID controllers."""

    action_shape = (3,)
    u_lower = np.r_[-1, -1,  0]
    u_upper = np.r_[+1, +1, +1]
    step_array = None

    def __init__(model):
        from agents.linear import PX4Controller
        model.ctrl = PX4Controller()

    def pwms(m, x, u):
        x[3:7] /= np.linalg.norm(x[3:7])
        pv_attq, pv_linvel, pv_angvel = x[3:7], x[7:10], x[10:13]
        sp_force = u  # m.ctrl.vel_ctrl(u, pv_linvel)
        sp_attq_thrust = m.ctrl.thrust_ctrl(sp_force, 0.0, pv_linvel, pv_attq)
        sp_attq, sp_thrust = sp_attq_thrust[0:4], sp_attq_thrust[4]
        sp_rates = m.ctrl.attq_ctrl(sp_attq, pv_attq)
        sp_alpha = m.ctrl.rates_ctrl(sp_rates, pv_angvel)
        return m.ctrl.mixer(np.r_[sp_alpha, sp_thrust])

    def step(m, x, u, t=0, dt=1/100, dt_ctrl=1/200):
        t0, x, R = t, x.copy(), 0.0
        step = super().step
        while t - t0 < dt - 1e-13:
            dt_step = min(dt_ctrl, dt - (t - t0))
            t, x, r = step(x, m.pwms(x, u), t=t, dt=dt_step)
            R += r
        return t, x, R

    ivp_solver_opts = {}  # 'atol': 1e-4, 'rtol': 1e-1}

if __name__ == "__main__":
    m=Quad()
    for i in range(10):
        print(m.state_rep(m.sample_states(1)))
