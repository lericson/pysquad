# Simulator for Quadrotors

... and other multicopters.

Written as part of my degree project, this software is based on PyQtGraph for
3D live viewing of the simulation, and Numba for high-speed matrix heavy
computation. This project includes a PID regulator-based controller inspired by
the PX4 flight stack by Meier et al.

It also includes an implementation of the iLQR algorithm, a trajectory
optimization method based on DDP. Trajectory optimization is actually what the
degree project is about.

## Setup

Prerequisites: Not Microsoft Windows. Python 3.x.

As this is definitely a package you're going to tinker with, I recommend
checking it out and installing a development version like so:

```sh
git clone https://github.com/lericson/pysquad
cd pysquad
pip3 install -e .
```

Perhaps slap on a virtualenv if you're that kind of a person.

## Rudimentary Demo

```sh
python3 -m squad.dyn squad.dyn.mc.Quad squad.px4.Agent
```

## FAQ

Q: There are questionable code practices in this software, and my OCD is itching.

A: Yes, it's a degree project, so focuses on the results and theory rather than
code quality. Fix the problems and send a pull request!

Q: You seem to have reinvented `argparse`. Why?

A: It was easier to implement.

Q: Why is there a FAQ, nobody asked these questions.

A: That's not a question. Perhaps it should be called a PAQ. Probably asked
question. Or MLAQ. Maximum likelihood asked question.

Q: Can I read the thesis you wrote on this stuff?

A: Let me just finish it first.
