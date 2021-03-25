class TrajectoryIterationCallback:
    def __init__(self, f):
        self.f = f
        self.trajectory = []

    def __call__(self, x, **kwargs):
        self.trajectory.append((x, self.f(x)))
