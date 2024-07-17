class Model:
    def __init__(self, x=0.0, y=0.0, theta=0.0, wheel_radius=0.05, axle_length=0.15):
        self.x = x
        self.y = y
        self.theta = theta
        self.wheel_radius = wheel_radius
        self.axle_length = axle_length

    def predict(self, u, dt):
        v, w = u
        x = self.x
        y = self.y
        theta = self.theta

        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt

        return x, y, theta