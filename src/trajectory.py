import numpy as np



class Trajectory:
    def __init__(self, waypoints):
        self.waypoints = waypoints

    def get_waypoints(self):
        return self.waypoints

    def set_waypoints(self, waypoints):
        self.waypoints = waypoints

    def add_waypoint(self, waypoint):
        self.waypoints.append(waypoint)

    def remove_waypoint(self, index):
        if 0 <= index < len(self.waypoints):
            del self.waypoints[index]
        else:
            raise IndexError("Waypoint index out of range.")

    def get_length(self):
        length = 0.0
        for i in range(1, len(self.waypoints)):
            length += np.linalg.norm(np.array(self.waypoints[i]) - np.array(self.waypoints[i - 1]))
        return length


class CircleTrajectory(Trajectory):
    def __init__(self, center:float, radius:float, duration:float):
        """waypoints in meters, duration in seconds
        duration is the time it takes to complete one full circle"""

        self.center = np.array(center)
        self.radius = radius
        self.duration = duration


    def generate_circle_waypoints(self, t):
        angles = (2 * np.pi / self.duration) * t
        waypoints = []
        x = self.center[0] + self.radius * np.cos(angles)
        y = self.center[1] + self.radius * np.sin(angles)
        z = self.center[2]

        return np.array([x, y, z]).T  # Transpose to get shape (N, 3)