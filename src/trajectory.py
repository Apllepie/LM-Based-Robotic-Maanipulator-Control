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
    def __init__(self, center: list | np.ndarray, radius: float, duration: float):
        """center in meters [x, y, z], radius in meters, duration in seconds."""
        self.center = np.array(center)
        self.radius = radius
        self.duration = duration

    def generate_circle_waypoints(self, t: float) -> np.ndarray:
        angle = (2 * np.pi / self.duration) * t
        x = self.center[0] + self.radius * np.cos(angle)
        y = self.center[1] + self.radius * np.sin(angle)
        z = self.center[2]

        return np.array([x, y, z])