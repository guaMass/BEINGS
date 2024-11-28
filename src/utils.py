import math
import matplotlib.pyplot as plt
import numpy as np


def get_transformation_matrix(position, rotation):
    T = np.eye(4, 4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return np.linalg.inv(T.T)


def get_projection_matrix(fovx, fovy, znear=0.01, zfar=100.0):
    tanHalfFovX = math.tan(fovx * 0.5)
    tanHalfFovY = math.tan(fovy * 0.5)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P.T

def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531

    >>> angle_mod([-4.0])
    np.array(2.28318531)

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle

def rotation_matrix_z(angle):
    # angle = angle/180*np.pi
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]).round(7)

def rotation_matrix_y(angle):
    # angle = angle/180*np.pi
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]).round(7)

def rotation_matrix_x(angle):
    # angle = angle/180*np.pi
    return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]).round(7)

class ProbMap:
    def __init__(self, N, M, H, W):
        self.N = N
        self.M = M
        self.H = H
        self.W = W
        self.raster = np.full((N, M), 1.0 / (N * M))
        self.x_min = -W / 2
        self.x_max = W / 2
        self.y_min = -H / 2
        self.y_max = H / 2

    def is_within_bounds(self, x, y):
        """Check if the (x, y) coordinates are within the raster map."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def get_raster_indices(self, x, y):
        """Determine which row and column the (x, y) coordinates correspond to."""
        if not self.is_within_bounds(x, y):
            return -1,-1
            # raise ValueError("Coordinates are out of bounds")

        # Calculate the indices in the raster grid
        col = int((x - self.x_min) / self.W * self.M)
        row = int((self.y_max - y) / self.H * self.N)

        return row, col

    def plot_raster(self):
        """Visualize the raster map with values in the center of each cell."""
        fig, ax = plt.subplots()

        # Define real-world coordinates for grid lines
        x_edges = np.linspace(self.x_min, self.x_max, self.M + 1)
        y_edges = np.linspace(self.y_min, self.y_max, self.N + 1)

        # Create a meshgrid for plotting
        x, y = np.meshgrid(x_edges, y_edges)

        # Plot the raster map as pcolormesh with black grid lines
        cax = ax.pcolormesh(x, y, self.raster, cmap='viridis', alpha=0.5, edgecolors='k',linewidth=0.5)

        # Add a color bar
        fig.colorbar(cax, ax=ax)

        # Label each cell with the numeric value
        for i in range(self.N):
            for j in range(self.M):
                # Calculate the center of each cell
                cell_x = (x_edges[j] + x_edges[j+1]) / 2
                cell_y = (y_edges[i] + y_edges[i+1]) / 2
                ax.text(cell_x, cell_y, f'{self.raster[i, j]:.4f}', ha='center', va='center', color='k')

        # Set axis labels
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')

        # Set axis limits
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)

        plt.show()

    def get_prob_at(self, x, y):
        """Get the probability at the given (x, y) coordinates."""
        row, col = self.get_raster_indices(x, y)
        return self.raster[row, col]
    
    def update_prob_at(self,x,y,q):
        if not self.is_within_bounds(x, y):
            return -1,-1
        row, col = self.get_raster_indices(x, y)
        p = self.raster[row, col]
        new_grid = self.raster.copy()
        new_grid[row, col] = p*(1-q)/(1-p*q)
        mask = np.ones_like(self.raster).astype(bool)
        mask[row, col] = 0
        new_grid[mask] *= (1/(1-p*q))
        self.raster = new_grid
        return self.raster
    
    def add_confidence_to(self, x, y, confidence):
        """Add the confidence value q to the probability at the given (x, y) coordinates."""
        row, col = self.get_raster_indices(x, y)
        new_grid = self.raster.copy()
        new_grid[row, col] += confidence
        self.raster = new_grid/np.sum(new_grid)
        return self.raster