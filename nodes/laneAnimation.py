
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from math import comb

def bezier_curve(points, n=100):
    """
    Compute a Bezier curve from control points.
    Works for arbitrary order depending on number of points.
    """
    points = np.array(points)
    N = len(points) - 1

    def bernstein_poly(i, n, t):
        return comb(n, i) * (t**i) * (1 - t)**(n - i)

    t_vals = np.linspace(0, 1, n)
    curve = np.zeros((n, 2))
    for i in range(N+1):
        curve += np.outer(bernstein_poly(i, N, t_vals), points[i])
    return curve

def piecewise_bezier(points, n=100):
    """
    Generate multiple quadratic Bezier curves from overlapping triplets of points.
    """
    curves = []
    for i in range(len(points) - 2):
        triplet = points[i:i+3]
        curve = bezier_curve(triplet, n)
        curves.append(curve)
    return curves

def average_curve(curves):
    """
    Take pointwise average of all piecewise curves.
    Assumes they are sampled with the same number of points.
    """
    stacked = np.stack(curves, axis=0)   # shape (num_curves, n, 2)
    avg = np.mean(stacked, axis=0)       # shape (n, 2)
    return avg

def plot_curves(points, curves, avg_curve):
    points = np.array(points)
    plt.figure(figsize=(7,7))
    plt.plot(points[:,0], points[:,1], 'ro--', label="Control Points")

    for i, curve in enumerate(curves):
        plt.plot(curve[:,0], curve[:,1], 'g-', alpha=0.4)

    plt.plot(avg_curve[:,0], avg_curve[:,1], 'b-', linewidth=2, label="Smoothed Avg Curve")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Piecewise Bézier Curves Averaged into One Smooth Path")
    plt.show()

if __name__ == "__main__":
    # Example with 5 points
    control_points = np.array([[0,0], [2,3], [4,4], [6,3]])
    curves = piecewise_bezier(control_points, n=200)
    avg_curve = average_curve(curves)
    plot_curves(control_points, curves, avg_curve)
