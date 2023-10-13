import numpy as np
import matplotlib.pyplot as plt
np.random.seed(101)

def latin_hypercube_sampling(points):
    """Generate 2D points using Latin Hypercube Sampling."""
    segments = np.linspace(0, 1, points + 1)
    sample_x = np.random.rand(points) * (segments[1] - segments[0]) + segments[:-1]
    sample_y = np.random.rand(points) * (segments[1] - segments[0]) + segments[:-1]
    np.random.shuffle(sample_x)
    np.random.shuffle(sample_y)
    return sample_x, sample_y

def log_uniform_lhs(points, min_value=0.5, max_value=2):
    """Generate 2D points using log-uniform Latin Hypercube Sampling."""
    
    # Generate Latin Hypercube Sample on a linear scale between 0 and 1
    x, y = latin_hypercube_sampling(points)
    
    # Transform to a log-uniform scale based on given range [min_value, max_value]
    x_log = np.exp(np.log(min_value) + x * (np.log(max_value) - np.log(min_value)))
    y_log = np.exp(np.log(min_value) + y * (np.log(max_value) - np.log(min_value)))

    return x_log, y_log


def plot_latin_hypercube(points, sampling_type="normal"):
    """Plot points generated using either normal or log-uniform Latin Hypercube Sampling."""
    
    plt.figure(figsize=(6, 5))
    plt.rcParams.update({'font.size': 12})
    
    if sampling_type == "normal":
        x, y = latin_hypercube_sampling(points)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
    elif sampling_type == "log_uniform":
        x, y = log_uniform_lhs(points)
        plt.xlim([np.min(x), np.max(x)])
        plt.ylim([np.min(y), np.max(y)])
        plt.xlabel('X (log-uniform)', fontsize=12)
        plt.ylabel('Y (log-uniform)', fontsize=12)
    else:
        raise ValueError("Invalid sampling_type. Choose either 'normal' or 'log_uniform'.")
    
    for i in np.linspace(0, 1, points + 1):
        plt.axhline(y=i, color='gray', linestyle='--')
        plt.axvline(x=i, color='gray', linestyle='--')

    plt.scatter(x, y)
    plt.title(f'2D {sampling_type.capitalize()} Latin Hypercube Sampling')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    npoints = 10
    plot_latin_hypercube(npoints, sampling_type="normal")
    plot_latin_hypercube(npoints, sampling_type="log_uniform")