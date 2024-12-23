from collections import deque, Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps


def load_config(file_path):
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '%' in line:
                value, key = line.split('%')
                params[key.strip()] = float(value.strip())
    return params

def initialize_lattice(L, p):
    lattice = np.random.rand(L, L) < p
    return lattice.astype(int)

def burning_method(lattice):
    L = lattice.shape[0]
    labels = np.zeros_like(lattice, dtype=int)
    t = 2

    for j in range(L):
        if lattice[0, j] == 1:
            labels[0, j] = t

    while True:
        new_burning = False
        for i in range(L):
            for j in range(L):
                if labels[i, j] == t:
                    for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                        if 0 <= ni < L and 0 <= nj < L and lattice[ni, nj] == 1 and labels[ni, nj] == 0:
                            labels[ni, nj] = t + 1
                            new_burning = True
                            if ni == L - 1:
                                return True, labels

        if not new_burning:
            break
        t += 1

    return False, labels

def max_cluster_size(lattice):
    rows, cols = lattice.shape
    visited = np.zeros_like(lattice, dtype=bool)

    def calculate_cluster_size(row, col):
        stack = [(row, col)]
        cluster_size = 0
        while stack:
            r, c = stack.pop()
            if 0 <= r < rows and 0 <= c < cols and lattice[r, c] == 1 and not visited[r, c]:
                visited[r, c] = True
                cluster_size += 1
                stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
        return cluster_size

    max_size = 0
    for row in range(rows):
        for col in range(cols):
            if lattice[row, col] == 1 and not visited[row, col]:
                cluster_size = calculate_cluster_size(row, col)
                max_size = max(max_size, cluster_size)
    return max_size

def hoshen_kopelman(lattice, update_labels=False):
    L = lattice.shape[0]
    labels = np.zeros_like(lattice, dtype=int)
    label_map = {}
    current_label = 2

    def find_root(label):
        root = label
        while label_map[root] < 0:
            root = -label_map[root]
        return root

    for i in range(L):
        for j in range(L):
            if lattice[i, j] == 1:
                neighbors = []

                if i > 0 and labels[i - 1, j] > 0:
                    neighbors.append(labels[i - 1, j])
                if j > 0 and labels[i, j - 1] > 0:
                    neighbors.append(labels[i, j - 1])

                if not neighbors:
                    labels[i, j] = current_label
                    label_map[current_label] = 1
                    current_label += 1
                else:
                    root_labels = [find_root(n) for n in neighbors]
                    primary_label = min(root_labels)
                    labels[i, j] = primary_label

                    label_map[primary_label] += 1

                    for root_label in root_labels:
                        if root_label != primary_label:
                            label_map[primary_label] += label_map[root_label]
                            label_map[root_label] = -primary_label

    if update_labels:
        for i in range(L):
            for j in range(L):
                if labels[i, j] > 0:
                    labels[i, j] = find_root(labels[i, j])

    cluster_sizes = Counter(
        size for size in label_map.values() if size > 0
    )

    return labels, cluster_sizes

def cluster_size_distribution(label_sizes):
    distribution = {}
    for size in label_sizes.values():
        if size not in distribution:
            distribution[size] = 0
        distribution[size] += 1
    return distribution

def monte_carlo(L, T, p0, dp, pk):
    # Prepare filenames for output files
    output_file = f"Ave-L{L}T{T}.txt"
    dist_file = f"Dist-p{p0}L{L}T{T}.txt"

    # Arrays to store the results
    p_values = np.arange(p0,pk + dp, dp)
    Pf, smax, n_s = [], [], []
    results = []

    for p in p_values:
        percolation_count = 0
        total_smax = 0
        cluster_sizes = Counter()

        for _ in range(T):
            lattice = initialize_lattice(L, p)
            # Check percolation
            if burning_method(lattice):
                percolation_count += 1
            # Find clusters and calculate max cluster size
            total_smax += max_cluster_size(lattice)
            labels, label_sizes = hoshen_kopelman(lattice)
            cluster_sizes.update(label_sizes)


        Pf.append(percolation_count/T)
        smax.append(total_smax/T)
        n_s.append(cluster_sizes)

        # Write the cluster size distribution to file
        with open(dist_file, 'a') as f:
            for size, count in cluster_sizes.items():
                if size > 0:
                    f.write(f"{size}  {count}\n")


        results.append((p, Pf[-1], smax[-1]))

    # Write the average results to output file
    with open(output_file, 'a') as f:
        for p, Pf_low, smax_avg in results:
            f.write(f"{p:.3f}  {Pf_low:.3f}  {smax_avg:.3f}\n")


    # Print results
    print(f"Data written to {output_file} and {dist_file}")

    return p_values, Pf, smax, n_s

def visualize_sample_configurations():
    L = 10
    ps = [0.4, 0.6, 0.8]

    for method in ["burning", "hoshen_kopelman"]:
        fig, axes = plt.subplots(1, len(ps), figsize=(15, 5))
        for idx, p in enumerate(ps):
            lattice = initialize_lattice(L, p)
            ax = axes[idx]
            if method == "burning":
                ax.set_title(f"Burning, p={p}")
                _, labels = burning_method(lattice)
                labels[labels > 0] -= 1

                ax.imshow(labels, cmap='viridis', interpolation='nearest')
                ax.set_aspect('equal', adjustable='box')
                ax.set_xticks([])
                ax.set_yticks([])
                for i in range(labels.shape[0]):
                    for j in range(labels.shape[1]):
                        if labels[i, j] != 0:
                            ax.text(j, i, str(labels[i, j]), ha='center', va='center',
                                 color='black' if labels[i, j] == 0 else 'white')
            else:
                ax.set_title(f"Hoshen-Kopelman, p={p}")
                labels, _ = hoshen_kopelman(lattice, True)
                ax.imshow(labels, cmap='viridis', interpolation='nearest')
                ax.set_aspect('equal', adjustable='box')
                ax.set_xticks([])
                ax.set_yticks([])
        plt.tight_layout(pad=2.0)
        plt.savefig(f"sample_{method}.png")
        plt.show()

def plot_percolation_probability():
    files = ["Ave-L10T1000.txt", "Ave-L50T1000.txt", "Ave-L100T1000.txt"]
    L_values = [10, 50, 100]
    plt.figure(figsize=(8, 6))

    for file, L in zip(files, L_values):
        data = np.loadtxt(file)
        p_values, P_low, _ = data.T
        plt.plot(p_values, P_low, label=f"L={L}", marker='o', linestyle='None')

    plt.xlabel("Occupation Probability (p)")
    plt.ylabel("Percolation Probability $P_{low}$")
    plt.title("Percolation Probability vs Occupation Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("percolation_probability.png")
    plt.show()

def plot_average_max_cluster_size():
    files = ["Ave-L10T1000.txt", "Ave-L50T1000.txt", "Ave-L100T1000.txt"]
    L_values = [10, 50, 100]
    plt.figure(figsize=(8, 6))

    for file, L in zip(files, L_values):
        data = np.loadtxt(file)
        p_values, _, avg_smax = data.T
        plt.plot(p_values, avg_smax, label=f"L={L}", marker='s', linestyle='-.')

    plt.xlabel("Occupation Probability (p)")
    plt.ylabel("Average Maximum Cluster Size $\\langle s_{max} \\rangle$")
    plt.title("Average Maximum Cluster Size vs Occupation Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("average_max_cluster_size.png")
    plt.show()

def plot_cluster_size_distribution():
    L = 100
    T = 100
    # Define the subsets of p-values
    subsets = [(0.2, 0.3, 0.4, 0.5), [0.592746], (0.6, 0.7, 0.8)]
    titles = ["p < pc", "p = pc", "p > pc"]

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (subset, title) in enumerate(zip(subsets, titles)):
        ax = axes[i]  # Get the current subplot axis
        for p in subset:
            _, _, _, n_s = monte_carlo(L, T, p, 0.1, p)  # Run Monte Carlo simulation
            cluster_sizes = n_s[0]  # Get the cluster size distribution

            # Prepare data for plotting
            sizes = sorted(cluster_sizes.keys())
            counts = [cluster_sizes[s] for s in sizes]

            # Plot on the current subplot
            ax.loglog(sizes, counts, marker='o', label=f'p = {p}', linestyle=None)
            ax.set_xlabel('Cluster Size (s)')
            ax.set_ylabel('Frequency n(s)')
            ax.set_title(title)
            ax.grid()

        ax.legend()  # Add legend to the current subplot

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()

def main():
    # Parameters
    p = 0.6  # Occupation probability

    config = load_config("perc-ini.txt")
    L = int(config['L']) # Lattice size
    T = int(config['T']) # Number of trials
    p0 = config['p0'] # Minimum probability
    pk = config['pk'] # Maximum probability
    dp = config['dp'] # Probability step size

    # Initialize the lattice
    lattice = initialize_lattice(L, p)
    print("Initialized Lattice:")
    print(lattice)

    # Check for a spanning cluster
    if burning_method(lattice):
        print("A spanning cluster exists!")
    else:
        print("No spanning cluster found.")

    # Find the maximum cluster size
    smax = max_cluster_size(lattice)
    print(f"The maximum cluster size (smax) is: {smax}")

    # Apply Hoshen-Kopelman algorithm
    labels, label_sizes = hoshen_kopelman(lattice)
    print("\nLattice with cluster labels:")
    print(labels)

    # Compute cluster size distribution
    distribution = cluster_size_distribution(label_sizes)
    print("\nCluster size distribution:")
    for size, count in sorted(distribution.items()):
        print(f"Size {size}: {count} cluster(s)")

    monte_carlo(L, T, p0, dp, pk)

if __name__ == "__main__":
    main()
    visualize_sample_configurations()
    plot_percolation_probability()
    plot_average_max_cluster_size()
    plot_cluster_size_distribution()

