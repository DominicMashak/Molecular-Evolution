import os
import matplotlib.pyplot as plt
import re

def plot_progress(progress_file, output_dir):
    """
    Plots the progress of simulated annealing from a given progress file.
    This function reads a progress file containing lines with iteration numbers,
    temperatures, and best beta values. It extracts this data and generates two
    plots: one for the best beta values over iterations and another for temperature
    over iterations. The plots are saved as PNG files in the specified output directory.
    Parameters:
        progress_file (str): Path to the file containing progress data from simulated annealing.
        output_dir (str): Directory where the plot images will be saved.
    Returns:
        None: The function saves plots to files and does not return any value.
    Notes:
        - The function assumes the progress file has lines in formats like:
          "Temperature: <value> Iteration: <value>" and "Best molecule: ... Beta: <value>".
        - If the progress file does not exist or lacks required data, the function returns early without plotting.
        - Requires matplotlib.pyplot (imported as plt), os, and re modules.
    """
    if not os.path.exists(progress_file):
        return
    
    iterations = []
    best_betas = []
    temperatures = []
    
    with open(progress_file, 'r') as f:
        for line in f:
            line = line.strip()
            if "Temperature:" in line and "Iteration:" in line:
                # Extract temperature and iteration
                match_temp = re.search(r'Temperature: ([\d\.\-]+)', line)
                match_iter = re.search(r'Iteration: (\d+)', line)
                if match_temp and match_iter:
                    temperatures.append(float(match_temp.group(1)))
                    iterations.append(int(match_iter.group(1)))
            elif "Best molecule:" in line and "Beta:" in line:
                # Extract best beta
                match = re.search(r'Beta: ([\d\.\-]+)', line)
                if match:
                    best_betas.append(float(match.group(1)))
    
    if not iterations or not best_betas or not temperatures:
        return
    
    # Plot for Best Beta
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, best_betas)
    plt.title('Best Beta over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Best Beta')
    plt.savefig(os.path.join(output_dir, 'sa_progress_beta.png'))
    plt.close()
    
    # Plot for Temperature
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, temperatures, color='red')
    plt.title('Temperature over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')
    # Remove grid
    plt.savefig(os.path.join(output_dir, 'sa_progress_temp.png'))
    plt.close()