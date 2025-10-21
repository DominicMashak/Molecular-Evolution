import os
import glob

def cleanup_mopac_files(current_mopac_files):
    for file_path in current_mopac_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
    for pattern in ["mol_*.mop", "mol_*.out", "mol_*.arc", "mol_*.log"]:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
            except Exception:
                pass
    current_mopac_files.clear()

def update_progress_file(file_path, iteration, temperature, best_smiles, best_beta, mutated_molecules):
    """
    Updates a progress file with the current iteration details, including temperature,
    best molecule information, and details of mutated molecules.

    This function appends a new entry to the specified file, logging the temperature,
    iteration number, the best molecule's SMILES string and beta value, and information
    about each mutated molecule along with its corresponding mutation type.

    Args:
        file_path (str): The path to the file where progress information will be appended.
        iteration (int): The current iteration number.
        temperature (float): The current temperature value.
        best_smiles (str): The SMILES string of the best molecule.
        best_beta (float): The beta value associated with the best molecule.
        mutated_molecules (list of tuple): A list of tuples, where each tuple contains
            (smiles, beta) for a mutated molecule. The list should have exactly 7 elements,
            corresponding to the predefined mutation types.

    Note:
        The mutation types are hardcoded and assumed to be in the order:
        "Change bond type", "Add atom inline", "Add branch", "Delete atom",
        "Change atom type", "Add ring", "Delete ring".
    """
    with open(file_path, 'a') as f:
        f.write(f"Temperature: {temperature:.6f}, Iteration: {iteration}\n")
        f.write(f"Best molecule: {best_smiles}, Beta: {best_beta:.6f}\n")
        mutation_types = [
            "Change bond type",
            "Add atom inline",
            "Add branch",
            "Delete atom",
            "Change atom type",
            "Add ring",
            "Delete ring"
        ]
        for i, (smiles, beta) in enumerate(mutated_molecules):
            type_name = mutation_types[i]
            f.write(f"Mutated {i+1} ({type_name}): {smiles}, Beta: {beta:.6f}\n")