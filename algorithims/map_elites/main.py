
import mapelites

def main():
    optimizer = mapelites.MAPElitesArchive(
        measure_dims = [10,10], 
        measure_keys = ["num_atoms", "num_bonds"], 
        objective_key = "rand"
    )
    
    
if __name__ == "__main__":
    main()