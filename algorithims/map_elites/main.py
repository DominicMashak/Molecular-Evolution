
import archive

def main():
    arch = archive.MAPElitesArchive(
        measure_dims = [10,10], 
        measure_keys = ["num_atoms", "num_bonds"], 
        objective_key = "rand"
    )
    
    #optimizer 
    
if __name__ == "__main__":
    main()