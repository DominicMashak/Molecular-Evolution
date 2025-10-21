"""
Common solvents and their dielectric constants (epsilon).
"""

SOLVENTS = {
    "water": 78.4,
    "methanol": 32.6,
    "ethanol": 24.3,
    "acetonitrile": 35.7,
    "dimethyl sulfoxide": 46.7,
    "acetone": 20.7,
    "chloroform": 4.81,
    "dichloromethane": 8.93,
    "benzene": 2.28,
    "toluene": 2.38,
    "hexane": 1.89,
    "tetrahydrofuran": 7.58,
    "formamide": 109.5,
    "carbon tetrachloride": 2.24,
    "diethyl ether": 4.33,
    "dimethylformamide": 36.7,
    "pyridine": 12.4,
    "dioxane": 2.21,
    "none": 1.0,  # Gas phase
    "isopropanol": 18.3,
    "propylene carbonate": 64.4,
    "ethyl acetate": 6.02,
    "butanol": 17.8,
    "cyclohexane": 2.02,
    "nitromethane": 36.0,
    "trifluoroacetic acid": 8.55,
    "dimethylacetamide": 37.8,
    "sulfolane": 43.3,
    "glycerol": 42.5,
    "1,2-dichloroethane": 10.36,
    "1,4-dioxane": 2.21,
    "carbon disulfide": 2.63,
    "ethyl ether": 4.33,
    "xylene": 2.57,
    "anisole": 4.33,
    "octanol": 10.3,
    "trichloroethylene": 3.42,
    "methyl ethyl ketone": 18.5,
    "naphthalene": 2.3,
    "phenol": 9.99,
    "dimethyl sulfone": 47.6,
}

def get_dielectric(solvent: str = "none") -> float:
    """Return dielectric constant for a given solvent name (case-insensitive). Default is 'none'/1.0."""
    if not solvent:
        solvent = "none"
    return SOLVENTS.get(solvent.lower(), 1.0)

if __name__ == "__main__":
    print("Available solvents and dielectric constants:")
    for name, eps in SOLVENTS.items():
        print(f"{name.title():<20} : {eps}")
