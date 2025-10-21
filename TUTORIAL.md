
sudo apt update

sudo apt install git

Get an access token from the website https://github.com/settings/tokens

Click the button "Generate new token (classic)"

Under "Select scopes" check "repo"

Name the "Note" Molecular-Evolution

Copy the TOKEN

git clone https://TOKEN@github.com/DominicMashak/Molecular-Evolution

cd Molecular-Evolution

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh
(say 'yes' for all install prompts)

source ~/.bashrc

conda --version
(to check if it installed correct)

echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc

conda env create -f environment.yml

conda activate mol-evo

cd quantum_chemistry

bash run_dft.sh
(first test)

Beta mean result should be 6.101466e+03 or 6,101.47 a.u. 
HOMO-LUMO gap should be 3.767602 eV
Total energy should be -491.949483 a.u.
(Hopefully, the calculation took around 10 or fewer minutes)

cd ..

cd algorithms/nsga2

bash run_nsga2.sh
(second test)

After generation 1, if there is a graph in the nsga2_results folder, then it works. 
