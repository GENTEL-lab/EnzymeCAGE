conda install -c conda-forge -c bioconda mmseqs2=15.6f452 -y
conda install rdkit=2022.09.5 -c conda-forge -y
pip install esm==3.1.1
pip install torch==2.2.0 torch-scatter torch-cluster torch-sparse torch-geometric torchvision -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install pyyaml==6.0.2 tqdm==4.66.2 ase==3.22.1 biopython==1.83 drfp==0.3.6 mlcrate==0.2.0 rxn-chem-utils==1.5.0 transformers==4.24.0

