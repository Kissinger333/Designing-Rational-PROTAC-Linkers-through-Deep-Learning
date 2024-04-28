import pandas as pd
import numpy as np
import requests
import os
from Bio.PDB import PDBParser

def check_pdb_existence(uniprot_id,path):
    pdb_filename = f"{uniprot_id}.pdb"
    save_path = os.path.join(path, pdb_filename)
    return os.path.exists(save_path)
def download_alphafold_pdb(uniprot_id,path):
    pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    response = requests.get(pdb_url)
    
    if response.status_code == 200:


        pdb_filename = f"{uniprot_id}.pdb"
        save_path = os.path.join(path, pdb_filename)
        with open(save_path, 'wb') as f:
            f.write(response.content)

    else:
        print(f"Cannot download {uniprot_id}")



def calculate_distance(atom1, atom2):
    """计算两个原子之间的欧氏距离"""
    return np.linalg.norm(atom1.coord - atom2.coord)

def contact_map(pdb_file, d0=3.8):
    """生成接触图"""
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)

    # 获取Cα原子
    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_atoms.append(residue["CA"])

    # 计算Cα原子之间的距离
    l = len(ca_atoms)
    contact_matrix = np.zeros((l, l))
    for i in range(l):
        for j in range(i,l):
            distance = calculate_distance(ca_atoms[i], ca_atoms[j])
            similarity =min(1.0, 2 / (1 + distance / d0))
            contact_matrix[i, j] = similarity
            contact_matrix[j, i] = similarity

    return contact_matrix


def download_pdb():
    df=pd.read_excel("data\pro_fea.xlsx")
    Uniprot=df['Uniprot'].to_list()
    path='data\pdb'
    for uniprot in Uniprot:
        if not check_pdb_existence(uniprot,path):
            download_alphafold_pdb(uniprot,path)
    for uniprot in Uniprot:
        pdb_file = f"{uniprot}.pdb"  
        contact_matrix = contact_map(pdb_file)
        np.save(f'data\contactmap\{uniprot}.npy', contact_matrix)

download_pdb()