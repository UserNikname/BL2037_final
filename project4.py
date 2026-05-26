# ============================================================
# FINAL PROJECT PIPELINE
# Protein Sequence + Structure Comparison
# ============================================================

# -----------------------------
# Imports
# -----------------------------
import os
import re
import shutil
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from Bio import AlignIO
from Bio.PDB import (
    PDBList,
    MMCIFParser,
    PDBIO,
    MMCIFIO,
    Select
)

from Bio.PDB.Polypeptide import protein_letters_3to1


# ============================================================
# INSTALL REQUIRED EXTERNAL TOOLS
# ============================================================

def install_tools():

    system = platform.system()

    # ---------- Clustal Omega ----------
    if shutil.which("clustalo") is None:

        print("Installing Clustal Omega...")

        if system == "Darwin":
            subprocess.run(
                ["brew", "install", "clustal-omega"],
                check=True
            )

        elif system == "Linux":
            subprocess.run(
                ["sudo", "apt-get", "install", "-y", "clustalo"],
                check=True
            )

    else:
        print("Clustal Omega already installed.")

    # ---------- TM-align ----------
    if shutil.which("TMalign") is None:

        print("Installing TM-align...")

        if system == "Darwin":
            subprocess.run(
                ["brew", "install", "tmalign"],
                check=True
            )

        elif system == "Linux":
            subprocess.run(
                ["sudo", "apt-get", "install", "-y", "tmalign"],
                check=True
            )

    else:
        print("TM-align already installed.")


install_tools()


# ============================================================
# PATHS
# ============================================================

INPUT_CSV = "/Users/nikolinawennerstrand/Desktop/BL2037/NikoW/final_project/data/inputs_finalproject.csv"

PDB_DIR = Path("/Users/nikolinawennerstrand/Desktop/BL2037/NikoW/final_project/data/pdb_files")
FASTA_DIR = Path("/Users/nikolinawennerstrand/Desktop/BL2037/NikoW/final_project/data/fasta")

RESULTS_DIR = Path("/Users/nikolinawennerstrand/Desktop/BL2037/NikoW/final_project/output4")
SEQUENCE_DIR = RESULTS_DIR / "sequence"
STRUCTURE_DIR = RESULTS_DIR / "structure"
FIGURE_DIR = RESULTS_DIR / "figures"
TREE_DIR = RESULTS_DIR / "trees"

# Create folders
for folder in [
    PDB_DIR,
    FASTA_DIR,
    RESULTS_DIR,
    SEQUENCE_DIR,
    STRUCTURE_DIR,
    FIGURE_DIR,
    TREE_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# READ INPUT CSV
# ============================================================

df = pd.read_csv(INPUT_CSV)

# print(df.head())


# ============================================================
# DOWNLOAD + CLEAN PDB FILES
# ============================================================

pdbl = PDBList()


class ChainSelect(Select):

    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):

        # Remove water molecules
        return residue.get_resname() != "HOH"


def extract_sequence(chain):

    sequence = ""

    for residue in chain:

        resname = residue.get_resname()

        if resname in protein_letters_3to1:
            sequence += protein_letters_3to1[resname]

    return sequence




# ============================================================
# PROCESS STRUCTURES
# ============================================================


# Store sequences
sequences = {}

# Store cleaned pdb files
pdb_files = []


def find_best_chain(requested_chain, available_chains):

    # Exact match
    if requested_chain in available_chains:
        return requested_chain

    # Startswith match
    for c in available_chains:
        if c.startswith(requested_chain):
            return c

    # Contains match
    for c in available_chains:
        if requested_chain in c:
            return c

    # Fallback
    return available_chains[0]


for item in df["PDBID_chain"]:

    pdb_id, chain_id = item.split("_")

    print(f"\nProcessing {pdb_id}_{chain_id}")

    # ------------------------------------------------
    # Download mmCIF
    # ------------------------------------------------
    pdbl.retrieve_pdb_file(
        pdb_id,
        pdir=str(PDB_DIR),
        file_format="mmCif"
    )

    cif_file = next(PDB_DIR.glob(f"{pdb_id.lower()}*.cif"))

    # ------------------------------------------------
    # Parse structure
    # ------------------------------------------------
    parser = MMCIFParser(QUIET=True)

    structure = parser.get_structure(
        pdb_id,
        cif_file
    )

    model = structure[0]

    available_chains = [c.id for c in model]

    # ------------------------------------------------
    # Find best chain
    # ------------------------------------------------
    if chain_id not in available_chains:

        print(f"WARNING: Chain {chain_id} not found in {pdb_id}")
        print(f"Available chains: {available_chains}")

        chain_id = find_best_chain(
            chain_id,
            available_chains
        )

        print(f"Using matching chain: {chain_id}")

    chain = model[chain_id]

    # ------------------------------------------------
    # Extract sequence
    # ------------------------------------------------
    sequence = extract_sequence(chain)

    # Skip empty sequences
    if len(sequence) == 0:

        print(f"WARNING: {pdb_id}_{chain_id} produced empty sequence")
        print("Skipping structure.")

        continue

    # print(sequence)

    actual_name = f"{pdb_id}_{chain_id}"

    sequences[actual_name] = sequence

    # ------------------------------------------------
    # Save cleaned PDB
    # ------------------------------------------------
    io = PDBIO()

    io.set_structure(structure)

    # Safe filename
    safe_chain = chain_id.replace("/", "_")

    output_pdb = PDB_DIR / f"{pdb_id}_{safe_chain}.pdb"

    try:

        io.save(
            str(output_pdb),
            ChainSelect(chain_id)
        )

        pdb_files.append(str(output_pdb))

    except Exception as e:

        print(f"WARNING: Could not save PDB for {actual_name}")
        print(e)

        continue



# ============================================================
# WRITE FASTA FILE
# ============================================================

fasta_file = FASTA_DIR / "all_sequences.fasta"

with open(fasta_file, "w") as f:

    for name, sequence in sequences.items():

        f.write(f">{name}\n")
        f.write(sequence + "\n")

print("\nFASTA file created.")


# ============================================================
# RUN CLUSTAL OMEGA
# ============================================================

aligned_file = SEQUENCE_DIR / "aligned.fasta"

subprocess.run([
    "clustalo",
    "-i", str(fasta_file),
    "-o", str(aligned_file),
    "--force"
])

print("Multiple sequence alignment completed.")


# ============================================================
# CALCULATE SEQUENCE IDENTITY MATRIX
# ============================================================

alignment = AlignIO.read(
    aligned_file,
    "fasta"
)

names = [record.id for record in alignment]

matrix = np.zeros((len(names), len(names)))

for i in range(len(names)):

    for j in range(len(names)):

        seq1 = alignment[i].seq
        seq2 = alignment[j].seq

        matches = sum(
            a == b
            for a, b in zip(seq1, seq2)
        )

        identity = matches / len(seq1)

        matrix[i, j] = identity

identity_df = pd.DataFrame(
    matrix,
    index=names,
    columns=names
)

identity_df.to_csv(
    SEQUENCE_DIR / "sequence_identity_matrix.csv"
)

print("Sequence identity matrix saved.")



# ============================================================
# TM-ALIGN RMSD CALCULATION UTILITY
# ============================================================

def calculate_rmsd(pdb1, pdb2):
    result = subprocess.run(
        ["TMalign", pdb1, pdb2],
        capture_output=True,
        text=True
    )

    output = result.stdout
    rmsd_match = re.search(
        r"RMSD=\s+([0-9.]+)",
        output
    )

    if rmsd_match:
        return float(rmsd_match.group(1))

    return np.nan


# ============================================================
# COMPUTE RMSD MATRIX
# ============================================================

n = len(pdb_files)
rmsd_matrix = np.zeros((n, n))

print("\nCalculating structural RMSD matrix using TM-align...")
for i in range(n):
    for j in range(i, n):
        rmsd = calculate_rmsd(
            pdb_files[i],
            pdb_files[j]
        )

        # Handle failed alignments safely
        if np.isnan(rmsd):
            rmsd = 0.0

        # Fill BOTH sides of the matrix for symmetry
        rmsd_matrix[i, j] = rmsd
        rmsd_matrix[j, i] = rmsd

# Ensure exact diagonal is zero
np.fill_diagonal(rmsd_matrix, 0)


# ============================================================
# CREATE RMSD DATAFRAME & SAVE
# ============================================================

# Force perfect symmetry
rmsd_matrix = (rmsd_matrix + rmsd_matrix.T) / 2

# Names corresponding to saved structures
rmsd_names = [
    Path(p).stem
    for p in pdb_files
]

# Create dataframe
rmsd_df = pd.DataFrame(
    rmsd_matrix,
    index=rmsd_names,
    columns=rmsd_names
)

# Save matrix
rmsd_df.to_csv(
    STRUCTURE_DIR / "rmsd_matrix.csv"
)

print("RMSD matrix saved.")


# ============================================================
# SEQUENCE HEATMAP (FIXED TO USE IDENTITY_DF)
# ============================================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    identity_df,  # Fixed: changed from rmsd_df to your sequence identities
    cmap="viridis",
    vmin=0, 
    vmax=1
)
plt.title("Sequence Similarity (Identity Fraction)")

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "sequence_heatmap.png"
)
plt.close()

print("Sequence heatmap saved.")


# ============================================================
# SEQUENCE CLUSTERING TREE
# ============================================================

distance_matrix = 1 - matrix

linkage_matrix = linkage(
    squareform(distance_matrix),
    method="average"
)

plt.figure(figsize=(10, 6))

dendrogram(
    linkage_matrix,
    labels=names
)

plt.title("Sequence Clustering")
plt.tight_layout()
plt.savefig(
    TREE_DIR / "sequence_tree.png"
)
plt.close()

print("Sequence clustering tree saved.")


# ============================================================
# RMSD HEATMAP
# ============================================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    rmsd_df,
    cmap="magma"
)

plt.title("Structural RMSD (Å)")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "rmsd_heatmap.png"
)
plt.close()

print("RMSD heatmap saved.")


# ============================================================
# STRUCTURAL CLUSTERING TREE
# ============================================================

linkage_matrix_struct = linkage(
    squareform(rmsd_matrix),
    method="average"
)

plt.figure(figsize=(10, 6))

dendrogram(
    linkage_matrix_struct,
    labels=rmsd_names
)

plt.title("Structural Clustering (RMSD)")
plt.tight_layout()
plt.savefig(
    TREE_DIR / "structure_tree.png"
)
plt.close()

print("Structural clustering tree saved.")

# ============================================================
# FINISHED
# ============================================================

print("\n===================================")
print("Pipeline completed successfully.")
print("===================================")
