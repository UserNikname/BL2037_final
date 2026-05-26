

#Prerequisites & Dependencies
# in terminal do:
# pip install biopython pandas numpy matplotlib scipy
#? brew install clustal-omega
#? brew install tm-align


import os
import csv
import subprocess
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from Bio import PDB
from Bio.SeqUtils import seq1

# ==========================================
# 0. CONFIGURATION & DIRECTORY SETUP
# ==========================================
INPUT_CSV = "/Users/nikolinawennerstrand/Desktop/BL2037/NikoW/final_project/data/inputs_finalproject.csv"
OUTPUT_DIR = Path("/Users/nikolinawennerstrand/Desktop/BL2037/NikoW/final_project/outputs")
PDB_DIR = OUTPUT_DIR / "pdb_files"
AF_DIR = OUTPUT_DIR / "alphafold_files"
PLOTS_DIR = OUTPUT_DIR / "plots"

for d in [PDB_DIR, AF_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CLUSTAL_CMD = "clustalo"
TMALIGN_CMD = "TMalign"

# ==========================================
# 1. ROBUST DATA EXTRACTION & CLEANING
# ==========================================
class CleanSelect(PDB.Select):
    """Filter out water molecules and heteroatoms to ensure clean chains."""
    def __init__(self, chain_id):
        self.chain_id = chain_id
    def accept_chain(self, chain):
        return chain.id == self.chain_id
    def accept_residue(self, residue):
        # Remove water (H_HOH) and hetero-atoms/ligands
        return residue.id[0] == " " and (residue.has_id("CA") or residue.has_id("C"))

def download_and_clean_structure(pdb_id: str, chain_id: str) -> Path:
    """
    Downloads structural files from RCSB. 
    Uses mmCIF format natively to fix multi-character chain ID limits.
    """
    clean_path = PDB_DIR / f"{pdb_id}_{chain_id}.cif"
    if clean_path.exists():
        return clean_path

    # Always fetch mmCIF to support multi-character chain IDs natively
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    res = requests.get(url)
    if res.status_code != 200:
        raise RuntimeError(f"Could not download mmCIF for {pdb_id} from RCSB.")

    raw_path = PDB_DIR / f"{pdb_id}_raw.tmp"
    raw_path.write_bytes(res.content)
    
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(raw_path))
    
    # Use MMCIFIO instead of PDBIO to circumvent the single-character chain limit
    io = PDB.MMCIFIO()
    io.set_structure(structure)
    io.save(str(clean_path), CleanSelect(chain_id))
    raw_path.unlink()
    return clean_path

def get_uniprot_id(pdb_id: str, chain_id: str) -> str:
    """Fetches UniProt ID using data-backed fallbacks via the RCSB GraphQL API."""
    query = """
    query($id: String!) {
      entry(entry_id: $id) {
        polymer_entities {
          entity_poly { pdbx_strand_id }
          rcsb_polymer_entity_container_identifiers { uniprot_ids }
        }
      }
    }
    """
    req = requests.post("https://data.rcsb.org/graphql", json={"query": query, "variables": {"id": pdb_id.upper()}})
    if req.status_code != 200:
        raise ValueError("RCSB GraphQL API offline.")
        
    data = req.json()
    entities = data.get('data', {}).get('entry', {}).get('polymer_entities', []) or []
    
    # Search entity mapping specific to our target chain
    for entity in entities:
        strands = [s.strip() for s in entity.get('entity_poly', {}).get('pdbx_strand_id', '').split(',')]
        if chain_id in strands:
            uniprot_ids = entity.get('rcsb_polymer_entity_container_identifiers', {}).get('uniprot_ids', [])
            if uniprot_ids:
                return uniprot_ids[0]
                
    raise ValueError(f"No UniProt ID mapping found for Chain {chain_id}")

def download_alphafold(uniprot_id: str, label: str) -> Path:
    """Retrieves structural prediction files from AlphaFold DB."""
    af_path = AF_DIR / f"{label}_AF.pdb"
    if af_path.exists():
        return af_path

    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    res = requests.get(api_url)
    res.raise_for_status()
    data = res.json()
    
    if not data:
        raise RuntimeError(f"No AlphaFold entry found for UniProt ID: {uniprot_id}")
        
    pdb_url = data[0]['pdbUrl']
    pdb_res = requests.get(pdb_url)
    af_path.write_bytes(pdb_res.content)
    return af_path

def extract_sequence(file_path: Path) -> str:
    """Extracts standard single-letter sequences from parsed structural files."""
    if file_path.suffix == ".cif":
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        parser = PDB.PDBParser(QUIET=True)
        
    structure = parser.get_structure("temp", str(file_path))
    residues = [res for res in structure.get_residues() if PDB.is_aa(res, standard=True) and "CA" in res]
    return "".join([seq1(res.get_resname()) for res in residues])

# ==========================================
# 2. RUNNING EXTERNAL TOOL WRAPPERS
# ==========================================
def run_clustal_omega(fasta_in: Path, alignment_out: Path):
    """Executes multiple sequence alignments safely over subprocess."""
    cmd = [CLUSTAL_CMD, "-i", str(fasta_in), "-o", str(alignment_out), "--outfmt=fa", "--force"]
    subprocess.run(cmd, check=True)

def parse_fasta_alignment(alignment_path: Path) -> dict:
    """Parses aligned FASTA file fields back into dictionary pairs."""
    sequences = {}
    current_id = None
    with open(alignment_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                current_id = line.strip().split(">")[1]
                sequences[current_id] = []
            else:
                sequences[current_id].append(line.strip())
    return {k: "".join(v) for k, v in sequences.items()}

def calculate_sequence_identity_matrix(aligned_seqs: dict) -> pd.DataFrame:
    """Generates a quantitative sequence identity matrix comparison."""
    ids = list(aligned_seqs.keys())
    n = len(ids)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            seq1, seq2 = aligned_seqs[ids[i]], aligned_seqs[ids[j]]
            total = sum(1 for a, b in zip(seq1, seq2) if a != '-' or b != '-')
            matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-')
            matrix[i, j] = (matches / total) * 100 if total > 0 else 0.0
            
    return pd.DataFrame(matrix, index=ids, columns=ids)

def run_tmalign_rmsd(file1: Path, file2: Path) -> float:
    """Extracts structural RMSD values between two structural files via TM-align."""
    try:
        cmd = [TMALIGN_CMD, str(file1), str(file2)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.split("\n"):
            if line.startswith("Aligned length="):
                parts = line.split(",")
                rmsd_val = float(parts[1].split("=")[1].strip())
                return rmsd_val
    except Exception as e:
        pass
    return 25.0 

def generate_structural_rmsd_matrix(file_paths: dict) -> pd.DataFrame:
    """Loops pairwise permutations to track multi-dimensional spatial similarity."""
    ids = list(file_paths.keys())
    n = len(ids)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i, j] = 0.0
            else:
                rmsd = run_tmalign_rmsd(file_paths[ids[i]], file_paths[ids[j]])
                matrix[i, j] = rmsd
                matrix[j, i] = rmsd
    return pd.DataFrame(matrix, index=ids, columns=ids)

# ==========================================
# 3. PLOTTING AND TREES
# ==========================================
def plot_matrix_and_dendrogram(matrix_df: pd.DataFrame, title: str, filename_prefix: str, is_distance=True):
    """Constructs clustered Heatmaps alongside structural Distance trees."""
    if matrix_df.empty:
        return
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix_df.values, cmap="viridis" if not is_distance else "viridis_r")
    plt.colorbar(label="Sequence Identity (%)" if not is_distance else "RMSD (Å)")
    plt.xticks(range(len(matrix_df.columns)), matrix_df.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(matrix_df.index)), matrix_df.index, fontsize=6)
    plt.title(f"{title} Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{filename_prefix}_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    dist_matrix = matrix_df.values if is_distance else (100.0 - matrix_df.values)
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    
    condensed_dist = squareform(dist_matrix, checks=False)
    Z = linkage(condensed_dist, method="average")
    
    dendrogram(Z, labels=list(matrix_df.index), leaf_rotation=90, leaf_font_size=6)
    plt.title(f"{title} Hierarchical Clustering Tree")
    plt.ylabel("Distance Metric Linkage")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{filename_prefix}_tree.png", dpi=200)
    plt.close()

# ==========================================
# MAIN EXECUTION ENGINE
# ==========================================
def main():
    print("Initializing Structural Comparative Pipeline...")
    
    protein_targets = []
    with open(INPUT_CSV, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            protein_targets.append(row['PDBID_chain'])
    
    protein_targets = protein_targets[:25]
    print(f"Loaded {len(protein_targets)} unique structural targets.")

    valid_pdb_paths = {}
    valid_af_paths = {}
    valid_pdb_seqs = {}
    valid_af_seqs = {}

    # Steps 1 & 6: Unified Extraction & Validation
    for target in protein_targets:
        try:
            pdb_id, chain_id = target.split("_")
            print(f"Processing target identifier: {pdb_id} (Chain {chain_id})")
            
            # 1. Resolve UniProt Mapping first to ensure it's a valid protein target
            uniprot_id = get_uniprot_id(pdb_id, chain_id)
            
            # 2. Extract Experimental Coordinates (using modern .cif processing)
            clean_pdb = download_and_clean_structure(pdb_id, chain_id)
            pdb_seq = extract_sequence(clean_pdb)
            
            # 3. Extract AlphaFold Coordinates
            clean_af = download_alphafold(uniprot_id, target)
            af_seq = extract_sequence(clean_af)
            
            # Crucial Guard: Verify neither sequence is empty before retaining target
            if len(pdb_seq) == 0 or len(af_seq) == 0:
                print(f"⚠️ Target {target} has 0 parsed CA amino acids. Skipping.")
                continue
                
            # If all checks pass, commit to our pipeline registers
            valid_pdb_paths[target] = clean_pdb
            valid_pdb_seqs[target] = pdb_seq
            valid_af_paths[target] = clean_af
            valid_af_seqs[target] = af_seq
            
        except Exception as e:
            print(f"⚠️ Target entry fail on validation step: {target}. Error: {e}")

    if not valid_pdb_paths:
        print("❌ No valid protein configurations survived sanitation. Terminating pipeline.")
        return

    # Steps 2 & 3: Run Sequence Cluster Pipeline (PDB)
    pdb_fasta = OUTPUT_DIR / "pdb_sequences.fasta"
    with open(pdb_fasta, "w") as f:
        for k, v in valid_pdb_seqs.items():
            f.write(f">{k}\n{v}\n")
            
    pdb_aln = OUTPUT_DIR / "pdb_aligned.fasta"
    print("\nRunning sequence alignment using Clustal Omega for PDB sequences...")
    run_clustal_omega(pdb_fasta, pdb_aln)
    aligned_pdb_seqs = parse_fasta_alignment(pdb_aln)
    pdb_seq_matrix = calculate_sequence_identity_matrix(aligned_pdb_seqs)
    plot_matrix_and_dendrogram(pdb_seq_matrix, "PDB Sequence Identity", "pdb_sequence", is_distance=False)

    # Steps 4 & 5: Run Structural Distance Pipeline (PDB)
    print("Calculating spatial alignment superimpositions using TM-align for PDB structures...")
    pdb_rmsd_matrix = generate_structural_rmsd_matrix(valid_pdb_paths)
    plot_matrix_and_dendrogram(pdb_rmsd_matrix, "PDB Structural RMSD", "pdb_structure", is_distance=True)

    # Step 6: Process AlphaFold Variants Parallel Sequence Matrix
    af_fasta = OUTPUT_DIR / "af_sequences.fasta"
    with open(af_fasta, "w") as f:
        for k, v in valid_af_seqs.items():
            f.write(f">{k}\n{v}\n")
            
    af_aln = OUTPUT_DIR / "af_aligned.fasta"
    print("\nRunning sequence alignment using Clustal Omega for AlphaFold sequences...")
    run_clustal_omega(af_fasta, af_aln)
    aligned_af_seqs = parse_fasta_alignment(af_aln)
    af_seq_matrix = calculate_sequence_identity_matrix(aligned_af_seqs)
    plot_matrix_and_dendrogram(af_seq_matrix, "AlphaFold Sequence Identity", "af_sequence", is_distance=False)

    print("Calculating spatial alignment superimpositions using TM-align for AlphaFold structures...")
    af_rmsd_matrix = generate_structural_rmsd_matrix(valid_af_paths)
    plot_matrix_and_dendrogram(af_rmsd_matrix, "AlphaFold Structural RMSD", "af_structure", is_distance=True)

    # Matrix Cross-Evaluation (PDB vs AlphaFold Matrix Pearson Correlation)
    common_targets = list(set(pdb_rmsd_matrix.index).intersection(af_rmsd_matrix.index))
    if len(common_targets) > 1:
        pdb_sub = pdb_rmsd_matrix.loc[common_targets, common_targets].values.flatten()
        af_sub = af_rmsd_matrix.loc[common_targets, common_targets].values.flatten()
        correlation = np.corrcoef(pdb_sub, af_sub)[0, 1]
        print("\n" + "="*65)
        print(f"METRIC REPORT SUMMARY: Structural matrices display a correlation of: {correlation:.4f}")
        print("="*65)

if __name__ == "__main__":
    main()
