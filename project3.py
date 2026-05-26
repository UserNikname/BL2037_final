

#!/usr/bin/env python3
"""
Structural Biology Computational Pipeline: Homology and Structural Comparison
Author: Project E-C/D/C Framework Adaptation
"""

import os
import sys
import shutil
import subprocess
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# BioPython Modules
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser, Select
from Bio.SeqUtils import seq1

# Clustering Modules
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# ==========================================
# PHASE 0: CONFIGURATION & DIRECTORY SETUP
# ==========================================

INPUT_CSV = Path("/Users/nikolinawennerstrand/Desktop/BL2037/NikoW/final_project/data/inputs_finalproject.csv")
OUTPUT_DIR = Path("/Users/nikolinawennerstrand/Desktop/BL2037/NikoW/final_project/output3")

# Subdirectories for organized data tracking
PDB_RAW_DIR = OUTPUT_DIR / "pdb_raw"
PDB_CLEAN_DIR = OUTPUT_DIR / "pdb_clean"
AF_RAW_DIR = OUTPUT_DIR / "alphafold_raw"
AF_CLEAN_DIR = OUTPUT_DIR / "alphafold_clean"
ALIGN_DIR = OUTPUT_DIR / "alignments"
MATRIX_DIR = OUTPUT_DIR / "matrices"
FIGURES_DIR = OUTPUT_DIR / "figures"

for d in [PDB_RAW_DIR, PDB_CLEAN_DIR, AF_RAW_DIR, AF_CLEAN_DIR, ALIGN_DIR, MATRIX_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Helper Class to Filter out non-hetero atoms and water during cleaning
class CleanProteinSelect(Select):
    def accept_residue(self, residue):
        # Reject water molecules
        if residue.get_resname() in ["HOH", "WAT", "H2O"]:
            return 0
        # Accept only standard amino acids (ignore hetero-atoms like ligands/ions)
        hetfield = residue.get_id()[0]
        if hetfield.strip() != "":
            return 0
        return 1

# ==========================================
# PHASE 1: DATA RETRIEVAL & CLEANING (STEP 1)
# ==========================================

def parse_input_csv(csv_path: Path) -> List[Tuple[str, str]]:
    """Reads the CSV file and returns a list of tuples containing (PDB_ID, Chain)."""
    if not csv_path.exists():
        print(f"[ERROR] Input CSV not found at {csv_path}. Using a fallback list for demonstration.")
        # Fallback list for pipeline testing
        fallback_ids = ["4WM6_A", "1ZAK_A", "1SYR_A", "5XV5_A", "1HG3_A"]
        return [tuple(x.split("_")) for x in fallback_ids]
    
    df = pd.read_csv(csv_path)
    # Assumes column name matches 'PDBID_chain' or is the first column
    col_name = df.columns[0]
    pairs = []
    for val in df[col_name].dropna():
        if "_" in str(val):
            pdb, chain = str(val).strip().split("_")
            pairs.append((pdb.upper(), chain))
    return pairs

def download_pdb(pdb_id: str) -> Path:
    """Downloads an experimental structure in .cif or .pdb format from RCSB."""
    pdb_id = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    dest_path = PDB_RAW_DIR / f"{pdb_id}.cif"
    if not dest_path.exists() or dest_path.stat().st_size == 0:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            dest_path.write_bytes(r.content)
        else:
            # Fallback to standard .pdb format if .cif fails
            url_pdb = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            dest_path = PDB_RAW_DIR / f"{pdb_id}.pdb"
            r_pdb = requests.get(url_pdb, timeout=30)
            if r_pdb.status_code == 200:
                dest_path.write_bytes(r_pdb.content)
            else:
                raise RuntimeError(f"Could not download PDB {pdb_id} from RCSB.")
    return dest_path

def clean_and_extract_chain(file_path: Path, pdb_id: str, chain_id: str, output_dir: Path) -> Tuple[Path, str]:
    """Cleans structure file (removes water/ligands), isolates the chain, and returns sequence."""
    suffix = file_path.suffix.lower()
    if suffix == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
        
    structure = parser.get_structure(pdb_id, str(file_path))
    model = next(structure.get_models())
    
    if chain_id not in model:
        # Fallback if preferred chain doesn't exist
        available_chains = [c.id for c in model.get_chains()]
        print(f"[WARN] Chain {chain_id} not found in {pdb_id}. Using first available chain: {available_chains[0]}")
        chain_id = available_chains[0]
        
    chain = model[chain_id]
    
    # Save clean structure
    io = PDB.PDBIO()
    io.set_structure(chain)
    clean_file_path = output_dir / f"{pdb_id}_{chain_id}_clean.pdb"
    io.save(str(clean_file_path), CleanProteinSelect())
    
    # Extract sequence
    clean_structure = PDBParser(QUIET=True).get_structure(f"{pdb_id}_clean", str(clean_file_path))
    clean_model = next(clean_structure.get_models())
    clean_chain = clean_model.get_list()[0] # Contains our single chain
    
    residues = [r for r in clean_chain.get_residues() if PDB.is_aa(r, standard=True)]
    sequence = "".join([seq1(r.get_resname()) for r in residues])
    
    return clean_file_path, sequence

# ==========================================
# PHASE 2: SEQUENCE ALIGNMENT & CLUSTERING (STEPS 2 & 3)
# ==========================================

def run_clustal_omega(fasta_in: Path, alignment_out: Path):
    """Executes external Clustal Omega tool via Python subprocess."""
    if not shutil.which("clustalo"):
        raise EnvironmentError("External command 'clustalo' not found in system PATH.")
        
    cmd = [
        "clustalo",
        "-i", str(fasta_in),
        "-o", str(alignment_out),
        "--outfmt=fa",
        "--force"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def parse_fasta_alignment(alignment_path: Path) -> Dict[str, str]:
    """Parses a multi-FASTA alignment file into a dictionary mapping headers to sequences."""
    alignment = {}
    current_header = None
    current_seq = []
    with open(alignment_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    alignment[current_header] = "".join(current_seq)
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header:
            alignment[current_header] = "".join(current_seq)
    return alignment

def calculate_identity_matrix(alignment: Dict[str, str]) -> pd.DataFrame:
    """Calculates a identity matrix from aligned, equal-length sequences."""
    names = list(alignment.keys())
    n = len(names)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            seq1_arr = alignment[names[i]]
            seq2_arr = alignment[names[j]]
            length = len(seq1_arr)
            
            matches = 0
            valid_positions = 0
            for k in range(length):
                if seq1_arr[k] != '-' or seq2_arr[k] != '-':
                    valid_positions += 1
                    if seq1_arr[k] == seq2_arr[k]:
                        matches += 1
                        
            matrix[i, j] = (matches / valid_positions) * 100 if valid_positions > 0 else 0.0
            
    return pd.DataFrame(matrix, index=names, columns=names)

# ==========================================
# PHASE 3: STRUCTURAL ALIGNMENT & RMSD (STEPS 4 & 5)
# ==========================================

def run_tm_align(probe_pdb: Path, target_pdb: Path) -> Tuple[float, float]:
    """Executes external TM-align program to retrieve sequence-independent RMSD and TM-score."""
    if not shutil.which("TMalign"):
        raise EnvironmentError("External command 'TMalign' not found in system PATH.")
        
    cmd = ["TMalign", str(probe_pdb), str(target_pdb)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    rmsd = 0.0
    tm_score = 0.0
    for line in result.stdout.split("\n"):
        if line.startswith("Aligned length="):
            # Example line: Aligned length= 120, RMSD=   1.45, Seq_ID=n.nnn
            parts = line.split(",")
            for part in parts:
                if "RMSD=" in part:
                    rmsd = float(part.split("=")[1].strip())
        elif line.startswith("TM-score="):
            # Take the first TM-score normalized by the average or specific chain length
            if tm_score == 0.0:
                tm_score = float(line.split("=")[1].split("(")[0].strip())
                
    return rmsd, tm_score

def compute_structural_matrices(file_paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates pairwise matrices for global structural deviations (RMSD & TM-score)."""
    names = list(file_paths.keys())
    n = len(names)
    rmsd_mat = np.zeros((n, n))
    tm_mat = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                rmsd_mat[i, j] = 0.0
                tm_mat[i, j] = 1.0
            else:
                try:
                    rmsd, tm = run_tm_align(file_paths[names[i]], file_paths[names[j]])
                    rmsd_mat[i, j] = rmsd
                    tm_mat[i, j] = tm
                except Exception:
                    rmsd_mat[i, j] = 99.9  # Masking failed structural comparisons
                    tm_mat[i, j] = 0.0
                    
    return pd.DataFrame(rmsd_mat, index=names, columns=names), pd.DataFrame(tm_mat, index=names, columns=names)

# ==========================================
# PHASE 4: CLUSTERING & PLOTTING UTILITIES
# ==========================================

def generate_heatmap(df: pd.DataFrame, title: str, filename: Path, cmap: str = "viridis"):
    """Saves a matrix heatmap to disk."""
    plt.figure(figsize=(10, 8))
    plt.imshow(df.values, cmap=cmap)
    plt.colorbar(label=title)
    plt.xticks(range(len(df.columns)), df.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(df.index)), df.index, fontsize=8)
    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def generate_dendrogram(df: pd.DataFrame, metric_type: str, title: str, filename: Path):
    """Calculates agglomerative hierarchical trees from distance space transformations."""
    plt.figure(figsize=(10, 6))
    
    if metric_type == "identity":
        # Convert percent similarity to a distance space
        distance_matrix = 100.0 - df.values
        np.fill_diagonal(distance_matrix, 0)
        dist_vector = squareform(distance_matrix, checks=False)
    elif metric_type == "tmscore":
        # Convert TM-scores (0 to 1 structural likeness) to dynamic distance range
        distance_matrix = 1.0 - df.values
        np.fill_diagonal(distance_matrix, 0)
        dist_vector = squareform(distance_matrix, checks=False)
    else: # RMSD metric
        dist_vector = squareform(df.values, checks=False)
        
    Z = linkage(dist_vector, method="average")
    dendrogram(Z, labels=list(df.index), leaf_rotation=90, leaf_font_size=8)
    plt.title(title, fontsize=12, pad=10)
    plt.ylabel("Linkage Distance Metric Space")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

# ==========================================
# PHASE 5: ALPHAFOLD EXPANSION (GRADE C STEP 6)
# ==========================================

def get_uniprot_id_from_pdb(pdb_id: str) -> Optional[str]:
    """Queries RCSB Graph API for SIFTS-annotated UniProt accession mapping links."""
    url = "https://data.rcsb.org/graphql"
    query = """
    query($id: String!) {
      entry(entry_id: $id) {
        polymer_entities {
          rcsb_polymer_entity_container_identifiers { uniprot_ids }
        }
      }
    }
    """
    try:
        r = requests.post(url, json={"query": query, "variables": {"id": pdb_id.upper()}}, timeout=20)
        if r.status_code == 200:
            data = r.json()
            entities = data.get("data", {}).get("entry", {}).get("polymer_entities", [])
            for entity in entities:
                uniprot_ids = entity.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids", [])
                if uniprot_ids:
                    return uniprot_ids[0]
    except Exception:
        pass
    return None

def download_alphafold_model(uniprot_id: str) -> Path:
    """Retrieves the highest validation version (v4/v6) target structure from AlphaFold DB."""
    uniprot_id = uniprot_id.upper()
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    
    r = requests.get(api_url, timeout=20)
    r.raise_for_status()
    data = r.json()
    
    if isinstance(data, list) and len(data) > 0:
        pdb_url = data[0].get("pdbUrl")
        if pdb_url:
            dest_path = AF_RAW_DIR / Path(pdb_url).name
            if not dest_path.exists():
                r_pdb = requests.get(pdb_url, timeout=30)
                dest_path.write_bytes(r_pdb.content)
            return dest_path
            
    raise RuntimeError(f"AlphaFold prediction structure mapping payload missing for {uniprot_id}")

# ==========================================
# MASTER PIPELINE EXECUTION CONTROL
# ==========================================

def run_pipeline():
    print("==========================================================================")
    print("STARTING BIOLOGICAL STRUCTURE PIPELINE PROCESSING")
    print("==========================================================================")
    
    # --- STEP 1: PARSING AND DOWNLOADING ---
    id_pairs = parse_input_csv(INPUT_CSV)
    print(f"[INFO] Processing {len(id_pairs)} targeted protein structural chains from registry.")
    
    experimental_sequences = {}
    experimental_clean_paths = {}
    pdb_to_uniprot_map = {}
    
    for pdb_id, chain_id in id_pairs:
        label = f"{pdb_id}_{chain_id}"
        try:
            print(f" -> Downloading and cleaning structural entity: {label}")
            raw_path = download_pdb(pdb_id)
            clean_path, seq = clean_and_extract_chain(raw_path, pdb_id, chain_id, PDB_CLEAN_DIR)
            experimental_sequences[label] = seq
            experimental_clean_paths[label] = clean_path
            
            # Extract UniProt reference for Phase 6 expansion tasks
            uniprot_id = get_uniprot_id_from_pdb(pdb_id)
            if uniprot_id:
                pdb_to_uniprot_map[label] = uniprot_id
        except Exception as e:
            print(f" [WARN] Failed processing for structural variant token: {label}. Error: {e}")

    # --- STEP 2 & 3: EXPERIMENTAL SEQUENCE IDENTITY ALIGNMENT MATRIX & TREE ---
    print("\n[INFO] Starting Step 2 & 3: Generation of Experimental Sequence Alignment Space Worksheets...")
    exp_fasta_in = ALIGN_DIR / "experimental_inputs.fasta"
    with open(exp_fasta_in, "w") as f:
        for lbl, seq in experimental_sequences.items():
            f.write(f">{lbl}\n{seq}\n")
            
    exp_alignment_out = ALIGN_DIR / "experimental_aligned.fasta"
    try:
        run_clustal_omega(exp_fasta_in, exp_alignment_out)
        exp_alignment = parse_fasta_alignment(exp_alignment_out)
        exp_identity_df = calculate_identity_matrix(exp_alignment)
        
        exp_identity_df.to_csv(MATRIX_DIR / "experimental_seq_identity.csv")
        generate_heatmap(exp_identity_df, "Sequence Identity (%)", FIGURES_DIR / "step2_experimental_seq_identity_heatmap.png")
        generate_dendrogram(exp_identity_df, "identity", "Hierarchical Sequence Tree (Experimental)", FIGURES_DIR / "step3_experimental_seq_tree.png")
        print(" -> Finished Sequence Alignment & Clustering Steps successfully.")
    except Exception as e:
        print(f" [CRITICAL] Core Sequence Processing Engine halted operations. Reason: {e}")

    # --- STEP 4 & 5 (GRADE E-D): EXPERIMENTAL STRUCTURAL RMSD ALIGNMENTS & TREE ---
    print("\n[INFO] Starting Step 4 & 5: Executing Global Topological Space Comparisons (TM-align Engine)...")
    if len(experimental_clean_paths) > 1:
        exp_rmsd_df, exp_tm_df = compute_structural_matrices(experimental_clean_paths)
        
        exp_rmsd_df.to_csv(MATRIX_DIR / "experimental_rmsd.csv")
        exp_tm_df.to_csv(MATRIX_DIR / "experimental_tm_score.csv")
        
        generate_heatmap(exp_rmsd_df, "Pairwise RMSD (Å)", FIGURES_DIR / "step4_experimental_rmsd_heatmap.png", cmap="magma")
        generate_dendrogram(exp_rmsd_df, "rmsd", "Hierarchical Structural Clustering Tree (PDB RMSD)", FIGURES_DIR / "step5_experimental_structural_tree.png")
        print(" -> Finished Structural Matrix Operations successfully.")
    else:
        print(" [WARN] Insufficient clean structures remaining to carry out matrix operations.")

    # --- STEP 6 (GRADE C): ALPHAFOLD COMPARISON LOOP ---
    print("\n[INFO] Starting Step 6 (Grade C Expansion Task): Incorporating AlphaFold Databases...")
    af_clean_paths = {}
    af_sequences = {}
    
    for exp_label, uniprot_id in pdb_to_uniprot_map.items():
        try:
            print(f" -> Downloading and cleaning AlphaFold model for {exp_label} via UniProt: {uniprot_id}")
            af_raw = download_alphafold_model(uniprot_id)
            af_clean, af_seq = clean_and_extract_chain(af_raw, f"AF_{uniprot_id}", "A", AF_CLEAN_DIR)
            af_clean_paths[f"AF_{exp_label}"] = af_clean
            af_sequences[f"AF_{exp_label}"] = af_seq
        except Exception as e:
            print(f" [WARN] Skipping prediction analysis for {exp_label}. Reason: {e}")

    if af_sequences:
        print(" -> Running Clustal Omega multiple sequence alignment across alternative AlphaFold variant groups...")
        af_fasta_in = ALIGN_DIR / "alphafold_inputs.fasta"
        with open(af_fasta_in, "w") as f:
            for lbl, seq in af_sequences.items():
                f.write(f">{lbl}\n{seq}\n")
        af_alignment_out = ALIGN_DIR / "alphafold_aligned.fasta"
        
        try:
            run_clustal_omega(af_fasta_in, af_alignment_out)
            af_alignment = parse_fasta_alignment(af_alignment_out)
            af_identity_df = calculate_identity_matrix(af_alignment)
            
            af_identity_df.to_csv(MATRIX_DIR / "alphafold_seq_identity.csv")
            generate_heatmap(af_identity_df, "AlphaFold Seq Identity (%)", FIGURES_DIR / "step6_alphafold_seq_identity_heatmap.png")
            generate_dendrogram(af_identity_df, "identity", "Hierarchical Sequence Tree (AlphaFold)", FIGURES_DIR / "step6_alphafold_seq_tree.png")
        except Exception as e:
            print(f" [WARN] AlphaFold sequence matrix step failed: {e}")
            
        if len(af_clean_paths) > 1:
            print(" -> Running TM-align matrix processing across candidate AlphaFold structures...")
            af_rmsd_df, af_tm_df = compute_structural_matrices(af_clean_paths)
            af_rmsd_df.to_csv(MATRIX_DIR / "alphafold_rmsd.csv")
            generate_heatmap(af_rmsd_df, "AlphaFold Pairwise RMSD (Å)", FIGURES_DIR / "step6_alphafold_rmsd_heatmap.png", cmap="magma")
            generate_dendrogram(af_rmsd_df, "rmsd", "Hierarchical Structural Clustering Tree (AlphaFold RMSD)", FIGURES_DIR / "step6_alphafold_structural_tree.png")

        # --- RE-SUPERIMPOSITION DIRECT ANALYSIS: EXPERIMENTAL REALITY VS MODEL PREDICTION ---
        print("\n -> Performing exact direct pairwise structural comparisons (PDB Experimental vs. Matching AlphaFold Model)...")
        direct_comparison_rows = []
        for exp_lbl, clean_pdb_path in experimental_clean_paths.items():
            af_lbl = f"AF_{exp_lbl}"
            if af_lbl in af_clean_paths:
                try:
                    rmsd, tm = run_tm_align(clean_pdb_path, af_clean_paths[af_lbl])
                    direct_comparison_rows.append({
                        "Structure_Label": exp_lbl,
                        "UniProt_ID": pdb_to_uniprot_map[exp_lbl],
                        "Direct_Pairwise_RMSD": rmsd,
                        "Direct_TM_Score": tm
                    })
                except Exception:
                    pass
        
        if direct_comparison_rows:
            direct_df = pd.DataFrame(direct_comparison_rows)
            direct_df.to_csv(OUTPUT_DIR / "direct_experimental_vs_alphafold_report.csv", index=False)
            print("\n==========================================================================")
            print("DIRECT STRUCTURAL DISCREPANCY SUMMARY (Reality vs. Prediction)")
            print("==========================================================================")
            print(direct_df.to_string(index=False))
            
    print("\n==========================================================================")
    print("PIPELINE PROCESSING CYCLES COMPLETE. Check paths under './final_project/output/'")
    print("==========================================================================")

if __name__ == "__main__":
    run_pipeline()
