from Bio.PDB import PDBParser   #reads local PDB file
from Bio.SeqUtils import seq1   #convert amino acid to codes
from Bio import ExPASy          #connect to internet database
from Bio import SwissProt       #reads data from internet database
import argparse  # Import to input the PDB file from the command line


# --- Setup arguments ---

# Accept arguments from .sh script
parser = argparse.ArgumentParser(description="Compare PDB and UniProt sequences.")
parser.add_argument('--pdb', required=True, help="Path to the PDB file")
parser.add_argument('--uniprot', required=True, help="The UniProt ID to fetch")
args = parser.parse_args()

pdb_filename = args.pdb
uniprot_id = args.uniprot

print("Reading sequence from local PDB file...")

try:
    # Parse local PDB
    reader = PDBParser(QUIET=True)
    structure = reader.get_structure(uniprot_id, pdb_filename)
    # Will store PDB sequence, (only chain A)
    # [model][chain]
    my_chain = structure[0]['A'] 
    pdb_sequence = ""
    for residue in my_chain:
        if residue.get_resname() != "HOH":
            try:
                pdb_sequence += seq1(residue.get_resname())
            except:
                pdb_sequence += 'X'


    # Fetch UniProt Sequence by connecting to the database
    handle = ExPASy.get_sprot_raw(uniprot_id)
    # This reads the information that came back from database
    record = SwissProt.read(handle)
    # .seq extracts the str letters from the data      
    uniprot_sequence = record.sequence
    # Prints first 50
    print(f"\nFound PDB sequence for Chain A:\n{pdb_sequence[:50]}\n")
    print(f"UniProt sequence:\n{uniprot_sequence[:50]}\n")

    # Compare and Output Details
    # We print the result so the Zsh script can "hear" it
    print("Comparing the two sequences...")
    # '==' checks if left side is identical to right side
    if pdb_sequence == uniprot_sequence:
        print(f"ID {uniprot_id} is perfect match")
        print("RESULT_MATCH")
    else:
        # Checks if different lengths (not counted in tail -n 1)
        # len counts ex characters

        if len(pdb_sequence) != len(uniprot_sequence):
            print(f"Their lengths are different: PDB has {len(pdb_sequence)-1} amino acids, UniProt has {len(uniprot_sequence)}.")

        #   [:-1] ignores the last X in the pdb sequence
        if pdb_sequence[:-1] in uniprot_sequence:
            print("\nThe PDB sequence is a substring of the UniProt sequence")
            start_index = uniprot_sequence.find(pdb_sequence[:-1])

            if start_index != -1:
                end_index = start_index + len(pdb_sequence)
                print(f"between the amino acids: {start_index + 1} and {end_index}\n")
            print("RESULT_MATCH")

        else:
            print("\nNo substring match found.")
            print("RESULT_MISMATCH")

        # The Zsh script needs this as the absolute last line bcz tail
        # print("RESULT_MISMATCH")

except Exception as e:
    print(f"ERROR: {e}")
    # Bash will read the line below via 'tail -n 1'
    print("RESULT_MISMATCH")

