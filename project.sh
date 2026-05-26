#!/usr/bin/env zsh


# define arguments here
BASE_DIR="/Users/nikolinawennerstrand/Desktop/BL2037/NikoW"
ID_LIST="$BASE_DIR/data/lab4/list.txt"
PDB_DIR="$BASE_DIR/results/lab4/pdb_files"
PYTHON_SCRIPT="$BASE_DIR/software/lab4/test2.py"
MATCH_CSV="$BASE_DIR/results/lab4/MATCHED.csv"
MISMATCH_CSV="$BASE_DIR/results/lab4/UNMATCH.csv"

# makes sure dirs exist
mkdir -p "$PDB_DIR"
mkdir -p "$BASE_DIR/results/lab4"

# Reset CSV files with headers
echo "UniProt_ID,Status" > "$MATCH_CSV"
echo "UniProt_ID,Status" > "$MISMATCH_CSV"



# DOWNLOAD PDB
# reads list.txt for the ids
while read -r uid; do
    # Clean ID and skip empty lines
    # -d delete
    # xargs removes spacing
    uid_clean=$(echo "$uid" | tr -d '\r' | xargs)
    # -z zero, if empty
    # continue skips loop is id empty
    [[ -z "$uid_clean" ]] && continue

    PDB_FILE="$PDB_DIR/${uid_clean}.pdb"
    # echo "------------------------------------------"
    echo "Processing ID: $uid_clean"

    # ! -s if file is empty/does not exist
    if [[ ! -s "$PDB_FILE" ]]; then



        echo "Downloading structure..."
        
        # Try AlphaFold v4 (The standard)
        curl -s -f -L "https://alphafold.ebi.ac.uk/files/AF-${uid_clean}-F1-model_v4.pdb" -o "$PDB_FILE"

        # Try AlphaFold v3 (Fallback for older records like P07550)
        if [[ ! -s "$PDB_FILE" ]]; then
            curl -s -f -L "https://alphafold.ebi.ac.uk/files/AF-${uid_clean}-F1-model_v3.pdb" -o "$PDB_FILE"
        fi

        # Try the "Swiss-Model" Repository (Another great source for P-numbers)
        if [[ ! -s "$PDB_FILE" ]]; then
            curl -s -f -L "https://swissmodel.expasy.org/repository/uniprot/${uid_clean}.pdb" -o "$PDB_FILE"
        fi


    fi

    # MATCH OR NO MATCH
    # -s if file has content
    if [[ -s "$PDB_FILE" ]]; then
        # Capture the result from the Python script
        # We assume the Python script prints "MATCH" or "MISMATCH" at the very end
        # Capture only the last line of output (the RESULT line)
        
        full_output=$(python3 "$PYTHON_SCRIPT" --uniprot "$uid_clean" --pdb "$PDB_FILE")
        # echo "$result" # This lets you see the "Mismatch Diagnostic"
        result=$(echo "$full_output" | tail -n 1) # This picks "RESULT_MISMATCH" for the CSV logic
        python3 "$PYTHON_SCRIPT" --uniprot "$uid_clean" --pdb "$PDB_FILE"

        if [[ "$result" == "RESULT_MATCH" ]]; then
            echo "$uid_clean,Match" >> "$MATCH_CSV"
            echo "Result: Added to MATCH.csv"
            echo "------------------------------------------"

        else
            echo "$uid_clean,Mismatch" >> "$MISMATCH_CSV"
            echo "Result: Added to MISMATCH.csv"
            echo "------------------------------------------"

        fi
    else
        echo "Error: Structure download failed for $uid_clean"

    fi
done < "$ID_LIST"

echo "Analysis Complete. Check: $BASE_DIR/results/lab4"


