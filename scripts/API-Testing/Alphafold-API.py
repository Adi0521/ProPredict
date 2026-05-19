# Tests ESMFold API inference and prediction
# ESMFold accepts a raw sequence (application/x-www-form-urlencoded)
# and returns a PDB string with pLDDT scores in the B-factor column.
import requests


def test_esmfold_api(sequence: str):
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(url, data=sequence, headers=headers)

    if response.status_code == 200:
        pdb_text = response.text
        print("Prediction successful!")

        # Extract pLDDT from B-factor column of CA atoms
        plddt_scores = []
        for line in pdb_text.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    plddt_scores.append(float(line[60:66].strip()))
                except ValueError:
                    pass

        if plddt_scores:
            mean_plddt = sum(plddt_scores) / len(plddt_scores)
            print(f"Residues predicted: {len(plddt_scores)}")
            print(f"Mean pLDDT: {mean_plddt:.2f}")
        else:
            print("Warning: no CA atoms found in response")

        return pdb_text
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


if __name__ == "__main__":
    sequence = "MKTAYIAKQRQISFVKSHFSRQDILDLWQYVQG"
    result = test_esmfold_api(sequence)
    if result:
        print(f"\nFirst 200 chars of PDB:\n{result[:200]}")
    else:
        print("Failed to get a valid response.")
