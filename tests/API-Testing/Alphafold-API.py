#File testing Alphafold's API Inference and Prediction
import requests
def test_alphafold_api(sequence):
    url = "https://alphafold.ebi.ac.uk/api/prediction"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "sequence": sequence
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        print("Prediction successful!")
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
# Example usage
if __name__ == "__main__":
    sequence = "MKTAYIAKQRQISFVKSHFSRQDILDLWQYVQG"
    result = test_alphafold_api(sequence)
    if result:
        print("Result:", result)  # Print the prediction result
    else:
        print("Failed to get a valid response.")
# Note: Ensure you have the 'requests' library installed in your Python environment.
# You can install it using pip if it's not already installed:
# pip install requests
# This code tests the Alphafold API for protein structure prediction using a given amino acid sequence.
# Make sure to handle the API response appropriately in your application.
