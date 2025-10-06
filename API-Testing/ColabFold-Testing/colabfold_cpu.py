import os
import sys
import subprocess
from pathlib import Path
from colabfold.batch import run, get_queries
import py3Dmol

def ensure_weights():
    """Check if model weights exist, otherwise download them."""
    cache_dir = Path.home() / "Library" / "Caches" / "colabfold" / "params"
    required_file = cache_dir / "params_model_3_ptm.npz"
    if not os.path.exists(required_file):
        print("Model weights not found, downloading...")
        os.makedirs(cache_dir, exist_ok=True)
        # Try CLI first (two possible entrypoint names), then fall back to Python API
        tried = []
        for cli in ("colabfold_download", "colabfold-download"):
            try:
                subprocess.run([cli], check=True)
                return
            except FileNotFoundError:
                tried.append(cli)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed running {cli}: {e}") from e

        # Python API fallback
        try:
            from colabfold.download import download_alphafold_params
        except Exception as e:
            raise RuntimeError(
                "ColabFold not installed or CLI not found. Tried: "
                + ", ".join(tried)
            ) from e

        download_alphafold_params("alphafold2_ptm", cache_dir.parent)
    else:
        print("Model weights already present.")

def view_structure(pdb_file: str, width: int = 600, height: int = 400):
    """Render structure in browser or Jupyter."""
    with open(pdb_file, "r") as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_data, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()

    try:
        from IPython.display import display
        return view.show()
    except ImportError:
        # Write HTML for standalone viewing
        html_str = view._make_html()
        out_file = "structure_view.html"
        with open(out_file, "w") as f:
            f.write(html_str)
        print(f"Structure view written to {out_file}. Open it in your browser.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python colabfold_cpu.py inputs/example.fasta")
        sys.exit(1)

    fasta_file = sys.argv[1]
    if not os.path.exists(fasta_file):
        print(f"Error: {fasta_file} not found.")
        sys.exit(1)

    ensure_weights()

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    queries, is_complex = get_queries(fasta_file)

    print(f"Running ColabFold on CPU for {fasta_file}...")
    run(
        queries,
        output_dir,
        model_type="alphafold2_ptm",
        use_gpu=False,
        num_models=1,
        is_complex=is_complex
    )

    print(f"\n Prediction complete! Check the '{output_dir}' folder for results.")

    # Auto-visualize the top ranked PDB
    ranked_pdb = os.path.join(output_dir, "ranked_0.pdb")
    if os.path.exists(ranked_pdb):
        print(f"Visualizing structure: {ranked_pdb}")
        view_structure(ranked_pdb)
    else:
        print(" No ranked_0.pdb file found to visualize.")

if __name__ == "__main__":
    main()
