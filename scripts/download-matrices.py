import os
import sys
import tarfile
import shutil
import requests

MATRIX_URLS = [
    "https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage4.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Bai/mhda416.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/HB/mcfe.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Bai/olm1000.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Sandia/adder_dcop_32.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/HB/west2021.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/DRIVCAV/cavity10.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Zitney/rdist2.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Williams/cant.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Simon/olafu.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Cube_Coup_dt0.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Janna/ML_Laplace.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk17.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Williams/mac_econ_fwd500.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Bai/mhd4800a.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Williams/cop20k_A.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Simon/raefsky2.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Bai/af23560.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Norris/lung2.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Fluorem/PR02R.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/FEM_3D_thermal1.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal1.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal2.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/thermomech_TK.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt80.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/IBM_EDA/dc1.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0302.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_1_k101.tar.gz",
    "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-PA.tar.gz",
]


def download_extract_exact_mtx(url, out_dir):
    # Get matrix base name (e.g. "olm1000")
    base_name = url.split("/")[-1].replace(".tar.gz", "")
    archive_name = base_name + ".tar.gz"
    archive_path = os.path.join(out_dir, archive_name)

    print(f"🔽 Downloading {archive_name}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"📦 Extracting {archive_name}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(out_dir)

    extracted_folder = os.path.join(out_dir, base_name)
    expected_mtx = base_name + ".mtx"
    expected_path = os.path.join(extracted_folder, expected_mtx)

    if not os.path.isfile(expected_path):
        print(
            f"⚠️ Warning: Expected file {expected_mtx} not found in {extracted_folder}"
        )
        return

    # Move and clean up
    final_path = os.path.join(out_dir, expected_mtx)
    shutil.move(expected_path, final_path)
    print(f"✅ Saved: {final_path}")

    os.remove(archive_path)
    shutil.rmtree(extracted_folder)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <output_directory>")
        sys.exit(1)

    output_dir = sys.argv[1]
    os.makedirs(output_dir, exist_ok=True)

    for url in MATRIX_URLS:
        try:
            download_extract_exact_mtx(url, output_dir)
        except Exception as e:
            print(f"❌ Error processing {url}: {e}")
