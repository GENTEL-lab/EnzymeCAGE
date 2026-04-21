import os
import argparse
import requests
from functools import partial
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

CNT_FAILED = 0
failed_uids = []


def get_cif_url(uid):
    """Query the AlphaFold API to get the current CIF download URL."""
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uid}"
    response = requests.get(api_url, timeout=30)
    if response.status_code == 200:
        data = response.json()
        if data and "cifUrl" in data[0]:
            return data[0]["cifUrl"]
    return None


def download_alphafold_structure(uid, save_dir):
    output_file = os.path.join(save_dir, f"{uid}.cif")

    if os.path.exists(output_file):
        return True

    try:
        cif_url = get_cif_url(uid)
        if cif_url is None:
            # Fallback to current known version
            cif_url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v6.cif"

        response = requests.get(cif_url, timeout=60)
        if response.status_code == 200:
            with open(output_file, 'wb') as file:
                file.write(response.content)
            return True
        else:
            print(f"No structure found for {uid}.")
            global CNT_FAILED, failed_uids
            CNT_FAILED += 1
            failed_uids.append(uid)
            return False
    except Exception as e:
        print(f"Error downloading {uid}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--uid_col', type=str, default='UniprotID', help='The column name of Uniprot ID in the data file')
    parser.add_argument('--n_process', type=int, default=10)
    args = parser.parse_args()
    
    assert os.path.exists(args.data_path), f"Data file {args.data_path} does not exist."
    save_dir = os.path.join(os.path.dirname(args.data_path), 'af2_structures')
    os.makedirs(save_dir, exist_ok=True)
    
    df_data = pd.read_csv(args.data_path)
    if args.uid_col not in df_data.columns:
        lower_to_col = {col.lower(): col for col in df_data.columns}
        matched_col = lower_to_col.get(args.uid_col.lower())
        if matched_col:
            args.uid_col = matched_col
        elif 'uniprotid' in lower_to_col:
            args.uid_col = lower_to_col['uniprotid']
        else:
            raise ValueError(f"Column {args.uid_col} does not exist in the data file.")
    
    uniprot_ids = set(df_data[args.uid_col])
    uids_to_download = [each for each in uniprot_ids if isinstance(each, str)]
    running_func = partial(download_alphafold_structure, save_dir=save_dir)

    with Pool(args.n_process) as pool:
        for _ in tqdm(pool.imap(running_func, uids_to_download), total=len(uids_to_download)):
            pass

    n_success = len(uids_to_download) - CNT_FAILED
    print(f"Downloaded {n_success} structures successfully. Failed: {CNT_FAILED}")
    
    if failed_uids:
        df_failed = pd.DataFrame({args.uid_col: failed_uids})
        save_path = os.path.join(save_dir, 'failed_uids.csv')
        df_failed.to_csv(save_path, index=False)
        print(f"Failed proteins: {save_path}")


if __name__ == "__main__":
    main()
