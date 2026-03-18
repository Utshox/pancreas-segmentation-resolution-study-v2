import os
import synapseclient
import synapseutils
import argparse

def download_btcv(token):
    base_dir = '/scratch/lustre/home/kayi9958/ish/data_external_btcv'
    os.makedirs(base_dir, exist_ok=True)
    
    print("--- Logging into Synapse ---")
    syn = synapseclient.Synapse()
    syn.login(authToken=token)
    
    print(f"--- Starting Download of BTCV (syn3193805) to {base_dir} ---")
    # syncFromSynapse will download the entire folder structure recursively
    files = synapseutils.syncFromSynapse(syn, 'syn3193805', path=base_dir)
    print(f"--- Download Complete. Total files: {len(files)} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, required=True, help='Synapse Auth Token')
    args = parser.parse_args()
    
    download_btcv(args.token)
