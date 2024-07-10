import argparse
import cv2
import fiftyone as fo
import os
import glob
import pandas as pd
import numpy as np
import json
import requests

from tqdm import tqdm
from yt_dlp import YoutubeDL
from joblib import Parallel, delayed

def download_with_pbar(url, filepath):
    """Downloads data from an URL using tqdm and requests

    Copied this code from https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    Args:
        url (str): URL to download from
        filepath (str): path to file to store download in

    Raises:
        RuntimeError: If the download fails 
    """
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True, timeout=20)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(filepath, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")

def extract_frames(video_path, output_folder, step=1):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    progress_bar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), unit="frames")
    with progress_bar as pb:
        while True:
            ret, frame = cap.read()

            # Break the loop if the video is finished
            if not ret:
                break

            # Save every i-th frame
            if frame_count % step == 0:
                frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)

            frame_count += 1
            pb.update(1)

        cap.release()

def check_hash(id, sample_dict):
    return {
        'id' : id, 
        'filename': os.path.split(sample_dict['filepath'][1]), 
        'existing_hash' : sample_dict['file_hash'].split('md5:')[1],
        'current_hash' : fo.core.utils.compute_filehash(sample_dict['filepath'], 
                                                        'md5')
    } 
        
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='.',
                    help='Path to directory containing samples.json')

if __name__ == '__main__':
    args = parser.parse_args()

    download_files = {
        'ConRebSeg.zip': 'https://figshare.com/ndownloader/files/47543210?private_link=8f14ff87159f1e0f6f11', 
        'metadata.json': 'https://figshare.com/ndownloader/files/47547509?private_link=8f14ff87159f1e0f6f11',
        'samples.json' : 'https://figshare.com/ndownloader/files/47547512?private_link=8f14ff87159f1e0f6f11'
    }
    
    download_with_pbar(download_files['metadata.json'], 'metadata.json')
    download_with_pbar(download_files['samples.json'], 'samples.json')

    dataset = fo.Dataset.from_dir(
        dataset_dir = args.dataset_dir,
        dataset_type = fo.types.FiftyOneDataset,
        name='ConRebSeg'
    )

    print(dataset)

    # Download langebro/vestersogade samples
    if not all([os.path.exists(x) for x in dataset.match_tags(['langebro', 
                                                               'vester_sogade'])]):
        download_with_pbar(download_files['ConRebSeg.zip'], 'ConRebSeg.zip')

    # Check integrity of langebro and vester_sogade samples
    print('Checking integrity of langebro samples and vester_sogade samples...', end='')
    lange_sogade = dataset.match_tags(['langebro', 'vester_sogade'])
 
    hashes = Parallel(n_jobs=-1, prefer='threads',
                      verbose=11)(delayed(check_hash)(
        sample.id, json.loads(sample.to_json())) for sample in lange_sogade
        )
        
    hashes = pd.DataFrame.from_records(hashes, index='id')
    assert (hashes['existing_hash'] == hashes['current_hash']).all(), print(hashes[~(hashes['existing_hash'] == hashes['current_hash'])])
    print("PASS!")
    
    # Create a list of youtube_ids:
    yt_ids = list(set(dataset.values('youtube_id')))
    yt_ids.remove(None)
    yt_ids = sorted(yt_ids)
    print(yt_ids)

    # Loop through the IDs and download the video
    for yt_id in yt_ids:
        print(yt_id)
        sequence = dataset.match(fo.ViewField('youtube_id') == yt_id)
        
        # Check if exists
        filepaths = sequence.values('filepath')
        if not all([os.path.exists(filepath) for filepath in filepaths]):

            # Extract video height to get quality
            height = set(sequence.values('metadata.height'))
            assert len(height) == 1, 'Inconsistent frame size'
            height = list(height)[0]

            # Extract frame step
            frame_step = np.unique(np.diff(sorted(sequence.values('frame_num'))))
            
            # Cefw94KfuI0 has no frame 200 - that's intended
            assert len(frame_step) == 1 or yt_id == 'Cefw94KfuI0', 'Inconsistent frame_step'
            frame_step = list(frame_step)[0]

            # Extract directory
            frame_dir = set([
                os.path.split(x)[0] for x in sequence.values('filepath')
                ])
            assert len(frame_dir) == 1
            frame_dir = list(frame_dir)[0]

            # Create temporary directory to download videos and delete them again
            if not os.path.isdir('.tmp'):
                os.makedirs('.tmp')

            with YoutubeDL(params={'format' : f'bv[height={height}]', 
                                'outtmpl' : '%(id)s.%(ext)s',
                                'paths' : {'home' : '.tmp/'}, 
                                'simulate' : False}) as ydl:
                ydl.download(f'https://www.youtube.com/watch?v={yt_id}')
            
            # Extract frames
            video_file = glob.glob(os.path.join('.tmp', f'{yt_id}*'))[0]
            extract_frames(video_file, frame_dir, step=frame_step)
        else:
            print(f"Frames for Youtube ID {yt_id} already exist, skipping download")
        
        # Check integrity
        print(f"Checking integrity of {yt_id} frames...", end='')
        hashes = [{'id': sample.id, 'filename' : sample.filename ,'existing_hash' : sample.file_hash.split(':')[1], 'current_hash' : fo.core.utils.compute_filehash(sample.filepath, 'md5')} for sample in sequence]
        hashes = pd.DataFrame.from_records(hashes, index='id')
        assert (hashes['existing_hash'] == hashes['current_hash']).all(), print(hashes[~(hashes['existing_hash'] == hashes['current_hash'])])
        # assert all([sample.file_hash.split(':')[1] == fo.core.utils.compute_filehash(sample.filepath, 'md5') for sample in sequence])

        print('PASS!')
        
        # Delete video
        # os.remove(video_file)
    # Delete temporary directory   
    # os.removedirs('.tmp')
    dataset.persistent = True
