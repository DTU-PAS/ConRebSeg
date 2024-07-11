import argparse
import cv2
import fiftyone as fo
import os
import glob
import pandas as pd
import numpy as np
import json
import requests
import shutil
import logging
import hashlib

from tqdm import tqdm
from yt_dlp import YoutubeDL
from joblib import Parallel, delayed
from zipfile import ZipFile

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
    """Extracts frames from a video

    Args:
        video_path (str): Path to video file 
        output_folder (str): Path to folder to store video frames in
        step (int, optional): Extract every n-th frame. Defaults to 1.
    """
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

def check_hash(sid, sample_dict):
    """Computes filehash of FiftyOne sample

    Args:
        id (str): ID of dataset sample
        sample_dict (dict): A dict representing the FiftyOne sample, containing
                            metadata information about it. 

    Returns:
        dict: A dict containing the sample id, its filename, current hash and existing hash
    """
    return {
        'id' : sid, 
        'filepath': sample_dict['filepath'], 
        'existing_hash' : sample_dict['file_hash'].split('md5:')[1],
        'current_hash' : fo.core.utils.compute_filehash(sample_dict['filepath'], 
                                                        'md5')
    }

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='.',
                    help='Path to directory containing samples.json')
parser.add_argument('--skip_integrity_check', action='store_true',
                    help='Skip the integrity check for the samples')
parser.add_argument('--skip_yt_download', action='store_true', 
                    help='do not download youtube videos')
parser.add_argument('--skip_selfcollected', action='store_true',
                    help='do not download langebro and vester sogade data')

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    args = parser.parse_args()
    failed_integrity_check = []
    logging.basicConfig(level=logging.INFO)

    download_files = {
        'ConRebSeg.zip': 'https://figshare.com/ndownloader/files/47543210?private_link=8f14ff87159f1e0f6f11', 
        'metadata.json': 'https://figshare.com/ndownloader/files/47547509?private_link=8f14ff87159f1e0f6f11',
        'samples.json' : 'https://figshare.com/ndownloader/files/47547512?private_link=8f14ff87159f1e0f6f11'
    }

    # Read checksum file
    with open('checksums.md5', 'r') as f:
        checksums = {x.split(' ')[-1].replace('\n', '') : x.split(' ')[0] for x in f.readlines()}
    
    # Check and download metadata.json
    redownload = False
    if os.path.exists('metadata.json'):
        with open('metadata.json', 'rb') as f:
            redownload = hashlib.file_digest(f, 'md5').hexdigest() != checksums['metadata.json']
            if not redownload:
                logging.info("Not downloading metadata.json, as it already exists and is up-to-date")
    elif redownload or not os.path.exists('metadata.json'):
        logging.info('Downloading metadata.json')
        download_with_pbar(download_files['metadata.json'], 'metadata.json')
        with open('metadata.json', 'rb') as f:
            assert hashlib.file_digest(f, 'md5').hexdigest() == checksums['metadata.json'], "Something is wrong with metadata.json"
        logging.info("Download of metadata.json completed and integrity check passed.")
    
    # Check and download samples.json
    redownload = False
    if os.path.exists('samples.json'):
        with open('samples.json', 'rb') as f:
            redownload = hashlib.file_digest(f, 'md5').hexdigest() != checksums['samples.json']
            if not redownload:
                logging.info("Not downloading samples.json, as it already exists and is up-to-date")
    elif redownload or not os.path.exists('samples.json'):
        logging.info('Downloading samples.json')
        download_with_pbar(download_files['samples.json'], 'samples.json')
        with open('samples.json', 'rb') as f:
            assert hashlib.file_digest(f, 'md5').hexdigest() == checksums['samples.json'], "Something is wrong with samples.json"
        logging.info("Download of samples.json completed and integrity check passed.")

    # Import/Load ConRebSeg FiftyOne dataset
    if not fo.dataset_exists('ConRebSeg'):
        logger.info("Importing FiftyOne Dataset metadata")
        dataset = fo.Dataset.from_dir(
            dataset_dir = args.dataset_dir,
            dataset_type = fo.types.FiftyOneDataset,
            name='ConRebSeg'
        )
    else:
        dataset = fo.load_dataset('ConRebSeg')
        logging.info("ConRebSeg is already imported. Script will download missing data and do integrity check unless overridden by cmd args")

    # Download langebro/vestersogade samples
    if not args.skip_selfcollected:
        logging.info("Checking if all self-collected images are stored on disk")
        if not all([os.path.exists(x.filepath) for x in dataset.match_tags(['langebro', 
                                                                'vester_sogade'])]):
            redownload = False
            if os.path.exists('ConRebSeg.zip'):
                logging.info('ConRebSeg.zip already downloaded, checking integrity. This might take a while.')
                with open('ConRebSeg.zip', 'rb') as f:
                    redownload = hashlib.file_digest(f, 'md5').hexdigest() != checksums['ConRebSeg.zip']
            elif redownload or not os.path.exists('ConRebSeg.zip'):
                logging.info('ZIP archive with samples not found, downloading ConRebSeg.zip')
                download_with_pbar(download_files['ConRebSeg.zip'], 'ConRebSeg.zip')
                logging.info("Checking integrity of downloaded archive. This might take a while")
                with open('ConRebSeg.zip', 'rb') as f:
                    assert hashlib.file_digest(f, 'md5').hexdigest()  == checksums['ConRebSeg.zip']

            
            # Extract archive
            logging.info("Extracting ConRebSeg.zip")
            with ZipFile('ConRebSeg.zip', 'r') as zf:
                for member in tqdm(zf.infolist()):
                    zf.extract(member)
            shutil.move('ConRebSeg/langebro', 'data/langebro')
            shutil.move('ConRebSeg/vester_sogade', 'data/vester_sogade')
            os.removedirs('ConRebSeg')
            # os.remove('ConRebSeg.zip')

    # Check integrity of langebro and vester_sogade samples
    if not args.skip_integrity_check:
        logging.info('Checking integrity of langebro samples and vester_sogade samples')
        lange_sogade = dataset.match_tags(['langebro', 'vester_sogade'])
    
        hashes = Parallel(n_jobs=-1, prefer='threads',
                        verbose=11)(delayed(check_hash)(
            sample.id, json.loads(sample.to_json())) for sample in lange_sogade
            )
            
        hashes = pd.DataFrame.from_records(hashes, index='id')
        if (hashes['existing_hash'] == hashes['current_hash']).all():
            logging.info("Integrity check langebro and vester_sogade PASSED!")
        else:
            logging.warning("Integrity check langebro and vester_sogade FAILED!\n"+
                            'Check the summary at the end for a list of' +
                            ' sequences that need attention')
            failed_integrity_check.append(
                hashes[~(hashes['existing_hash'] == hashes['current_hash'])])
    
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

            if not args.skip_yt_download:
                with YoutubeDL(params={'format' : f'bv[height={height}][ext=mp4]', 
                                    'outtmpl' : '%(id)s.%(ext)s',
                                    'paths' : {'home' : '.tmp/'}, 
                                    'simulate' : False}) as ydl:
                    ydl.download(f'https://www.youtube.com/watch?v={yt_id}')

                # Extract frames
                video_file = glob.glob(os.path.join('.tmp', f'{yt_id}*'))[0]
                extract_frames(video_file, frame_dir, step=frame_step)
        else:
            logging.info("Frames for Youtube ID %s already exist, skipping download", yt_id)
        
        if not args.skip_integrity_check:
            # Check integrity
            logging.info("Checking integrity of %s frames", yt_id)
            hashes = [check_hash(sample.id, json.loads(sample.to_json()))
                      for sample in sequence]
            hashes = pd.DataFrame.from_records(hashes, index='id')
            if (hashes['existing_hash'] == hashes['current_hash']).all():
                logging.info('Result integrity check of %s: PASS!', yt_id)
            else:
                logging.warning('Result integrity check of %s: FAILED!\n' +
                             'Check the summary at the end for a list of' +
                             ' sequences that need attention', yt_id)
                failed_integrity_check.append(
                    hashes[~(hashes['existing_hash'] == hashes['current_hash'])])

    if len(failed_integrity_check) == 0:
        logging.info("Import of ConRebSeg completed successfully without errors.")
    else:
        all_failed = pd.concat(failed_integrity_check, axis=0)
        sequences = set([x.split('/')[-2] for x in all_failed['filepath']])
        logging.warning("Import of ConRebSeg is completed, but integrity errors have been detected." +
                        "Please check the following sequences and make sure the labels align with image contents:\n%s" % sequences)

    dataset.persistent = True
