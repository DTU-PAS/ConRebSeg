# ConRebSeg: A Segmentation Dataset for Reinforced Concrete Construction
[**DTU Electro, Automation & Control Group**](https://electro.dtu.dk/research/research-areas/electro-technology/automation-og-control)

[Patrick Schmidt](https://orbit.dtu.dk/en/persons/patrick-schmidt) and [Lazaros Nalpantidis](https://lanalpa.github.io/)

![Example of the ConRebSeg dataset](example.gif)

----
This repository contains code to initizliaze the ConRebSeg dataset. The main 
component of the repository is the Python script `import_dataset.py`, which 
initializes the FiftyOne dataset and downloads the data for the dataset in a fully
automated manner.

The accompanying self-collected data is indexed in the Technical University of Denmark (DTU) Data repository.

DOI: [10.11583/DTU.26213762](https://doi.org/10.11583/DTU.26213762)

**NB:** Note that the initialization script will download from YouTube. Always ensure you comply with local laws regarding this process.

## Initialization
To initialize the FiftyOne dataset and to download the data, please follow these steps:
1) Create a Python 3.12 virtual environment with Conda or venv and install the pip dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2) Execute the initialization script. Note that this will download the self-collected data from DTU Data (~22 GB) and will unpack it in this directory under `data/langebro` and `data/vester_sogade`. It will also download videos from YouTube, extract frames and store them under `data/youtube`.
   ```bash
   python import_dataset.py
   ```
   The command above in general doesn't need to be modified. You can opt for the following options:
   - `--skip_integrity_check`: The script checks the integrity of each sample in the dataset to ensure that the download and import was execute without errors and that the data hasn't changed. **It is not recommended to use this option**, but it can save you time if needed.
   - `--skip_yt_download`: This disables the download process for YouTube videos. Useful if you're not certain about the legal status of downloading YouTube data in your jurisdiction.
   - `--skip_selfcollected` This disables the download of the self-collected sequences.

3) After the script has been executed successfully, open the FiftyOne app as follows:
   ```bash
   fiftyone app launch ConRebSeg
   ```
   This will open a browser window and present you with the dataset explorer. Happy exploring!

## Further information
Further information about this dataset, its structure and characteristics can be found in the accompanying journal article [Segmentation dataset for reinforced concrete construction - ScienceDirect](https://doi.org/10.1016/j.autcon.2025.105990)

## Contributors
Thanks to [Rasmus E. Andersen](https://scholar.google.com/citations?user=CxGlLlAAAAAJ&hl=en), Javier Casas Lorenzo and Carlos Gascon Bononad for helping me in the collection process! Thanks to Christiansen \& Essenbæk A/S for organizing access to the construction sites.

## Citation
If you found this dataset and the article/paper useful, please cite us as follows:
```
@article{SCHMIDT2025105990,
title = {Segmentation dataset for reinforced concrete construction},
journal = {Automation in Construction},
volume = {171},
pages = {105990},
year = {2025},
issn = {0926-5805},
doi = {https://doi.org/10.1016/j.autcon.2025.105990},
url = {https://www.sciencedirect.com/science/article/pii/S0926580525000305},
author = {Patrick Schmidt and Lazaros Nalpantidis},
keywords = {Dataset, Construction robotics, Segmentation, Rebar detection, Shotcrete, Digitization}
}
```

## Acknowledgements
This work has been funded and supported by the EU Horizon Europe project
“RobetArme” under the Grant Agreement 101058731.
