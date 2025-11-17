Scripts should be run from the main project folder `pt-pbmc` rather than
the `scripts` folder to ensure relative file paths can be resolved correctly.

## Plate layout randomization
To generate randomized plate layouts, use `randomize.sh`. The layouts will be
saved in [`meta/layout`](../meta/layout/). Note that the final plate layouts we used are provided separately 
with the dataset and do not all correspond to the output of the current version of the script, as we needed 
to make some modifications along the way, e.g., due to missing and low cell count samples.

## Chrometric feature extraction
The `extract_chrometric.py` script can be used to extract chrometric features 
from images that have already been preprocessed (see `notebooks` for other pre-processing 
information). The script takes the plates it should be run on as an argument. As
extracting chrometric features can take a while it is recommended to run it on multiple
plates in parallel in separate screen sessions if enough CPU compute is available.
Example usage:

```
python3 extract_chrometric.py -p 1 2 3
```
