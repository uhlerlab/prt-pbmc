import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import glob
import argparse
import tifffile
from skimage.measure import regionprops
import pandas as pd
from skimage import measure
import numpy as np
import cv2 as cv

import sys
sys.path.append('/home/unix/hschluet/projects/chrometrics')
from nmco.nuclear_features import (
    global_morphology as BG,
    img_texture as IT,
    int_dist_features as IDF,
    boundary_local_curvature as BLC
)


def extract_single_cell_nuclear_chromatin_feat(image_3d_path:str,  
                                   calliper_angular_resolution:int = 10, 
                                   measure_simple_geometry:bool = True, 
                                   measure_calliper_distances:bool = True, 
                                   measure_radii_features:bool = True,
                                   step_size_curvature:int = 2, 
                                   prominance_curvature:float = 0.1, 
                                   width_prominent_curvature:int = 5, 
                                   dist_bt_peaks_curvature:int = 10,
                                   measure_int_dist_features:bool = True, 
                                   measure_hc_ec_ratios_features:bool = True, 
                                   hc_threshold:float = 1, 
                                   gclm_lengths:list = [1, 5, 20],
                                   measure_gclm_features: bool = True, 
                                   measure_moments_features: bool = True,
                                   normalize:bool=False):    
    """
    Function that reads in the raw and segmented/labelled images for a field of view and computes nuclear features. 
    Note this has been used only for DAPI stained images
    Args:
        image_3d_path: path pointing to the 3d crop image with mask
    """
    crop_3d = tifffile.imread(image_3d_path)
    raw_image = (crop_3d[:, 0].max(axis=0) * 255).astype(int)
    labelled_image = crop_3d[:, 1].max(axis=0).astype(int)

    # Insert code for preprocessing image
    # Eg normalize
    if normalize:
        raw_image = cv.normalize(
         raw_image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F
        )
        raw_image[raw_image < 0] = 0.0
        raw_image[raw_image > 255] = 255.0

    # Get features for the individual nuclei in the image
    props = measure.regionprops(labelled_image, raw_image)
    assert len(props) == 1
    props = props[0]
    
    features = pd.concat(
            [
                BG.measure_global_morphometrics(props.image, 
                                                angular_resolution = calliper_angular_resolution, 
                                                measure_simple = measure_simple_geometry,
                                                measure_calliper = measure_calliper_distances, 
                                                measure_radii = measure_radii_features).reset_index(drop=True),
                BLC.measure_curvature_features(props.image, step = step_size_curvature, 
                                            prominance = prominance_curvature, 
                                            width = width_prominent_curvature, 
                                            dist_bt_peaks = dist_bt_peaks_curvature).reset_index(drop=True),
                IDF.measure_intensity_features(props.image, props.intensity_image, 
                                            measure_int_dist = measure_int_dist_features, 
                                            measure_hc_ec_ratios = measure_hc_ec_ratios_features, 
                                            hc_alpha = hc_threshold).reset_index(drop=True),
                IT.measure_texture_features(props.image, props.intensity_image, lengths=gclm_lengths,
                                            measure_gclm = measure_gclm_features,
                                            measure_moments = measure_moments_features)],
            axis=1,
        )
    return features


def get_features(plate):
    info = pd.read_csv(f'/ewsc/hschluet/pbmc5/bundled_data/plate_{plate}_info.csv')

    img_dir = f'/ewsc/hschluet/pbmc5/seg_voronoi_otsu/Plate{plate}/3d_crops/'
    files = pd.DataFrame()
    files['path'] = glob.glob(f'{img_dir}/*.tiff')
    files['fname'] = files['path'].str.split('/', expand=True)[7]
    files['plate'] = plate
    files['well'] = files['fname'].str.split('Well', expand=True)[1].str[:3]
    files['series'] = files['fname'].str.split('_', expand=True)[3].astype(int)
    files['cell'] = files['fname'].str.split('_', expand=True)[4].str.split('.', expand=True)[0].astype(int)

    plate_info = info[info['plate'] == plate]
    old_shape = plate_info.shape
    plate_info = plate_info.merge(files, on=['plate', 'well', 'series', 'cell'], how='left')
    assert old_shape[0] == plate_info.shape[0]

    features = []
    for i in trange(len(plate_info)):
        try:
            feats = extract_single_cell_nuclear_chromatin_feat(plate_info.loc[i, 'path'],  
                                            step_size_curvature=5,
                                            hc_threshold=1.5,
                                            gclm_lengths=[5, 25, 100],
                                            normalize=True,
                                            prominance_curvature=0.1)
        except (ValueError, IndexError) as e:
            # chrometric feature extraction failed. perhaps because image isn't actually a real cell.
            print(e)
            features.append(pd.read_csv('meta/empty_feat.csv'))
            continue

        features.append(feats)

    features = pd.concat(features, ignore_index=True)
    save_dir = f'/ewsc/hschluet/pbmc5/bundled_data/'
    features.to_csv(f'{save_dir}plate_{plate}_chrometric.csv', index=False)
    return features
        


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--plates", required=True, nargs='+', type=int)
  args = parser.parse_args()
  for p in args.plates:
    get_features(p)