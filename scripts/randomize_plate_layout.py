# randomly assign patients and healthy controls to plates for one diagnosis
# include constraints
# randomize layout within plate
# output as csv (2D and flattened)

import pandas as pd 
import numpy as np
import argparse
from itertools import product

times = ['Time point 1', 'Time point 2', 'Time point 3', 'Time point 4', 'Time point 5']
rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
cols = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
boundary = ['A' + c for c in cols] + ['H' + c for c in cols] + [r + '01' for r in rows[1:-1]] + [r + '12' for r in rows[1:-1]]


def randomize_layout(diagnosis, plate_info, patient_info):
  np.random.seed(123144)
  # shuffle healthy order
  all_healthy = np.array([f'H{i:02d}' for i in range(1, 51)])
  np.random.shuffle(all_healthy[7:])  # first 7 already used so we cannot randomize them anymore
  print('healthy order:', all_healthy)

  print('Randomizing layouts for', diagnosis)
  np.random.seed(92582 * len(diagnosis))
  free_patients = patient_info.index
  if not plate_info['Must include'].isna().all():
    constraint_patients = [p.strip() for p in ','.join(plate_info['Must include'].dropna()).split(',')]
    free_patients = free_patients[~np.isin(free_patients, constraint_patients)].values

  for plate in plate_info.index:
    # add patients that must be included
    if not plate_info['Must include'].isna().loc[plate]:
      patients = [p.strip() for p in plate_info.loc[plate, 'Must include'].split(',')]
    else: 
      patients = []
    # select remaining patients randomly
    random_patients = np.random.choice(free_patients, plate_info.loc[plate, 'Patient samples'] - len(patients), 
                                       replace=False).tolist()
    free_patients = free_patients[~np.isin(free_patients, random_patients)]
    patients += random_patients

    # healthy patients
    left = 5 + (plate - 3) * 2
    healthy = all_healthy[left:left+plate_info.loc[plate, 'Healthy samples']].tolist()

    # all ids within plate including technical replicates
    ids_unique = healthy
    ids_replicates = healthy * (plate_info.loc[plate, 'Healthy technical replicates'] - 1)
    for p in patients:
      p_times = patient_info.loc[p, times].dropna().values.tolist()
      ids_unique += p_times
      ids_replicates += p_times * (plate_info.loc[plate, 'Patient technical replicates'] - 1)
    empty = ['NA'] * (len(rows) * len(cols) - len(ids_unique) - len(ids_replicates))

    # pick NA locations uniformly at random
    locs = list(map(lambda t: t[0]+t[1], product(rows, cols)))
    np.random.shuffle(locs)
    na_locs = locs[:len(empty)]

    # pick samples for remaining boundary locations without replicates
    boundary_locs = list(set(boundary) - set(na_locs))
    np.random.shuffle(ids_unique)
    boundary_samples = ids_unique[:len(boundary_locs)]

    # shuffle all remaining samples
    inner_locs = list((set(locs) - set(boundary)) - set(na_locs))
    inner_samples = ids_unique[len(boundary_locs):] + ids_replicates
    np.random.shuffle(inner_samples)

    # assemble plate
    layout = np.full((len(rows), len(cols)), 'P999_9')   # fill values has correct length
    layout = pd.DataFrame(layout, index=rows, columns=cols)
    for sample, loc in zip(empty + boundary_samples + inner_samples, na_locs + boundary_locs + inner_locs):
      layout.loc[loc[0], loc[1:]] = sample

    # save 2D table
    layout.to_csv(f'meta/layout/plate_{plate}_2d_layout.csv')

    # save flattened table
    flat_layout = pd.melt(layout.reset_index(), id_vars=['index'], value_vars=cols, value_name='sample')
    flat_layout['well'] = flat_layout['index'] + flat_layout['variable']
    flat_layout['condition'] = ''
    flat_layout.loc[flat_layout['sample'].str.startswith('NA'), 'condition'] = 'NA'
    flat_layout.loc[flat_layout['sample'].str.startswith('P'), 'condition'] = diagnosis
    flat_layout.loc[flat_layout['sample'].str.startswith('H'), 'condition'] = 'Healthy'
    flat_layout.loc[flat_layout['sample'].str.startswith('PHC'), 'condition'] = 'Correction control'
    flat_layout = flat_layout[['well', 'condition', 'sample']].sort_values('well').set_index('well')
    flat_layout.to_csv(f'meta/layout/plate_{plate}_layout.csv')



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--diagnosis', required=True, type=str, 
                      help='Diagnosis name to select from plate info csv file and arrange ' +
                           'plate layouts for. Must appear in \'Diagnosis\' column.')
  parser.add_argument('-s', '--sample_info', required=True, nargs='+', type=str, help='Sheet names in sample_info.csv for selected diagnosis.')
  args = parser.parse_args()

  plate_info = pd.read_csv('meta/plates.csv')
  plate_info = plate_info[plate_info['Diagnosis'] == args.diagnosis]
  plate_info = plate_info.set_index('Plate ID')
  # plate_info.columns = plate_info.columns.str.replace('\n', ' ').str.replace('  ', ' ')
  # rename to distinguish 'technical replicates' columns to distinguish
  plate_info.columns = ['Diagnosis', 'Patient samples', 'Multiplied by Time points',
        'Patient technical replicates', 'Total patient samples', 'Healthy samples',
        'Healthy technical replicates', 'Total healthy samples', 'Correction control',
        'Control technical replicates', 'Total correction control samples',
        'Total samples per plate', 'Must include']

  patient_infos = [pd.DataFrame()]
  for sheet in args.sample_info:
    info = pd.read_excel('meta/sample_info.xlsx', skiprows=3, sheet_name=sheet)
    info.columns = info.columns.str.replace('\n', ' ').str.replace('  ', ' ')
    info = info.set_index('Patient ID')
    patient_infos.append(info)
  patient_info = pd.concat(patient_infos)

  randomize_layout(args.diagnosis, plate_info, patient_info)