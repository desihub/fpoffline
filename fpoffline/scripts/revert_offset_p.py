#####################
# 
#
# @Author: Abby Bault
# @Date: 2025-02-21
# @Filename: revert_offset_p.py
#
#
#####################

import argparse
import pathlib

import pandas as pd
import numpy as np
from astropy.table import Table
from datetime import datetime

def revert_offsets(oldfile, currentfile):
    '''
    function to revert to old values of OFFSET_P and calculates new POS_P values based on the old OFFSET_P values. Creates a dataframe to save as a csv file to use with set_calibrations.py (KPNO) to update database values. Assumes oldfile and newfile are already filtered for selected positioners to only update previously changed values from update_offset_p.py.

    Arguments
    ----------
    oldfile (dataframe):     filtered dataframe from the end-of-night fp-obsnight.escv file. This is the selected OFFSET_P values to return to. Expects that the columns DEVICE_ID, DEVICE_TYPE, POS_P and OFFSET_P are included. 
    currentfile (dataframe): filtered dataframe from the end-of-night fp-obsnight.escv file. Must be the most recent end-of-night file or the POS_P values will be out of date. Expects that the columns DEVICE_ID, LOCATION, POS_P, and OFFSET_P are included. 

    Returns
    ----------
    df (dataframe): dataframe with updated OFFSET_P and POS_P values to be used with set_calibrations.py to update the KPNO database.
    '''

    #assert statements to ensure that files are same length and in same order
    assert len(oldfile) == len(currentfile)
    assert oldfile['LOCATION'].all() == currentfile['LOCATION'].all()
    
    #create array of True for the two commit columns in df/table
    commit = [True for i in range(len(oldfile))]

    #get old offset_p values that will be used to revert the database
    old_offsetp = oldfile['OFFSET_P']

    #get current pos_p values 
    current_posp = currentfile['POS_P']

    #calculate new pos_p values based on old offset and current pos_p
    new_posp = current_posp - old_offsetp

    #create df
    data = {'POS_ID': oldfile['DEVICE_ID'], 'POS_P': new_posp, 'OFFSET_P': old_offsetp, 'COMMIT_OFFSET_P':commit, 'COMMIT_POS_P':commit}
    df = pd.DataFrame(data)

    return df
    
    

def main():
    parser = argparse.ArgumentParser(
        description='Create csv file to revert offset_p to previous values and update pos_p values in the database. Uses end of night output located at NERSC in desi/spectro/data/OBSNIGHT/EXPID/fp-OBSNIGHT.ecsv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--csv', type = pathlib.Path, required = True, help = 'path to most recent update_offsets csv file or csv saved list of POS_ID to revert')
    parser.add_argument('--old-obsnight', type = int, required = True, help = 'OLD obsnight in the form YYYYMMDD to use for reverting back to old offset values')
    parser.add_argument('--old-expid', type = int, required = True, help = 'OLD expid of the park robots script run on obsnight in the form NNNNNN to use for reverting back to old offset values')

    parser.add_argument('--current-obsnight', type = int, required= True, help = 'CURRENT obsnight in the form YYYYMMDD')
    parser.add_argument('--current-expid', type = int, required = True, help = 'CURRENT expid of the park robots script run on obsnight. in the form NNNNNN.')

    parser.add_argument('--output', type = pathlib.Path, default = pathlib.Path('.'), help = 'path to save csv file. default is current directory')
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'verbosity level')

    #read in arguments and set common variables
    args = parser.parse_args()
    old_obsnight = args.old_obsnight
    old_expid = args.old_expid
    current_obsnight = args.current_obsnight
    current_expid = args.current_expid

    #set paths to old and current end of night files
    ROOT = pathlib.Path('/global/cfs/cdirs/desi/spectro/data/')
    old_filename = f'fp-{old_obsnight}.ecsv'
    old_path = ROOT/str(old_obsnight)/('00' + str(old_expid))/old_filename
    if args.verbose: print(f'Reverting to OFFSET_P values from {old_obsnight}/{old_expid}.')
    current_filename = f'fp-{current_obsnight}.ecsv'
    current_path = ROOT/str(current_obsnight)/('00' + str(current_expid))/current_filename
    if args.verbose: print(f'Using POS_P values from {current_obsnight}/{current_expid}.')
    

    #read in as astropy table and convert to pandas dataframe
    if args.verbose: print('reading in data')
    oldfile = Table.read(old_path)
    currentfile = Table.read(current_path)
    #convert to pandas df (faster)
    oldfile = oldfile.to_pandas()
    currentfile = currentfile.to_pandas()

    #read in posids that you want to revert offset_p for 
    csv = pd.read_csv(args.csv)
    list_of_posids = csv['POS_ID']
    if args.verbose: print(f'Found {len(list_of_posids)} positioners to revert.')
    
    #create masks for the old data and current data based on list_of_posids
    oldmask = np.isin(oldfile['DEVICE_ID'], list_of_posids)
    currentmask = np.isin(currentfile['DEVICE_ID'], list_of_posids)

    #select from the old data and current data only the posids to change
    # if args.verbose: print('Selecting positioners')
    oldfile = oldfile[oldmask]
    currentfile = currentfile[currentmask]

    #create df called csv_revert to update offsets and posp
    csv_revert = revert_offsets(oldfile, currentfile)
    csv_name = f'revert-offsets-{datetime.today().strftime("%Y%m%d")}.csv'

    #save df as csv file
    if args.verbose: print(f'saving output to: {args.output/csv_name}')
    csv_revert.to_csv(args.output/csv_name, index = False)
    
    print('Make sure to run set_calibrations.py at KPNO to commit these changes to the database.')
    


if __name__ == '__main__':
    main()
