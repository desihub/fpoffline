#####################
# 
#
# @Author: Abby Bault
# @Date: 2025-02-21
# @Filename: update_offset_p.py
#
#
#####################

import argparse
import pathlib

import pandas as pd
import numpy as np
from astropy.table import Table

def update_offsets(file, new_offset_p = 0):
    '''
    function calculates new values of POS_P based on the given new value of OFFSET_P. creates a dataframe to save as a csv file to use with set_calibrations.py to update database values.

    Arguments
    ----------
    file (dataframe):   filtered dataframe from the end of night fp-obsnight.escv file. Expects that the columns DEVICE_ID, DEVICE_TYPE, POS_P, and OFFSET_P are included. 
    new_offset_p (int): new value to use for OFFSET_P. Will apply to ALL positioners in file.

    Returns
    ----------
    df (dataframe): dataframe with updated OFFSET_P and POS_P values to be used with set_calibrations.py to update the KPNO database.
    '''
    #keep only lines in table that are positioners, not fiducuials 
    file = file[file['DEVICE_TYPE'] == 'POS']
    list_of_posids = file['DEVICE_ID']
    #get list of new offset_p values that is the same length as list_of_posids
    offset_p = [new_offset_p] * len(list_of_posids)
    #create array of True for the two commit columns in df/table
    commit = [True for posid in list_of_posids]
    
    pos_p = []
    for posid in list_of_posids:
        old_offset_p = file[file['DEVICE_ID'] == str(posid)]['OFFSET_P'].iloc[0]
        old_pos_p = np.asarray(file[file['DEVICE_ID'] == str(posid)]['POS_P'])[-1]
        new_pos_p = old_pos_p + old_offset_p
        # print(new_pos_p)
        pos_p.append(new_pos_p)

    #create df
    data = {'POS_ID': list_of_posids, 'POS_P': pos_p, 'OFFSET_P': offset_p, 'COMMIT_OFFSET_P': commit, 'COMMIT_POS_P': commit}
    df = pd.DataFrame(data)


    # df.to_csv('offset_p_update_test.csv', index = False)
    return df
    

def main():
    parser = argparse.ArgumentParser(
        description='Create csv file to update offset_p and pos_p values in the database. Uses end of night output located at NERSC in desi/spectro/data/OBSNIGHT/EXPID/fp-OBSNIGHT.ecsv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--obsnight', type = int, required = True, help = 'CURRENT obsnight in the form YYYYMMDD')
    parser.add_argument('--expid', type = int, required = True, help = 'CURRENT expid of the park robots script run on obsnight. in the form NNNNNN.')
    parser.add_argument('--output', type = pathlib.Path, default = pathlib.Path('.'), help = 'path to save csv file. default is current directory')
    parser.add_argument('--offset', type = float, default = 0., help = 'new value to use for OFFSET_P. default is 0.')
    parser.add_argument('--cut', type = float, default = 15., help = 'absolute cutoff value of offset_p to use. default is 15')
    parser.add_argument('--linphi-only', action = 'store_true', help = 'whether or not to only update offsets for linphi positioners')
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'verbosity level')

    #read in arguments and set common variables
    args = parser.parse_args()
    obsnight = args.obsnight
    expid = args.expid

    #set paths to end of night file
    ROOT = pathlib.Path('/global/cfs/cdirs/desi/spectro/data/')
    filename = f'fp-{obsnight}.ecsv'
    path = ROOT/str(obsnight)/('00' + str(expid))/filename

    #read in as astropy table
    if args.verbose: print('reading in data')
    file = Table.read(path)
    #convert to pandas df (faster)
    file = file.to_pandas()

    #select only linphi positioners???
    if args.linphi_only == True:
        if args.verbose: print('selecting only linear phi positioners')
        file = file[file['LINPHI'] == True]
    
    #select only positioners with offsets > the cutoff
    if args.verbose: print(f'OFFSET_P cut being used: |{args.cut}|')
    file = file[np.abs(file['OFFSET_P']) > args.cut]
    
    if args.verbose: print(f'changing offsets to {args.offset} for {len(file)} positioners.')

    
    #create df called csv to update offsets
    csv = update_offsets(file, args.offset)
    csv_name = f'offsets-{args.obsnight}.csv'

    #save df to csv file
    if args.verbose: print(f'saving output to: {args.output/csv_name}')
    csv.to_csv(args.output/csv_name, index = False)
    
    print('Make sure to run set_calibrations.py at KPNO to commit these changes to the database.')
    


if __name__ == '__main__':
    main()
