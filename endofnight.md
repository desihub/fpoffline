# Focal Plane End-of-Night Analysis

Describe the endofnight script: how is it run, maitained, etc

Describe the checks it performs and summary statistics it reports.

Describe the viewer at https://data.desi.lbl.gov/desi/users/dkirkby/endofnight/

## Data Products

Describe the directory layout at nersc...

### FVC Images

JPEG images showing the focal plane at the end of the night, stored as `fvc-back-YYYYMMDD.jpg` and `fvc-front-YYYYMMDD.jpg`.  These images are captured by the [end-of-night park procedure](https://desi.lbl.gov/trac/wiki/FPS/ObservingScripts#EndofNightPark) performed by the lead observer, then processed to remove instrument signatures and highlight relevant details.  There are two images: a back-illuminated image where only the fiber tips are visible, and a front-illuminated image where the focal plane itself is visible.

### Calibration Table

Stored as `calib-YYYYMMDD.csv` and derived from any calibration rows added during this observing night (but many nights do not generate any new rows).  See the [calibration database schema](https://docs.google.com/spreadsheets/d/1e8yyjNFI9nCOT_KsJAxI3uzl8qSKqhuiDVXtXvxHNqM/edit#gid=836120262) for details.

### Moves Table

Stored as `moves-YYYYMMDD.csv.gz` and derived from the rows added to the move table during this observing night. Many of these columns are directly copied from the moves database, which is documented [here](https://docs.google.com/spreadsheets/d/1e8yyjNFI9nCOT_KsJAxI3uzl8qSKqhuiDVXtXvxHNqM/edit#gid=0).

| Column | Unit | Description |
| ------ | ---- | ------------|
| time_recorded | hr | Hours relative to local noon before this observing night |
| pos_id | - | Positioner ID string, e.g. M01234 |
| location | - | 1000 * PETAL_LOC + DEVICE_LOC |
| ctrl_enabled | - | 0 = disabled, 1 = enabled (non-functional devices are always disabled) |
| move_cmd | - | string describing the move performed |
| log_note | - | log note attached to this move |
| exposure_id | - | unique exposure identifier |
| exposure_iter | - | iteration within this exposure (0=blind, 1=correction) |
| flags | - | bitmask of status flags documented [here](https://desi.lbl.gov/trac/wiki/FPS/PositionerFlags) |
| pos_t,p | deg | Internal theta and phi angles recorded in the moves database |
| req_t,p | deg | Requested angles stored in the log_note as "req_posintTP=(...,...)" if available or NaN |
| fvc_t,p | deg | Angles derived from an FVC spot if available or NaN. Uses an immediately following FVC feedback if present. |
| req_dt,dp | deg | Requested change of angles obtained as the sum of the angles appearing in the move_val1,2 text columns |
| act_dt,dp | deg | Actual change of angles obtained as the change in fvc_t,p from the previous to current row |
| ptl_x,y | mm | PTL coords from the moves DB converted to FP coords with nominal petal alignments |
| obs_x,y | mm | FP coords from the moves DB, derived from FVC spots |
| req_x,y | mm | requested petal coords from log note converted to FP coords with nominal petal alignments |
| pred_x,y| mm | predicted FP coords obtained by transforming pos_t,p using desimeter and nominal petal alignments |

Notes:
 - obs-ptl includes effects of [petal misalignments](https://observablehq.com/@dkirkby/desi-petal-metrology) implemented in PlateMaker (since ptl_x,y are calculated for nominal alignments)
 - obs-req is due to positioning errors and turbulence (or just turbulence for non-functional robots)
 - act/req != 1 indicates a potential motor problem or collision
 - fvc_t,p will be NaN for the blind (exposure_iter=1) positioning performed before each cosmic split since there is no accompanying FVC image.

### Summary Table

Stored as `fp-YYYYMMDD.ecsv` and...
