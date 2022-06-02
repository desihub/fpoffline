# Focal Plane End-of-Night Analysis

## Data Products

### Moves Table

Stored as `moves-YYYYMMDD.csv.gz`

| Column | Description | Rounding | Unit |
| ------ | ------------| -------- | ---- |
| pos_t,p | internal angles from the moves DB | 0.01 | deg |
| obs_x,y | FP coords from the moves DB | 1e-5 | mm |
| ptl_x,y | PTL coords from the moves DB converted to FP coords with nominal alignments | 1e-5 | mm |
| req_x,y | requested petal coords from log note converted to FP coords with nominal alignments | mm |


Notes:
 - obs-ptl is due to petal misalignments
 - ptl-req is due to positioning errors
