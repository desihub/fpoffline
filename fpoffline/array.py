import numpy as np

from . import const


def items(S, max_items=12):
    n = len(S)
    l = sorted([str(s) for s in S])[:max_items]
    if n > max_items:
        l.append('...')
    txt = ','.join(l)
    return f'({n}) {txt}'


class DeviceArray:

    def __init__(self, POS=True, ETC=False, FIF=False, GIF=False):
        PTL = const.get_petal_design()
        self.data = np.zeros((10, len(PTL.holes)))
        self.mask = PTL.holes.DEVICE_TYPE.map(dict(POS=POS, ETC=ETC, FIF=FIF, GIF=GIF)).to_numpy()
        self.all_locs = PTL.holes.DEVICE_LOC.to_numpy()
        self.ndata = np.count_nonzero(self.mask)
        self.locmap = PTL.locmap

    def _decode(self, locations):
        if isinstance(locations, slice) and (locations == slice(None)):
            locations = np.concatenate([petal_loc * 1000 + self.all_locs[self.mask] for petal_loc in range(10)])
        else:
            locations = np.asarray(locations)
        petal_locs = locations // 1000
        device_locs = locations % 1000
        invalid = ~np.isin(device_locs, self.all_locs[self.mask])
        if np.any(invalid):
            raise ValueError('Invalid locations: ' + items(locations[invalid]))
        return petal_locs, self.locmap[device_locs]

    def __getitem__(self, locations):
        return self.data[self._decode(locations)]

    def __setitem__(self, locations, values):
        values = np.asarray(values)
        self.data[self._decode(locations)] = values

    def _encode(self, selection):
        petal_locs, petal_idx = np.where(selection)
        locations = petal_locs * 1000 + self.all_locs[petal_idx]
        return locations

    def where(self, condition):
        return self._encode(condition(self.data) & self.mask)
