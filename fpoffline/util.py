"""General utilities with minimal dependencies.
"""


"""Flag names defined at https://desi.lbl.gov/trac/wiki/FPS/PositionerFlags
"""
flagNames = [
  "MATCHED","PINHOLE","POS","FIF","FVCERROR","FVCBAD","MOTION","GIF","ETC",
  "FITTEDPINHOLE","MATCHEDCENTER","AMBIGUOUS","FVCPROC","reserved13","STATIONARY",
  "CONVERGED","CTRLDISABLED","FIBERBROKEN","COMERROR","OVERLAP","FROZEN","UNREACHABLE",
  "BOUNDARIES","MULTIPLE","NONFUNCTIONAL","REJECTED","EXPERTLIMIT",
  "BADNEIGHBOR","MISSINGSPOT","BADPERFORMANCE","RELAYOFF","reserved31"]


def flagToString(value, mask=0, separator='|'):
    """Convert a flag bitmask into a string listing the corresponding flag names.
    """
    value = value & ~mask
    bits = [flagNames[k] for k in range(32) if (value & (1<<k))]
    return separator.join(bits)

def stringToFlag(string, separator='|'):
    """Convert a string listing flag names into a corresponding bitmask value.
    """
    mask = 0
    for name in string.split(separator):
        if name not in flagNames:
            raise ValueError(f'Invalid bit name: {name}.')
        bit = flagNames.index(name)
        mask |= (1 << bit)
    return mask
