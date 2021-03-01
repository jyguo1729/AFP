from math import log10, floor
def round_to_4(x):
    return round(x, 3-int(floor(log10(abs(x)))))