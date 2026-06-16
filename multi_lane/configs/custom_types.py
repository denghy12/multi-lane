import argparse

def restricted_float(x, range=(0.0, 1.0)):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < range[0] or x > range[1]:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x

def non_negative_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
    
    if x < 0:
        raise argparse.ArgumentTypeError("%r is negative" % (x,))
    return x

def str2bool(x):
    if isinstance(x, bool):
        return x
    value = str(x).lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError("%r is not a valid boolean value" % (x,))
