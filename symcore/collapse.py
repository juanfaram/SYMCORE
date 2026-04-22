import math

def collapse_window(window, sym_type, params):
    k = window.shape[0]
    if sym_type == 'mirror':
        base_len = math.ceil(k / 2)
        return window[:base_len], {}
    elif sym_type == 'periodic':
        p = params['period']
        return window[:p], params
    elif sym_type == 'scale':
        half = k // 2
        return window[:half], params
    else:
        raise ValueError(f"Unknown symmetry type: {sym_type}")