from .Data.helpers import convert_to_list

def compose(x, funcs, *args, order_key='_order', **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(convert_to_list(funcs), key=key): x = f(x, **kwargs)
    return x

