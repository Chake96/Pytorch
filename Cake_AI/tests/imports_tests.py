from ..utilities import *
import operator

def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def test_eq(a,b): test(a,b,operator.eq,'==')

test_eq(convert_to_set('aa'), {'aa'})
test_eq(convert_to_set(['aa',1]), {'aa',1})
test_eq(convert_to_set(None), set())
test_eq(convert_to_set(1), {1})
test_eq(convert_to_set({1}), {1})
