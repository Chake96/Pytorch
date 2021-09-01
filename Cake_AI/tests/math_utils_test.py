from math_utils import *

a = [1,2,3,4,5,6,7,8,9,10]
b = [100,2000,2500,10000,50]

if __name__ == "main":
    assert variance(a) == 8.25
    assert variance(b) == 13467600
    assert variance(1000000) == 0

    