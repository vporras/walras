import numpy as np
from itertools import product
from random import sample, randint, random
import matplotlib.pyplot as plt
import argparse
from enum import Enum

class Trade():
    
    def __init__(self, source, origin, size, joint_mrs):
        self.source = source
        self.origin = origin
        self.size = size
        self.joint_mrs = joint_mrs

    def __str__(self):
        return ("From {src},\n To {origin},\n with size {size},\n at {mrs}".format(
            src = self.source,
            origin = self.origin,
            size = self.size,
            mrs = self.joint_mrs))

class Dir(Enum):
    """Direction of trade in terms of good 1"""
    buy = 1
    sell = -1
    
class Trader():
    """Trader class"""

    def __init__(self, name, preference, allocation):
        self.name = name
        self.preference = preference
        self.allocs = [allocation]
        self.alloc = allocation

    def mrs(self):
        alpha = self.preference
        x1, x2 = self.alloc
        if x1 == 0:
            return 10000
        return abs(-alpha*x2/((1-alpha)*x1))

    def plot(self):
        alpha = self.preference
        xlist = np.linspace(0, 10.0, 100)
        ylist = np.linspace(0, 10.0, 100)
        X1, X2 = np.meshgrid(xlist, ylist)
        U = X1**alpha + X2**(1-alpha)
        plt.figure()
        cp = plt.contour(X1, X2, U)
        plt.clabel(cp, inline=True, fontsize=10)
        plt.title("alpha = %f" % alpha)
        plt.xlabel("Good 1")
        plt.ylabel("Good 2")

        x1, x2 = self.alloc
        plt.plot([x1], [x2], "o", label="current")
        x1s, x2s = zip(*self.allocs)
        plt.plot(x1s, x2s, ".-")

        plt.show()

    def change_allocation(self, delta):
        self.alloc = (self.alloc[0] + delta[0],
                              self.alloc[1] + delta[1])
        self.allocs.append(self.alloc)

    def joint_mrs(self, other):
        return (self.mrs() * other.mrs()) ** 0.5

    def get_size(self, other, joint_mrs, abs_size):
       # We should change this to used a closed form calculation of the maximum size
       # If it's fixed, it needs to be scaled according to the good
       if self.mrs() > other.mrs():
           return min(abs_size, joint_mrs * self.alloc[1], other.alloc[0])
       elif self.mrs() < other.mrs():
           return -min(abs_size, joint_mrs * other.alloc[1], self.alloc[0])
       else:
           return 0

    def trade(self, other, abs_size):
        joint_mrs = self.joint_mrs(other)
        size = self.get_size(other, joint_mrs, abs_size)
        self.change_allocation((size, -size / joint_mrs))
        other.change_allocation((-size, size / joint_mrs))
        return Trade(self, other, size, joint_mrs)

    def __str__(self):
        return ("Name:{name},\n alpha=({preference}),\n alloc=({alloc})".format(
            name = self.name,
            preference = self.preference,
            alloc = self.alloc))

def trade_random(traders, trade_size):
    a,b = sample(traders,2)
    return(a.trade(b, trade_size))

def random_traders(n):
    return [Trader(i, random(), (randint(1,10),randint(1,10)))
            for i in range(n)]

def mrs_range(traders):
    # currently centralised, not distributed
    # could be calculated incrementally
    mrss = [t.mrs() for t in traders]
    return max(mrss) - min(mrss)

def run(num_traders, trade_size):
    sizes = []
    traders = random_traders(num_traders)
    while (mrs_range(traders) > 0.01):
        trade = trade_random(traders, trade_size)
        sizes.append(abs(trade.size))
    sizes = [[0,1][a>0.1] for a in sizes]
    bucket_size = 25 
    sizes = zip(*[sizes[n:] for n in range(bucket_size)])
    sizes = [sum(a) for a in sizes]

    mrss = [t.mrs() for t in traders]
    print(mrss)
    
    for t in traders:
        t.plot()

             
    # plt.plot(sizes)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_traders", type=int)
    parser.add_argument("-s", "--trade_size", type=float, default=1.0,
                        help="size of trade (default: 1.0)")
    args = parser.parse_args()
    run(args.num_traders, args.trade_size)


