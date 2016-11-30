import numpy as np
from itertools import product
import random
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
    none = 0 

    def inv(self):
        if (self == Dir.buy):
            return Dir.sell
        elif (self == Dir.sell):
            return Dir.buy
        else:
            return Dir.none

class Trader():
    """Trader class"""

    def __init__(self, name, preference, alloc):
        self.name = name
        self.preference = preference
        self.allocs = [alloc]
        self.alloc = alloc
        self.buy_constraint  = 10000
        self.sell_constraint = 0.0001

    def mrs(self, dir):
        # TODO: explain the units used here
        alpha = self.preference
        x1, x2 = self.alloc
        if x1 == 0:
            return self.buy_constraint
        mrs = alpha*x2/((1-alpha)*x1)
        if (dir == Dir.buy):
            return min(mrs, self.buy_constraint)
        elif (dir == Dir.sell):
            return max(mrs, self.sell_constraint)
        else:
            # should be unreachable!
            assert(False)
    
    def get_dir(self, other, threshold):
       if self.mrs(Dir.buy) > other.mrs(Dir.sell) + threshold:
           return Dir.buy
       elif self.mrs(Dir.sell) < other.mrs(Dir.buy) - threshold:
           return Dir.sell
       else:
           return Dir.none

    def joint_mrs(self, other, dir):
        if (dir == Dir.none):
            return 1
        return (self.mrs(dir) * other.mrs(dir.inv())) ** 0.5

    def get_size(self, other, dir, joint_mrs, abs_size):
       # We should change this to used a closed form calculation of the maximum size
       # If it's fixed, it needs to be scaled according to the good
       if dir == Dir.buy:
           return min(abs_size, joint_mrs * self.alloc[1], other.alloc[0])
       elif dir == Dir.sell:
           return -min(abs_size, joint_mrs * other.alloc[1], self.alloc[0])
       else:
           return 0

    def change_alloc(self, delta):
        self.alloc = (self.alloc[0] + delta[0], self.alloc[1] + delta[1])
        self.allocs.append(self.alloc)

    def trade(self, other, abs_size, threshold):
        dir = self.get_dir(other, threshold) 
        joint_mrs = self.joint_mrs(other, dir)
        size = self.get_size(other, dir, joint_mrs, abs_size)
        self.change_alloc((size, -size / joint_mrs))
        other.change_alloc((-size, size / joint_mrs))
        return Trade(self, other, size, joint_mrs)

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

        x1s, x2s = zip(*self.allocs)
        plt.plot(x1s, x2s, ".-")
        x1, x2 = self.alloc
        plt.plot([x1], [x2], "ro")

        plt.show()

    def __str__(self):
        return ("Name:{name},\n alpha=({preference}),\n alloc=({alloc})".format(
            name = self.name,
            preference = self.preference,
            alloc = self.alloc))

def trade_random(traders, trade_size, threshold):
    a,b = random.sample(traders,2)
    return(a.trade(b, trade_size, threshold))

def random_traders(n):
    return [Trader(i, random.random(), (random.randint(1,10),random.randint(1,10)))
            for i in range(n)]

def run(config):
    trades = []
    traders = random_traders(config.num_traders)
    # wait for finish-count 0-size trades
    while (len(trades) < config.finish_count or sum(trades[-config.finish_count:]) > 0):
        trade = trade_random(traders, config.trade_size, config.threshold)
        trades.append(abs(trade.size))
    bucket_size = 25 
    smoothed = zip(*[trades[n:] for n in range(config.bucket_size)])
    smoothed = [sum(a) for a in smoothed]

    mrss = [t.mrs(Dir.buy) for t in traders]
    print(mrss)
    
    if (config.plot):
        for t in traders:
            t.plot()

             
    # plt.plot(smoothed)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-traders", type=int)
    parser.add_argument("-r", "--rounds", type=int, default=1)
    parser.add_argument("-s", "--trade-size", type=float, default=1.0,
                        help="size of trade (default: 1.0)")
    parser.add_argument("-t", "--threshold", type=float, default=0.01,
                        help="minimum difference in MRS to do a trade (default: 0.01)")
    parser.add_argument("-f", "--finish-count", type=int, default=10,
                        help="number of empty trades to finish a round (default: 10)")
    parser.add_argument("--seed", default=str(random.randint(0, 10000)),
                        help="seed to initialize PRNG")
    parser.add_argument("-p", "--plot", action="store_true", help="plot the traders")
    parser.add_argument("-b", "--bucket-size", type=int, default=25,
                        help="bucket size for convergence graph smoothing")
    args = parser.parse_args()
    random.seed(args.seed)
    print("seed: " + args.seed)
    args
    print(args)
    run(args)


