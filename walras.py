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
        if self == Dir.buy:
            return Dir.sell
        elif self == Dir.sell:
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

    def utility(self, alloc):
        alpha = self.preference
        if alloc[0] < 0 or alloc[1] < 0:
            return 0
        return alloc[0]**alpha + alloc[1]**(1-alpha)

    def mrs(self, dir):
        # TODO: explain the units used here
        alpha = self.preference
        x1, x2 = self.alloc
        if x2 == 0:
            return self.buy_constraint
        mrs = alpha*x2/((1-alpha)*x1)
        # mrs = (1 - alpha)*x2 / (alpha*x1)
        # mrs = (1-alpha)*x1 / (alpha*x2)
        if dir == Dir.buy:
            return min(mrs, self.buy_constraint)
        elif dir == Dir.sell:
            return max(mrs, self.sell_constraint)
        else:
            # should be unreachable!
            assert(False)
    
    def get_dir(self, other):
       if self.mrs(Dir.buy) > other.mrs(Dir.sell):
           return Dir.buy
       elif self.mrs(Dir.sell) < other.mrs(Dir.buy):
           return Dir.sell
       else:
           print("constrained")
           return Dir.none

    def joint_mrs(self, other, dir):
        if dir == Dir.none:
            return 1
        return (self.mrs(dir) * other.mrs(dir.inv())) ** 0.5
        # return random.uniform(self.mrs(dir), other.mrs(dir.inv()))

    def new_alloc(self, size, joint_mrs):
        # positive size is potentially buying good 1
        return (self.alloc[0] + size, self.alloc[1] - size / joint_mrs)

    def is_plus_u(self, other, size, joint_mrs):
        u1_0 = self.utility(self.alloc)
        u1_1 = self.utility(self.new_alloc(size, joint_mrs))
        u2_0 = other.utility(other.alloc)
        u2_1 = other.utility(other.new_alloc(-size, joint_mrs))

        return u1_1 > u1_0 and u2_1 > u2_0

    def get_size(self, other, dir, joint_mrs, min_size):
        size = 0
        if dir == Dir.buy:
            size = min_size
        elif dir == Dir.sell:
            size = -min_size

        if self.is_plus_u(other, size, joint_mrs):
            return size
        else:
            return 0

    def change_alloc(self, new_alloc):
        self.alloc = new_alloc
        self.allocs.append(self.alloc)

    def trade(self, other, abs_size):
        dir = self.get_dir(other) 
        joint_mrs = self.joint_mrs(other, dir)
        size = self.get_size(other, dir, joint_mrs, abs_size)
        self.change_alloc(self.new_alloc(size, joint_mrs))
        other.change_alloc(other.new_alloc(-size, joint_mrs))
        return Trade(self, other, size, joint_mrs)

    def plot(self, rows, index):
        alpha = self.preference
        xlist = np.linspace(0, 10.0, 100)
        ylist = np.linspace(0, 10.0, 100)
        X1, X2 = np.meshgrid(xlist, ylist)
        # TODO: figure out why this doesn't work
        #U = self.utility((X1, X2))
        U = X1**alpha + X2**(1-alpha)
        plt.subplot(rows, 1, index)
        cp = plt.contour(X1, X2, U, 25, colors="g")
        plt.clabel(cp, inline=True, fontsize=10)
        plt.title("trader: %s, alpha = %f" % (self.name, alpha))
        plt.xlabel("Good 1")
        plt.ylabel("Good 2")

        x1s, x2s = zip(*self.allocs)
        plt.plot(x1s, x2s, ".-")
        x1, x2 = self.alloc
        plt.plot([x1], [x2], "ro")

        buyX = np.linspace(x1, x1 + 1, 5)
        buyY = x2 + (buyX - x1) * -self.mrs(Dir.buy)  
        plt.plot(buyX, buyY, "--b")

        sellX = np.linspace(x1 - 1, x1, 5)
        sellY = x2 + (sellX - x1) * -self.mrs(Dir.sell)  
        plt.plot(sellX, sellY, "--r")

    def __str__(self):
        return ("Name:{name},\n alpha=({preference}),\n alloc=({alloc})".format(
            name = self.name,
            preference = self.preference,
            alloc = self.alloc))

def trade_random(traders, min_size):
    a,b = random.sample(traders,2)
    return(a.trade(b, min_size))

def random_traders(n):
    return [Trader(i, random.random(), (random.randint(1,10),random.randint(1,10)))
            for i in range(n)]

def run(config):
    trades = []
    traders = random_traders(config.num_traders)
    # wait for finish-count 0-size trades
    while (len(trades) < config.finish_count or sum(trades[-config.finish_count:]) > 0):
        trade = trade_random(traders, config.min_size)
        trades.append(abs(trade.size))
        if args.verbose and abs(trade.size) > 0:
            print(trade)
    bucket_size = 25 
    smoothed = zip(*[trades[n:] for n in range(config.bucket_size)])
    smoothed = [sum(a) for a in smoothed]

    mrss = [t.mrs(Dir.buy) for t in traders]
    print(mrss)

    plt.figure(figsize=(10, 12))
    
    if config.plot:
        for i, t in enumerate(traders):
            t.plot(config.num_traders, i + 1)
        plt.subplots_adjust(hspace=.3, bottom=.05, top=.95)
        plt.show()

    # plt.plot(smoothed)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-traders", type=int)
    parser.add_argument("-r", "--rounds", type=int, default=1)
    parser.add_argument("-m", "--min-size", type=float, default=0.1,
                        help="minimum size of trade (default: 0.1)")
    parser.add_argument("-f", "--finish-count", type=int, default=10,
                        help="number of empty trades to finish a round (default: 10)")
    parser.add_argument("-s", "--seed", default=str(random.randint(0, 10000)),
                        help="seed to initialize PRNG")
    parser.add_argument("-p", "--plot", action="store_true", help="plot the traders")
    parser.add_argument("-v", "--verbose", action="store_true", help="print each trade")
    parser.add_argument("-b", "--bucket-size", type=int, default=25,
                        help="bucket size for convergence graph smoothing")
    args = parser.parse_args()
    random.seed(args.seed)
    print("seed: " + args.seed)
    args
    print(args)
    run(args)


