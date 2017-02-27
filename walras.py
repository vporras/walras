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

    def __init__(self, name, preference, endowment):
        self.name = name
        self.preference = preference
        self.allocs = [endowment]
        self.alloc = endowment
        self.buy_constraint  = 100
        self.sell_constraint = 0.01

    def utility(self, alloc):
        alpha = self.preference
        x1, x2 = alloc
        if x1 < 0 or x2 < 0:
            return 0
        return x1**alpha * x2**(1-alpha)

    def mrs(self, dir):
        # TODO: explain the units used here
        alpha = self.preference
        x1, x2 = self.alloc
        if x1 == 0:
            return self.buy_constraint
        mrs = alpha*x2/((1-alpha)*x1)
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
        # return random.choice([self.mrs(dir), other.mrs(dir.inv())])
        # return random.uniform(self.mrs(dir), other.mrs(dir.inv()))
        # return self.mrs(dir)

    def new_alloc(self, size, joint_mrs):
        # positive size is potentially buying good 1
        return (self.alloc[0] + size, self.alloc[1] - size * joint_mrs)

    def is_plus_u(self, other, size, joint_mrs):
        u1_0 = self.utility(self.alloc)
        u1_1 = self.utility(self.new_alloc(size, joint_mrs))
        u2_0 = other.utility(other.alloc)
        u2_1 = other.utility(other.new_alloc(-size, joint_mrs))
       
        return u1_1 > u1_0 and u2_1 > u2_0

    def get_size(self, other, dir, joint_mrs, min_size, dynamic):
        size = 0
        if dir == Dir.buy:
            size = min_size
        elif dir == Dir.sell:
            size = -min_size

        if self.is_plus_u(other, size, joint_mrs):
            # doubles size and recalculates
            if dynamic:
                return max(size, self.get_size(other, dir, joint_mrs, min_size*2, True))
            else:
                return size
        else:
            return 0

    def change_alloc(self, new_alloc):
        self.alloc = new_alloc
        self.allocs.append(self.alloc)

    def trade(self, other, min_size, dynamic):
        dir = self.get_dir(other) 
        joint_mrs = self.joint_mrs(other, dir)
        size = self.get_size(other, dir, joint_mrs, min_size, dynamic)
        self.change_alloc(self.new_alloc(size, joint_mrs))
        other.change_alloc(other.new_alloc(-size, joint_mrs))
        return Trade(self, other, size, joint_mrs) 
    
    def plot(self, rows, index):
        alpha = self.preference
        xlist = np.linspace(0, 1.0, 100)
        ylist = np.linspace(0, 1.0, 100)
        X1, X2 = np.meshgrid(xlist, ylist)
        U = np.array([[self.utility((x, y)) for x in xlist] for y in ylist])

        plt.subplot(rows, 1, index)
        cp = plt.contour(X1, X2, U, colors="g")
        plt.clabel(cp, inline=True, fontsize=10)
        plt.title("trader: %s, alpha = %f" % (self.name, alpha))
        plt.xlabel("Good 1")
        plt.ylabel("Good 2")
        
        x1s, x2s = zip(*self.allocs)
        plt.plot(x1s, x2s, ".-")
        x1, x2 = self.alloc
        plt.plot([x1], [x2], "ro")

        buyX = np.linspace(x1, x1 + .1, 5)
        buyY = x2 + (buyX - x1) * -self.mrs(Dir.buy)  
        plt.plot(buyX, buyY, "--b")

        sellX = np.linspace(x1 - .1, x1, 5)
        sellY = x2 + (sellX - x1) * -self.mrs(Dir.sell)  
        plt.plot(sellX, sellY, "--r")

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
    def plot_inverse(self):
        alpha = self.preference
        xlist = np.linspace(0, 1.0, 100)
        ylist = np.linspace(0, 1.0, 100)
        X1, X2 = np.meshgrid(xlist, ylist)
        X1 = 1 - X1
        X2 = 1 - X2
        U = np.array([[self.utility((x, y)) for x in xlist] for y in ylist])

        cp = plt.contour(X1, X2, U, colors="r")
        plt.clabel(cp, inline=True, fontsize=10)
        
        x1s, x2s = zip(*self.allocs)
        plt.plot(x2s, x1s, ".-")

        x1, x2 = self.alloc
        plt.plot([x2], [x1], "bo")

        buyX = np.linspace(x1, x1 + .1, 5)
        buyY = x2 + (buyX - x1) * -self.mrs(Dir.buy)  
        plt.plot(buyX, buyY, "--b")

        sellX = np.linspace(x1 - .1, x1, 5)
        sellY = x2 + (sellX - x1) * -self.mrs(Dir.sell)  
        plt.plot(sellX, sellY, "--r")

    def __str__(self):
        return ("Name:{name},\n alpha=({preference}),\n alloc=({alloc})".format(
            name = self.name,
            preference = self.preference,
            alloc = self.alloc))

def trade_random(traders, min_size, dynamic):
    a,b = random.sample(traders,2)
    return(a.trade(b, min_size, dynamic))

def random_traders(n):
    return [Trader(i, random.random(), (random.random(),random.random()))
            for i in range(n)]

def run(config):
    trades = []
    traders = []
    if config.traders and len(config.traders) % 3 == 0:
        config.num_traders = len(config.traders) // 3
        traders = (
            [Trader(i,
                    config.traders[i*3],
                    (config.traders[i*3 + 1],
                     config.traders[i*3 + 2]))
             for i in range(config.num_traders)]
        )
    else:
        traders = random_traders(config.num_traders)
    # wait for until the last finish-count trades have been 0-size
    while (len(trades) < config.finish_count or sum(trades[-config.finish_count:]) > 0):
        trade = trade_random(traders, config.min_size, config.dynamic)
        trades.append(abs(trade.size))
        if args.verbose:# and abs(trade.size) > 0:
            print(trade)
    b = config.buckets
    bsz = len(trades) // b
    smoothed = [sum(trades[i*bsz:(i+1)*bsz]) for i in range(b)]

    mrss = [t.mrs(Dir.buy) for t in traders]
    print(mrss)

    if config.plot:
        plt.figure("allocations over time", figsize=(10, 12))

        # traders[0].plot(config.num_traders, 1)
        # traders[1].plot_inverse()

        # only plot first 4
        n_to_plot = min(len(traders), 4)
        for i, t in enumerate(traders[:n_to_plot]):
            t.plot(n_to_plot, i + 1)
        plt.subplots_adjust(hspace=.3, bottom=.05, top=.95)
        plt.show()

    if config.convergence:
        plt.figure("sum of trade sizes")
        plt.plot(smoothed)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-traders", type=int, default=2)
    parser.add_argument("-t", "--traders", nargs="+", type=float,
                        help="traders as triples ALPHA X1 X2")
    parser.add_argument("-r", "--rounds", type=int, default=1)
    parser.add_argument("-m", "--min-size", type=float, default=0.01,
                        help="minimum size of trade (default: 0.01)")
    parser.add_argument("-f", "--finish-count", type=int, default=25,
                        help="number of empty trades to finish a round (default: 25)")
    parser.add_argument("-s", "--seed", default=str(random.randint(0, 10000)),
                        help="seed to initialize PRNG")
    parser.add_argument("-p", "--plot", action="store_true", help="plot the traders")
    parser.add_argument("-v", "--verbose", action="store_true", help="print each trade")
    parser.add_argument("-b", "--buckets", type=int, default=25,
                        help="number of buckets for convergence graph smoothing (default: 25)")
    parser.add_argument("-c", "--convergence", action="store_true", help="plot convergence")
    parser.add_argument("-d", "--dynamic", action="store_true", help="dynamic size (binary search)")
    args = parser.parse_args()
    random.seed(args.seed)
    print("seed: " + args.seed)
    args
    run(args)
