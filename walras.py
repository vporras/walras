import numpy as np
from itertools import product
import random
import matplotlib.pyplot as plt
import argparse
from enum import Enum

# TODO pick good defaults here
BUY_CONSTRAINT = 10
SELL_CONSTRAINT = 0.1

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

def gmean(x, y):
    return (x * y) ** 0.5

class Trader():
    """Trader class"""

    def __init__(self, name, preference, endowment):
        self.name = name
        self.preference = preference
        self.endowment = endowment
        self.allocs = [endowment]
        self.alloc = endowment
        self.buy_constraint  = BUY_CONSTRAINT
        self.sell_constraint = SELL_CONSTRAINT
        self.last_trade_mrs = None

        # these include constraints, du and dw from their creation
        self.old_buy = None
        self.old_sell = None

    def reset(self):
        self.update_constraints()

        self.last_trade_mrs = None
        self.allocs = [self.endowment]
        self.alloc = self.endowment

    # should_tighten is true if a net buyer and this is a buy constraint or vice versa
    def pick_constraint(self, constraint, old, du, dw, should_tighten):
        if old != None:
            old_constraint, odu, odw = old
            if constraint != old_constraint:
                if du < odu or (du == odu and dw < odw):
                    return (gmean(constraint, old_constraint), (old_constraint, odu, odw))
                else:
                    return (constraint, (constraint, du, dw))
            
        if dw < 0 and should_tighten:
            # return (self.last_trade_mrs, (constraint, du, dw))
            return (gmean(constraint, self.last_trade_mrs), (constraint, du, dw))

        else:
            return (constraint, (constraint, du, dw))

    def update_constraints(self):
        dw = self.d_wealth()
        du = self.d_utility()
        self.buy_constraint, self.old_buy = self.pick_constraint(
            self.buy_constraint, self.old_buy, du, dw, self.alloc[0] > self.endowment[0])
        self.sell_constraint, self.old_sell = self.pick_constraint(
            self.sell_constraint, self.old_sell, du, dw, self.alloc[0] < self.endowment[0])
        
    def utility(self, alloc):
        alpha = self.preference
        x1, x2 = alloc
        if x1 < 0 or x2 < 0:
            return 0
        return x1**alpha * x2**(1-alpha)

    def d_utility(self):
        return self.utility(self.alloc) - self.utility(self.endowment)

    def wealth(self, alloc):
        x1, x2 = alloc
        return x1 + x2 * self.last_trade_mrs
        
    # returns the total change in wealth this round, in terms of x1
    def d_wealth(self):
        if self.last_trade_mrs != None:
            return self.wealth(self.alloc) - self.wealth(self.endowment)
        else:
            return 0

    def mrs(self, dir):
        # This is the exchange rate between x1 and x2, in terms of x1.
        # eg if mrs(Dir.buy) == 5.0, then the trader is willing to pay up to 5 units
        #   of x2 in order to get one unit of x1
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
           return Dir.none

    def joint_mrs(self, other, dir):
        if dir == Dir.none:
            return 1
        # convergence must better than using self.mrs
        return gmean(self.mrs(dir), other.mrs(dir.inv()))
        # return random.uniform(self.mrs(dir), other.mrs(dir.inv()))
        # return random.uniform(self.buy_constraint, self.sell_constraint)

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

        if abs(size) > 0:
            self.change_alloc(self.new_alloc(size, joint_mrs))
            other.change_alloc(other.new_alloc(-size, joint_mrs))
            self.last_trade_mrs = joint_mrs
            other.last_trade_mrs = joint_mrs
            
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
    
def do_round(config, traders, is_last):
    trades = []

    # wait for until the last finish-count trades have been 0-size
    while (len(trades) < config.finish_count or sum(trades[-config.finish_count:]) > 0):
        trade = trade_random(traders, config.min_size, config.dynamic)
        trades.append(abs(trade.size))
        if args.verbose and abs(trade.size) > 0:
            print(trade)

    b = config.buckets
    bsz = len(trades) // b
    smoothed = [sum(trades[i*bsz:(i+1)*bsz]) for i in range(b)]

    def pretty(l):
        return " ".join(["%.3f" % x for x in l])

    mrss = [t.mrs(Dir.buy) for t in traders]
    dw = [t.d_wealth() for t in traders]
    du = [t.d_utility() for t in traders]
    C = np.log(BUY_CONSTRAINT) - np.log(SELL_CONSTRAINT)
    c = [(np.log(t.buy_constraint) - np.log(t.sell_constraint)) / C for t in traders]
    print("W: %.3f U: %.3f M: %.3f C: %.3f" %
          (sum(np.abs(dw)), sum(du), np.std(mrss), np.mean(c)))

    if is_last:
        print("mrss:", pretty(mrss))
        print("dw:  ", pretty(dw))
        print("du:  ", pretty(du))

    if config.plot and is_last or config.plot_all:
        plt.figure("allocations over time", figsize=(10, 12))

        # only plot first 4 traders due to screen size
        n_to_plot = min(len(traders), 4)
        for i, t in enumerate(traders[:n_to_plot]):
            t.plot(n_to_plot, i + 1)
        plt.subplots_adjust(hspace=.3, bottom=.05, top=.95)
        plt.show()

    if config.convergence and is_last:
        plt.figure("sum of trade sizes")
        plt.plot(smoothed)
        plt.show()

def run(config):
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
        
    for i in range(config.rounds):
        do_round(config, traders, i == config.rounds - 1)
    
        for t in traders:
            t.reset()


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
    parser.add_argument("-p", "--plot", action="store_true", help="plot the allocations")
    parser.add_argument("-P", "--plot-all", action="store_true", help="plot the allocations every round ")
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
