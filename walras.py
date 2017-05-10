import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import sys
import os
import json
import multiprocessing as mp
from time import time
from enum import Enum
from operator import itemgetter

# TODO pick good defaults here
BUY_CONSTRAINT = 10
SELL_CONSTRAINT = 0.1

bench_time = 0

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

    def __init__(self, name, preference, endowment, config):
        self.name = name
        self.preference = preference
        self.endowment = endowment
        self.allocs = [endowment]
        self.alloc = endowment
        self.last_trade_mrs = None
        self.round = 0

        # these lists store histories of constraints by round
        self.buy_constraints  = [BUY_CONSTRAINT]
        self.sell_constraints = [SELL_CONSTRAINT]
        self.du = []
        self.dw = []
        # these store the round with which the current constraint is compared
        self.buy_lookbacks = [0]
        self.sell_lookbacks = [0]

        self.constraint_mode = config.constraint_mode
        self.reversion = config.reversion
        self.constraint_factor = config.constraint_factor
        self.backtracks = config.backtracks
        self.backtrack_prob = config.backtrack_prob
        self.backtrack_threshold = config.backtrack_threshold

    def reset(self):
        self.update_constraints()

        self.round = self.round + 1
        self.last_trade_mrs = None
        self.allocs = [self.endowment]
        self.alloc = self.endowment
        
    def pick_constraint(self, constraints, lookbacks, dir, net_buyer):
        cur = self.round
        du = self.du
        dw = self.dw
        th = self.backtrack_threshold 
        # Check [backtrack] rounds in the past and revert if utility has fallen
        for bt in self.backtracks:
            if cur >= bt:
                if du[cur] < du[cur - bt] * th and random.random() < self.backtrack_prob:
                    constraints.append(constraints[cur - bt])
                    lookbacks.append(lookbacks[cur - bt])
                    return

        # Checking whether improvement was successful, and maybe reverting
        if cur > 0:
            old = lookbacks[cur]
            if constraints[cur] != constraints[old]:
                if du[cur] < du[old] or (du[cur] == du[old] and dw[cur] < dw[old]):
                    if self.reversion == "mean":
                        constraints.append(gmean(constraints[cur], constraints[old]))
                        lookbacks.append(old)
                    elif self.reversion == "total":
                        constraints.append(constraints[old])
                        lookbacks.append(lookbacks[old])
                    # elif self.reversion == "random":
                    else:
                        r = random.randint(0, old)
                        constraints.append(constraints[r])
                        lookbacks.append(lookbacks[r])
                # success!
                else:
                    constraints.append(constraints[cur])
                    lookbacks.append(cur)
                return
            
        # Changing constraints if I lost wealth
        if dw[cur] < 0 and ((net_buyer and dir == Dir.buy) or (not net_buyer and dir == Dir.sell)):
            lookbacks.append(cur)
            if self.constraint_mode == "last":
                constraints.append(self.last_trade_mrs)
            elif self.constraint_mode == "mean":
                constraints.append(gmean(constraints[cur], self.last_trade_mrs))
            # elif self.constraint_mode == "fixed":
            else:
                factor = 1 - self.constraint_factor if net_buyer else 1 + self.constraint_factor
                constraints.append(constraints[cur] * factor)
        # Don't change anything
        else:
            lookbacks.append(lookbacks[cur])
            constraints.append(constraints[cur])

    def update_constraints(self):
        self.dw.append(self.d_wealth())
        self.du.append(self.d_utility())
        # TODO: check how many buys and sells per trader
        net_buyer = self.alloc[0] > self.endowment[0]
        self.pick_constraint(self.buy_constraints, self.buy_lookbacks, Dir.buy, net_buyer)
        self.pick_constraint(self.sell_constraints, self.sell_lookbacks, Dir.sell, net_buyer)
        
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
        if self.last_trade_mrs != None:
            return x1 + x2 * self.last_trade_mrs
        else:
            # TODO: is this the right fallback?
            return x1 + x2
        
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
            return self.buy_constraints[self.round]
        mrs = alpha*x2/((1-alpha)*x1)
        if dir == Dir.buy:
            return min(mrs, self.buy_constraints[self.round])
        #elif dir == Dir.sell:
        else:
            return max(mrs, self.sell_constraints[self.round])
    
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
        return gmean(self.mrs(dir), other.mrs(dir.inv()))
        # return random.uniform(self.mrs(dir), other.mrs(dir.inv()))
        # return random.uniform(self.buy_constraints[self.round], self.sell_constraint[self.round])

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
        global bench_time
        dir = self.get_dir(other) 
        start = time()
        joint_mrs = self.joint_mrs(other, dir)
        bench_time += time() - start
        size = self.get_size(other, dir, joint_mrs, min_size, dynamic)

        if size != 0:
            self.change_alloc(self.new_alloc(size, joint_mrs))
            other.change_alloc(other.new_alloc(-size, joint_mrs))
            self.last_trade_mrs = joint_mrs
            other.last_trade_mrs = joint_mrs
            
        return size

    # No access to derivative, so can't guess direction
    def dumb_trade(self, other, min_size, dynamic):
        dir = Dir.buy
        # Pick a random price
        # TODO: truncated normal?
        joint_mrs = random.uniform(other.sell_constraints[self.round], self.buy_constraints[self.round])
        size = self.get_size(other, Dir.buy, joint_mrs, min_size, dynamic)

        if size != 0:
            self.change_alloc(self.new_alloc(size, joint_mrs))
            other.change_alloc(other.new_alloc(-size, joint_mrs))
            self.last_trade_mrs = joint_mrs
            other.last_trade_mrs = joint_mrs
            
        return size
        
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

def trade_random(traders, min_size, dynamic, utility_only):
    a,b = random.sample(traders,2)
    if utility_only:
        return(a.dumb_trade(b, min_size, dynamic))
    else:
        return(a.trade(b, min_size, dynamic))

def do_round(config, traders, round):
    # wait for until the last [finish_count] trades have been 0-size
    consecutive_zero_trades = 0
    total_trades = 0
    while consecutive_zero_trades < config.finish_count:
        size = trade_random(traders, config.min_size, config.dynamic, config.utility_only)
        total_trades += 1
        if size == 0:
            consecutive_zero_trades += 1
        else:
            consecutive_zero_trades = 0

    total_wealth = sum([t.wealth(t.alloc) for t in traders])

    # mrss = [t.mrs(Dir.buy) for t in traders]
    mrss_raw = np.array([t.last_trade_mrs for t in traders])
    # filter out the ones without a last trade
    mrss = mrss_raw[mrss_raw != np.array(None)]
    dw = [t.d_wealth() / total_wealth for t in traders]
    du = [t.d_utility() for t in traders]
    C = np.log(BUY_CONSTRAINT) - np.log(SELL_CONSTRAINT)
    c = [(np.log(t.buy_constraints[t.round]) - np.log(t.sell_constraints[t.round])) / C * 2 - 1 for t in traders]
    # TODO: should this include empty trades?
    res = (round, sum(np.abs(dw)) / 2, sum(du), np.std(mrss), np.mean(c), total_trades/config.num_traders)
    if config.verbosity >= 3:
        print("%4d W: %.3f U: %.3f M: %.3f C: %.3f T: %2.2f" % res)

    is_last = round == config.rounds - 1
    if  config.verbosity == 4 and is_last:
        def pretty(l):
            return " ".join(["%.3f" % x for x in l if not x is None])
        
        print("mrss:", pretty(mrss))
        print("dw:  ", pretty(dw))
        print("du:  ", pretty(du))

    if config.plotting >= 2 and is_last or config.plotting >= 3:
        plt.figure("allocations over time", figsize=(10, 12))

        # only plot first 4 traders due to screen size
        n_to_plot = min(len(traders), 4)
        for i, t in enumerate(traders[:n_to_plot]):
            t.plot(n_to_plot, i + 1)
        plt.subplots_adjust(hspace=.3, bottom=.05, top=.95)
        plt.show()

    return res

def random_traders(n, config):
    return [Trader(i, random.random(), (random.random(),random.random()), config)
            for i in range(n)]

def write_log(config, elapsed, w, u, m, c):
    
    if config.log_path:
        obj = {}
        obj["command"] = " ".join(sys.argv)
        obj["config"] = vars(config)
        obj["elapsed"] = elapsed
        obj["wealths"] = w.tolist()
        obj["utilities"] = u.tolist()
        obj["mrs closeness"] = m.tolist()
        obj["constrainedness"] = c.tolist()

        i = 0
        while os.path.exists("%s/log%03d" % (config.log_path, i)):
            i += 1
        with open("%s/log%03d" % (config.log_path, i), "w") as f:
            json.dump(obj, f)


class TrialSummary():
    """Summary statistics for a trial"""

    class Stats(tuple):
        """Stats for summaries"""
        __slots__ = []
        def __new__(cls, w, u, m, c, s):
            return tuple.__new__(cls, (w, u, m, c, s))

        def __getnewargs__(self):
            return (self.wealth, self.utility, self.mrs_closeness, self.constrainedness, self.seconds)

        wealth          = property(itemgetter(0))
        utility         = property(itemgetter(1))
        mrs_closeness   = property(itemgetter(2))
        constrainedness = property(itemgetter(3))
        seconds         = property(itemgetter(4))


        # def __new__(cls, seconds, wealth, utility, mrs_closeness, constrainedness):

        @classmethod
        def from_idx(cls, idx, wealth, utility, mrs_closeness, constrainedness, seconds):
            return cls(wealth[idx], utility[idx], mrs_closeness[idx], constrainedness[idx], seconds[idx])

        def __add__(self, other):
            if other is None or other is 0:
                return self
            return TrialSummary.Stats(*[x + y for (x, y) in zip(self, other)])

        def __radd__(self, other):
            return self + other

        def __truediv__(self, other):
            return TrialSummary.Stats(*[x / other for x in self])

        def __str__(self):
            return ("W: %.3f U: %.3f M: %.3f C: %.3f S: %2.2f"
                    % (self.wealth, self.utility, self.mrs_closeness, self.constrainedness, self.seconds))

    def find_conv(self, config, w):
        bsz = config.bucket_size
        trailing = sum(w[0   : bsz])   / bsz
        leading  = sum(w[bsz : bsz*2]) / bsz
        var   = np.var(w[0   : bsz*2])
        conv_idx = -1
        for i in range(bsz, config.rounds - bsz): #TODO possibly +1 at end?

            var2 = np.var(w[i - bsz : i + bsz])
            print("%d %0.5f %0.5f %0.5f" % (i, abs(trailing - leading), var, var2))
            if abs(trailing - leading) < config.conv_threshold:
                conv_idx = i
                break
            dvar = w[i - bsz] - (trailing + leading)/2
            print((trailing + leading)/2)
            trailing += (w[i]       - w[i - bsz]) / bsz
            leading  += (w[i + bsz] - w[i]      ) / bsz
            print((trailing + leading)/2)
            var += (w[i + bsz] - w[i - bsz]) * (w[i + bsz] - (trailing + leading)/2 + dvar) / (bsz - 1)
            
        return conv_idx

    def __init__(self, config, wealth, utility, mrs_closeness, constrainedness, seconds):
        self.seed = config.seed
        self.end  = self.Stats.from_idx(config.rounds - 1, wealth, utility, mrs_closeness, constrainedness, seconds)
        self.div_idx = -1
        self.did_div = self.div_idx >= 0
        self.conv_idx = self.find_conv(config, wealth)
        self.did_conv = self.conv_idx >= 0 and not self.did_div or self.conv_idx < self.div_idx
        if self.did_conv:
            self.conv = self.Stats.from_idx(self.conv_idx, wealth, utility, mrs_closeness, constrainedness, seconds)
            

    def __str__(self):
        res = "seed: %d" % self.seed 
        res += "\nend: %s" % str(self.end)
        if self.did_conv:
            res += "\nconv idx: %d" % self.conv_idx
            res += "\nconv: %s" % str(self.conv)
        return res

def do_trial(config, results):
    random.seed(config.seed)

    traders = []
    if config.trader_file:
        with open(config.trader_file) as file:
            i = 0
            for line in file:
                strs = line.split()
                traders.append(Trader(i, float(strs[0]), (float(strs[1]), float(strs[2])), config)) 
                i += 1
        config.num_traders = len(traders)
    else:
        traders = random_traders(config.num_traders, config)

    wealths         = np.empty([config.rounds])
    utilities       = np.empty([config.rounds])
    mrs_closeness   = np.empty([config.rounds])
    constrainedness = np.empty([config.rounds])
    seconds         = np.empty([config.rounds])
        
    start_time = time()
    for i in range(config.rounds):
        _, w, u, m, c, t = do_round(config, traders, i)
        wealths[i] = w
        utilities[i] = u
        mrs_closeness[i] = m
        constrainedness[i] = c
        seconds[i] = time() - start_time
    
        for t in traders:
            t.reset()

    elapsed = time() - start_time
    global bench_time

    #if config.verbosity >= 2:
        # print("%3.2f seconds elapsed" % elapsed)
        # print("%3.2f bench time" % bench_time)

    write_log(config, elapsed, wealths, utilities, mrs_closeness, constrainedness) 

    summary = TrialSummary(config, wealths, utilities, mrs_closeness, constrainedness, seconds)
    if config.verbosity >= 2:
        print(summary)

    results.put(summary)

    if config.plotting >= 1 and config.rounds > 1:
        # normalization to make it fit on plot with other variables
        # utilities /= max(utilities)
        utilities /= config.num_traders / 5


        plt.figure("multiround data", figsize=(10,8))
        plt.plot(wealths,         label="wealth transfers (%)")
        plt.plot(utilities,       label="utility gains")
        plt.plot(mrs_closeness,   label="mrs closeness (stddev)")
        plt.plot(constrainedness, label="constrainedness %")

    
        ax = plt.gca()
        cur = ax.get_position()
        ax.set_position([cur.x0, cur.y0 + cur.height * 0.1, cur.width, cur.height * 0.9])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

        plt.show()

def run(config):
    results = mp.Queue()
    for i in range(0, config.trials):
        p = mp.Process(target=do_trial, args=(config, results))
        p.start()
        config.seed += 1

    data = []
    for i in range(0, config.trials):
        data.append(results.get())

    avg_end = sum([d.end for d in data]) / config.trials
    pct_conv = len([d for d in data if d.did_conv]) / config.trials * 100
    combined = "converged: %3.f%%\nend: %s" % (pct_conv, str(avg_end))

    i = 0
    while os.path.exists("results/%d_%d" % (config.experiment, i)):
        i += 1
    with open("results/%d_%d" % (config.experiment, i), "w") as f:
        print(combined, file=f)
        if config.verbosity >= 1:
            print(combined)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-traders", type=int, default=2)
    parser.add_argument("-r", "--rounds", type=int, default=1)
    parser.add_argument("-l", "--log-path", help="directory to save logs to")
    parser.add_argument("-s", "--seed", type=int, default=random.randint(0, 10000),
                        help="integer seed to initialize PRNG")
    parser.add_argument("-t", "--trials", type=int, default=1,
                        help="number of trials to test on, incrementing seed by 1 each time")
    parser.add_argument("-e", "--experiment", type=int, default=0,
                        help="id of experiment being run")

    # Intraday args
    parser.add_argument("--trader-file", help="file to load traders from")
    parser.add_argument("-m", "--min-size", type=float, default=0.01,
                        help="minimum size of trade (default: 0.01)")
    parser.add_argument("-d", "--dynamic", action="store_true", help="dynamic size (binary search)")
    parser.add_argument("--finish-count", type=int, default=25,
                        help="number of empty trades to finish a round (default: 25)")
    parser.add_argument("-u", "--utility-only", action="store_true", help="don't use derivatives of utility function")

    # Constraint args
    parser.add_argument("-c", "--constraint-mode", choices=["last", "mean", "fixed"], default="mean",
                        help="how new constraints are calculated. last is the last price, mean (default) is " \
                        "the mean of the last price and current constraint, " \
                        "fixed is a fixed percentage of the current constraint")
    parser.add_argument("--reversion", choices=["mean", "total", "random"], default="mean",
                        help="revert bad constraints to mean (default) of old and new constraint, totally " \
                        "to the old constraint, or randomly" )
    parser.add_argument("-f", "--constraint-factor", type=float, default=0.1,
                        help="factor used for fixed constraining")
    parser.add_argument("-b", "--backtracks", type=int, nargs='*', default=[],
                        help="revert constraint if utility dropped since [BACKTRACK] (default: [])")
    parser.add_argument("--backtrack-prob", type=float, default=0.50,
                        help="backtrack with [PROBABILITY] if utility has fallen (default: 0.5)")
    parser.add_argument("--backtrack-threshold", type=float, default=0.95,
                        help="backtrack if utility has fallen below [THRESHOLD] * previous (default: 0.95)")

    # Summary args
    parser.add_argument("--bucket-size", type=int, default=5,
                        help="number of rounds to average when calculating convergence and divergence (default: 5)")
    parser.add_argument("--conv-threshold", type=float, default=0.005,
                        help="register convergence if moving average wealths are within [THRESHOLD] (default: 0.005)")

    # Plotting args
    parser.add_argument("-p", "--plotting", type=int, default=1,
                        help="plotting level: 0 is none, 1 is summary chart, 2 is last round, 3 is every round")
    parser.add_argument("-v", "--verbosity", type=int, default=3,
                        help="verbosity level: 0 is nothing, 1 includes summary of all trials, 2 includes " \
                        "summary of each trial, 3 includes data after each round, 4 includes data from each trader")

    args = parser.parse_args()
    run(args)
