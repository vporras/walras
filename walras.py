import numpy as np
import random
import argparse
import sys
import os
import json
import multiprocessing as mp
import warnings
from time import time
from enum import Enum
from operator import itemgetter


# warnings.simplefilter("error")

BUY_CONSTRAINT = 10
SELL_CONSTRAINT = 0.1
DYNAMIC = True

##### Latex Plotting
import matplotlib as mpl
mpl.use('pgf')

def figsize(scale, ratio):
    fig_width_pt = 390                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27    
    
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*ratio              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 12,               # LaTeX default is 10pt font.
    "text.fontsize": 12,
    "legend.fontsize": 10,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
}
mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt

def savefig(filename):
    plt.savefig('{}.pgf'.format(filename), bbox_inches="tight")


# Simple plot
# fig, ax  = newfig(0.6)

# def ema(y, a):
#     s = []
#     s.append(y[0])
#     for t in range(1, len(y)):
#         s.append(a * y[t] + (1-a) * s[t-1])
#     return np.array(s)
    
# y = [0]*200
# y.extend([20]*(1000-len(y)))
# s = ema(y, 0.01)

# ax.plot(s)
# ax.set_xlabel('X Label')
# ax.set_ylabel('EMA')




#####

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
        self.round = 0
        
        if config.utility_type == "normal":
            self.utility = self.cd_utility
            self.trade = self.grad_trade
        elif config.utility_type == "leontief":
            self.utility = self.leontief_utility
            self.trade = self.nograd_trade
        else:
            self.utility = self.cd_utility
            self.trade = self.nograd_trade

        # these lists store histories of constraints by round
        self.buy_constraints  = [BUY_CONSTRAINT]
        self.sell_constraints = [SELL_CONSTRAINT]
        self.du = []
        self.dw = []
        # these store the round with which the current constraint is compared
        self.buy_lookbacks = [0]
        self.sell_lookbacks = [0]
        self.last_trade_mrs = None

        self.constraint_mode = config.constraint_mode
        self.reversion = config.reversion
        self.constraint_factor = config.constraint_factor
        self.backtracks = config.backtracks
        self.backtrack_prob = config.backtrack_prob
        self.backtrack_threshold = config.backtrack_threshold

    
    last_trade_mrs = None

    def reset(self):
        self.update_constraints()

        self.round = self.round + 1
        self.last_trade_mrs = None
        Trader.last_trade_mrs = None
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
        self.dw.append(self.d_wealth(self.last_trade_mrs))
        self.du.append(self.d_utility())
        # TODO: check how many buys and sells per trader
        net_buyer = self.alloc[0] > self.endowment[0]
        self.pick_constraint(self.buy_constraints, self.buy_lookbacks, Dir.buy, net_buyer)
        self.pick_constraint(self.sell_constraints, self.sell_lookbacks, Dir.sell, net_buyer)
        
    def cd_utility(self, alloc):
        alpha = self.preference
        x1, x2 = alloc
        if x1 < 0 or x2 < 0:
            return 0
        return x1**alpha * x2**(1-alpha)
        
    def leontief_utility(self, alloc):
        alpha = self.preference
        x1, x2 = alloc
        return min(x1, x2/alpha)

    def d_utility(self):
        return self.utility(self.alloc) - self.utility(self.endowment)

    def wealth(self, alloc, mrs):
        x1, x2 = alloc
        if mrs != None:
            return x1 + x2 * mrs
        else:
            # TODO: is this the right fallback?
            return x1 + x2
       
    # returns the total change in wealth this round, in terms of x1
    def d_wealth(self, mrs):
        if mrs != None:
            return self.wealth(self.alloc, mrs) - self.wealth(self.endowment, mrs)
        else:
            return 0

    def mrs(self, dir):
        # This is the exchange rate between x1 and x2, x2 per x1.
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

    def get_size(self, other, dir, joint_mrs, min_size):
        size = 0
        if dir == Dir.buy:
            size = min_size
        elif dir == Dir.sell:
            size = -min_size

        if self.is_plus_u(other, size, joint_mrs):
            # doubles size and recalculates
            if DYNAMIC:
                return max(size, self.get_size(other, dir, joint_mrs, min_size*2))
            else:
                return size
        else:
            return 0

    def change_alloc(self, new_alloc):
        self.alloc = new_alloc
        self.allocs.append(self.alloc)

    
    def grad_trade(self, other, min_size):
        dir = self.get_dir(other) 
        joint_mrs = self.joint_mrs(other, dir)
        size = self.get_size(other, dir, joint_mrs, min_size)

        if size != 0:
            self.change_alloc(self.new_alloc(size, joint_mrs))
            other.change_alloc(other.new_alloc(-size, joint_mrs))
            Trader.last_trade_mrs = joint_mrs
            self.last_trade_mrs = joint_mrs
            other.last_trade_mrs = joint_mrs
            
        return size

    # No access to gradient, so can't guess direction
    def nograd_trade(self, other, min_size):
        dir = Dir.buy
        # Pick a random price
        # TODO: truncated normal?
        joint_mrs = random.uniform(other.sell_constraints[self.round], self.buy_constraints[self.round])
        size = self.get_size(other, Dir.buy, joint_mrs, min_size)

        if size != 0:
            self.change_alloc(self.new_alloc(size, joint_mrs))
            other.change_alloc(other.new_alloc(-size, joint_mrs))
            Trader.last_trade_mrs = joint_mrs
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

def trade_random(traders, min_size):
    a,b = random.sample(traders,2)
    return(a.trade(b, min_size))

def do_round(config, traders, round):
    # wait for until the last [finish_count * num_traders] trades have been 0-size
    consecutive_zero_trades = 0
    total_trades = 0
    zero_trades = 0
    while consecutive_zero_trades < config.finish_count * config.num_traders:
        size = trade_random(traders, config.min_size)
        total_trades += 1
        if size == 0:
            consecutive_zero_trades += 1
            zero_trades += 1
        else:
            consecutive_zero_trades = 0

    total_wealth = sum([t.wealth(t.alloc, Trader.last_trade_mrs) for t in traders])

    # mrss = [t.mrs(Dir.buy) for t in traders]
    mrss_raw = np.array([t.last_trade_mrs for t in traders])
    # filter out the ones without a last trade
    mrss = mrss_raw[mrss_raw != np.array(None)]
    if len(mrss) == 0:
        mrss = [0]
    dw = [t.d_wealth(Trader.last_trade_mrs) / total_wealth for t in traders]
    starting_u = np.average([t.utility(t.endowment) for t in traders])
    du = [t.d_utility() for t in traders] / starting_u
    C = np.log(BUY_CONSTRAINT) - np.log(SELL_CONSTRAINT)
    c = [(np.log(t.buy_constraints[t.round]) - np.log(t.sell_constraints[t.round])) / C * 2 - 1 for t in traders]
    res = (round, sum(np.abs(dw)) / 2, np.average(du), np.std(mrss), np.mean(c), total_trades/config.num_traders)
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
        plt.figure("allocations (seed %d)" % config.seed, figsize=figsize(1.2, 1.0))

        # only plot first 4 traders due to screen size
        n_to_plot = min(len(traders), 4)
        for i, t in enumerate(traders[:n_to_plot]):
            t.plot(n_to_plot, i + 1)
        plt.subplots_adjust(hspace=.3, bottom=.05, top=.95)

        savefig("report/images/allocations_seed_%d" % config.seed)
        # plt.show()

    return res

def random_traders(n, config):
    return [Trader(i, random.random(), (random.random(),random.random()), config)
            for i in range(n)]

class TrialSummary():
    """Summary statistics for a trial"""

    class Stats(tuple):
        """Stats for summaries"""
        __slots__ = []
        def __new__(cls, w, u, m, c, s, t):
            return tuple.__new__(cls, (w, u, m, c, s, t))

        def __getnewargs__(self):
            return (self.wealth, self.utility, self.mrs_deviation, self.constrainedness, self.seconds, self.trades)

        wealth          = property(itemgetter(0))
        utility         = property(itemgetter(1))
        mrs_deviation   = property(itemgetter(2))
        constrainedness = property(itemgetter(3))
        seconds         = property(itemgetter(4))
        trades          = property(itemgetter(5))

        @classmethod
        def from_idx(cls, idx, wealth, utility, mrs_deviation, constrainedness, seconds, trades):
            return cls(wealth[idx], utility[idx], mrs_deviation[idx], constrainedness[idx], seconds[idx], trades[idx])

        def __add__(self, other):
            if other is None or other is 0:
                return self
            return TrialSummary.Stats(*[x + y for (x, y) in zip(self, other)])

        def __sub__(self, other):
            if other is None or other is 0:
                return self
            return TrialSummary.Stats(*[x - y for (x, y) in zip(self, other)])

        def __pow__(self, e):
            if e is None:
                return self
            return TrialSummary.Stats(*[x ** e for x in self])


        def __radd__(self, other):
            return self + other

        def __truediv__(self, other):
            return TrialSummary.Stats(*[x / other for x in self])

        def __str__(self):
            return ("W: %.3f U: %.3f M: %.3f C: %.3f S: %2.2f T: %3.2f"
                    % (self.wealth, self.utility, self.mrs_deviation, self.constrainedness, self.seconds, self.trades))

    # divergence is a drop below beginning - threshold and mrs over min
    def find_div(self, config, u, m):
        bsz = config.div_bucket_size
        div_idx = -1
        u_start = np.average(u[0 : bsz])
        for i in range(bsz, config.rounds):
            u_cur     = np.average(u[i : i + bsz])
            m_max     =        max(m[i : i + bsz])
            # print("%d %0.5f %0.5f" % (i, u_start - u_cur, m_max))
            if u_start - u_cur > config.div_utility_drop and m_max > config.div_mrs_threshold:
                div_idx = i
                break
            
        return div_idx

    def find_conv(self, config, w):
        conv_idx = -1
        max_w = w[config.rounds - 1]
        min_w = max_w
        for i in range(config.rounds - 2, -1, -1):
            max_w = max(w[i], max_w)
            min_w = min(w[i], min_w)
            #print("%d %0.5f" % (i, abs(max_w - min_w)))
            if max_w - min_w > config.conv_threshold:
                if config.rounds - i > config.conv_min_rounds:
                    conv_idx = i + 1
                break
            
        return conv_idx

    def __init__(self, config, wealth, utility, mrs_deviation, constrainedness, seconds, trades):
        self.seed = config.seed
        self.end  = self.Stats.from_idx(config.rounds - 1, wealth, utility, mrs_deviation, constrainedness, seconds, trades)
        self.div_idx = self.find_div(config, utility, mrs_deviation)
        self.did_div = self.div_idx >= 0
        self.conv_idx = self.find_conv(config, wealth)
        self.did_conv = self.conv_idx >= 0 and not self.did_div
        if self.did_conv:
            self.conv = self.Stats.from_idx(self.conv_idx, wealth, utility, mrs_deviation, constrainedness, seconds, trades)
            

    def __str__(self):
        res = "seed: %d" % self.seed 
        res += "\nend: %s" % str(self.end)
        if self.did_conv:
            res += "\nconv idx: %d" % self.conv_idx
            res += "\nconv: %s" % str(self.conv)
        if self.did_div:
            res += "\ndiv idx: %d" % self.div_idx
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
    mrs_deviation   = np.empty([config.rounds])
    constrainedness = np.empty([config.rounds])
    seconds         = np.empty([config.rounds])
    trades          = np.empty([config.rounds])
    trades_accum = 0
        
    start_time = time()
    for i in range(config.rounds):
        _, w, u, m, c, t = do_round(config, traders, i)
        wealths[i] = w
        utilities[i] = u
        mrs_deviation[i] = m
        constrainedness[i] = c
        seconds[i] = time() - start_time
        trades_accum += t
        trades[i] = trades_accum
    
        for t in traders:
            t.reset()

    summary = TrialSummary(config, wealths, utilities, mrs_deviation, constrainedness, seconds, trades)
    if config.verbosity >= 2:
        print()
        print(summary)
        print()

    results.put(summary)

    if config.plotting >= 1 and config.rounds > 1:
        # normalization to make it fit on plot with other variables
        # utilities /= max(utilities)
        # utilities /= config.num_traders / 5


        plt.figure("seed %d" % config.seed, figsize=figsize(1.2, 0.8))
        plt.plot(wealths,         label="Wealth Transfers (%)")
        plt.plot(utilities,       label="Utility Gains (%)")
        plt.plot(mrs_deviation,   label="MRS Deviation")
        plt.plot(constrainedness, label="Constrainedness (%)")

    
        ax = plt.gca()
        cur = ax.get_position()
        ax.set_position([cur.x0, cur.y0 + cur.height * 0.1, cur.width, cur.height * 0.9])

        ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), ncol=1)

        ax.set_xlabel('Day')
        

        savefig("report/images/rounds_seed_%d" % config.seed)

def run(config):
    results = mp.Queue()
    for i in range(0, config.trials):
        p = mp.Process(target=do_trial, args=(config, results))
        p.start()
        config.seed += 1

    data = []
    for i in range(0, config.trials):
        data.append(results.get())

    avg_end  = sum([d.end for d in data]) / config.trials 
    num_conv = len([d for d in data if d.did_conv]) 
    if num_conv > 0:
        avg_conv = sum([d.conv for d in data if d.did_conv]) / num_conv
    pct_conv = num_conv / config.trials
    pct_div  = len([d for d in data if d.did_div]) / config.trials

    var_end = sum([(d.end - avg_end) ** 2 for d in data]) / config.trials
    std_end = var_end ** 0.5

    # We make two strings, human and machine readable

    human = "end avg: %s" % str(avg_end)
    human += "\nend std: %s" % str(std_end)
    if pct_conv > 0:
        human += "\nconv avg: %s" % str(avg_conv)
    human += "\nconverged: %3.f%%" % (pct_conv * 100)
    human += "\ndiverged: %3.f%%" % (pct_div * 100)

    # hack to fix printing later
    if pct_conv == 0:
        avg_conv = [np.nan] * 6
    mach = [str(x) for x in [config.experiment, pct_div, pct_conv, *avg_conv, *avg_end, *std_end]]

    if config.verbosity >= 1:
        print("seed conv  div")
        for d in data:
            print("%4d %4d %4d" % (d.seed, d.conv_idx, d.div_idx))
        print(human)

    i = 0
    while os.path.exists("results/%s_%d" % (config.experiment, i)):
        i += 1
    with open("results/%s_%d" % (config.experiment, i), "w") as f:
        print(" ".join(sys.argv), file=f)
        print(human, file=f)
        print(",".join(mach), file=f)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-traders", type=int, default=2)
    parser.add_argument("-r", "--rounds", type=int, default=1)
    parser.add_argument("-s", "--seed", type=int, default=random.randint(0, 10000),
                        help="integer seed to initialize PRNG")
    parser.add_argument("-t", "--trials", type=int, default=1,
                        help="number of trials to test on, incrementing seed by 1 each time")
    parser.add_argument("-e", "--experiment", default="test",
                        help="name of experiment being run")

    # Intraday args
    parser.add_argument("--trader-file", help="file to load traders from")
    parser.add_argument("-m", "--min-size", type=float, default=0.01,
                        help="minimum size of trade (default: 0.01)")
    # parser.add_argument("-d", "--dynamic", action="store_true", help="dynamic size (binary search)")
    parser.add_argument("--finish-count", type=float, default=1.0,
                        help="number of empty trades per trader to finish a round (default: 1.0)")
    parser.add_argument("-u", "--utility-type", choices=["normal", "nogradient", "leontief"], default="normal",
                        help="utility functions, where normal is Cobb-Douglas with access to the gradient" \
                        "nogradient is Cobb-Douglas without gradients, leontief is Leontief")

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
    parser.add_argument("--backtrack-threshold", type=float, default=0.99,
                        help="backtrack if utility has fallen below [THRESHOLD] * previous (default: 0.99)")

    # Summary args
    parser.add_argument("--div-bucket-size", type=int, default=10,
                        help="number of rounds to average when calculating divergence (default: 10)")
    parser.add_argument("--div-utility-drop", type=float, default=0.05,
                        help="drop in utility necessary for divergence (default: 0.05)")
    parser.add_argument("--div-mrs-threshold", type=float, default=0.05,
                        help="minimun mrs deviation for divergence (default: 0.05)")
    parser.add_argument("--conv-min-rounds", type=int, default=10,
                        help="convergence when wealths are within threshold for [CONV_MIN_ROUNDS] (default: 10)")
    parser.add_argument("--conv-threshold", type=float, default=0.01,
                        help="convergence when wealths are within [THRESHOLD] for min rounds (default: 0.01)")

    # Plotting args
    parser.add_argument("-p", "--plotting", type=int, default=1,
                        help="plotting level: 0 is none, 1 is summary chart, 2 is last round, 3 is every round")
    parser.add_argument("-v", "--verbosity", type=int, default=3,
                        help="verbosity level: 0 is nothing, 1 includes summary of all trials, 2 includes " \
                        "summary of each trial, 3 includes data after each round, 4 includes data from each trader")

    args = parser.parse_args()
    run(args)
