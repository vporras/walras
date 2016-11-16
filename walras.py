from itertools import product
from random import sample, randint
import matplotlib.pyplot as plt

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

        
        
class Trader():
    """Trader class"""

    def __init__(self, name, alpha, allocation=(0,0)):
        self.name = name
        self.preference = preference
        self.allocation = allocation

    def mrs(self):
        l1,l2 = self.preference
        return l1/l2

    def change_allocation(self, delta):
        self.allocation  = (self.allocation[0] + delta[0],
                             self.allocation[1] + delta[1])

    def joint_mrs(self, other):
        return (self.mrs() * other.mrs()) ** 0.5

    def get_size(self, other, joint_mrs):
       if self.mrs() > other.mrs():
           return min(joint_mrs * self.allocation[1], other.allocation[0])
       elif self.mrs() < other.mrs():
           return -min(joint_mrs * other.allocation[1], self.allocation[0])
       else:
           return 0

    def trade(self, other):
        joint_mrs = self.joint_mrs(other)
        size = self.get_size(other, joint_mrs)
        self.change_allocation((size, -size / joint_mrs))
        other.change_allocation((-size, size / joint_mrs))
        return Trade(self, other, size, joint_mrs)

    def __str__(self):
        return ("Name:{name},\n preference=({preference}),\n allocation=({allocation})".format(
            name = self.name,
            preference = self.preference,
            allocation = self.allocation))

def trade_random(traders):
    a,b = sample(traders,2)
    return(a.trade(b))

def random_traders(n):
    return [Trader(i,(randint(1,10),randint(1,10)),
                      (randint(1,10),randint(1,10))) for i in range(n)]

if __name__ == "__main__":
    #i = Trader("I", (1, 8), (1, 1))
    #j = Trader("J", (2, 1), (1, 1))
    #k = Trader("K", (9, 2), (1, 1))
    #print(i.trade(j))
    #print(i.trade(k))
    #print(j.trade(k))
    
    sizes = []
    traders = random_traders(5)
    for i in range(1000):
        trade = trade_random(traders)
        sizes.append(abs(trade.size))
    plt.plot(sizes)
    plt.show()
