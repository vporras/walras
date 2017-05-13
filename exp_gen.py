
prefix = "python3 walras.py -r 500 -n 100 -m 0.0001 -s 0 -v 1 -p 0 -t 10"

ps = [0.25, 0.5, 0.75, 1]
ts = [0.99, 0.97, 0.95, 0.93, 0.9]
bs = [[5], [10], [25], [5, 10], [5, 25], [5, 50], [5, 25, 100], [10, 25], [10, 50], [10, 25, 100]]  

for p in ps:
    for t in ts:
        print("p: %.2f t: %.2f" % (p, t))
        for b in bs:
            sb = [str(x) for x in b]
            e = "bt_p%2.0f_t%2.0f_%s" % (p * 100, t * 100, "_".join(sb))
            print("echo %s" % e)
            print("%s --backtrack-prob %.2f --backtrack-threshold %.2f -b %s -e %s &" % (prefix, p, t, " ".join(sb), e))
        print("wait\n\n")
