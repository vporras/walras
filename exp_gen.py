
prefix = "python3 walras.py -r 500 -n 100 -m 0.0001 -s 0 -v 1 -p 0 -t 100"

ps = [0.5, 0.75, 1]
bs = [[5], [25], [5, 25], [5, 25, 100]]  

for p in ps:
    for b in bs:
        sb = [str(x) for x in b]
        e = "bt_p%2.0f_t99_%s" % (p * 100, "_".join(sb))
        print("echo %s" % e)
        print("%s --backtrack-prob %.2f -b %s -e %s" % (prefix, p, " ".join(sb), e))
