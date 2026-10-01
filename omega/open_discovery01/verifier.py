N=17
M=1<<N
MASK=(1<<M)-1

def initial_channels():
    ch=[]
    for i in range(N):
        bits=0
        block=1<<i
        period=block<<1
        # input index bit i supplies channel i
        for start in range(block,M,period):
            bits |= ((1<<block)-1)<<start
        ch.append(bits)
    return ch

INITIAL=initial_channels()

def evaluate(network, stop_after=None):
    ch=INITIAL.copy()
    for a,b in network:
        if not (0 <= a < b < N):
            return M
        x,y=ch[a],ch[b]
        ch[a]=x & y
        ch[b]=x | y
    bad=0
    for i in range(N-1):
        bad |= ch[i] & (MASK ^ ch[i+1])
    return bad.bit_count()

def verify(network):
    return len(network), evaluate(network), evaluate(network)==0
