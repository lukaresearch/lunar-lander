import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fn = ['', 'Double', 'Duel']
colors = ['red', 'green', 'blue']
i = 0
for f in fn:
    
    infile = open(f + 'dqn1500.txt', 'r') 

    x = []
    y = []
    for line in infile:
        ep, sc = line.split()
        ep = int(ep)
        sc = float(sc)
        x.append(ep)
        y.append(sc)

    plt.plot(x,y, color=colors[i])
    infile.close()
    i+=1

plt.xlabel('episode')
plt.ylabel('score')
plt.show()



    