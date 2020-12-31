import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
import os


with open('abc.txt', 'r') as f:
    x = f.read().splitlines()
    aa = (eval(x[0]))


filename = 'xyz'

plt.clf()
plt.grid()
    
erpa = aa['pa']
erpa1 = aa['pa1']
erpa2 = aa['pa2']
eravgpc = aa['avgpc']
ermaxpc = aa['maxpc']
eravgpg = aa['avgpg']
ermaxpg = aa['maxpg']

rounds = aa['rounds']
mk = 100
q = 3
rr = rounds

plt.plot(rr, erpa, 'o-', label='PA', color='green', linewidth=1, markersize=9, markevery=mk)
plt.plot(rr, erpa1, '*-', label='PA-I', color='blue', linewidth=1, markersize=9, markevery=mk)
plt.plot(rr, erpa2, '.-', label='PA-II', color='red', linewidth=1, markersize=9, markevery=mk)

plt.plot(rr, eravgpc, '^--', label='Avg Perceptron', color='olive', linewidth=1, markersize=9, markevery=mk)
plt.plot(rr, ermaxpc, 'd:', label='Max Perceptron', color='black', linewidth=1, markersize=9, markevery=mk)
plt.plot(rr, eravgpg, 'x--', label='Avg Pegasos', color='magenta', linewidth=1, markersize=9, markevery=mk)
plt.plot(rr, ermaxpg, 's:', label='Max Pegasos', color='brown', linewidth=1, markersize=9, markevery=mk)

plt.xlabel('Rounds',fontsize = "xx-large")
plt.ylabel('Error Rate',fontsize = "xx-large")
plt.title('MNIST Dataset (Partial Label Set Size ='+str(q+1)+')',fontsize = "xx-large")
plt.legend(loc='upper right',fontsize="large")

savepath = os.path.join(filename + ".png")
plt.savefig(savepath)
