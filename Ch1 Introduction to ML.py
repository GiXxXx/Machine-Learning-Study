import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def error(f, x, y):
    return sp.sum((f(x)-y)**2)

def plot_graph(x, y, d):
    fp = sp.polyfit(x, y, d)
    fd = sp.poly1d(fp)
    fx = sp.linspace(0, x[-1], 1000)
    plt.plot(fx, fd(fx), linewidth=2)
    print(error(fd,x,y),"\n")
    temp = "d = " + str(d)
    legend.append(temp)
    plt.legend(legend, loc = "upper left")

legend = []
    
data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")

data[0][0] = 1

x = data[:,0]
y = data[:,1]

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()

plot_graph(x,y,1)
plot_graph(x,y,2)
plot_graph(x,y,3)
plot_graph(x,y,5)
plot_graph(x,y,10)

plt.show()

#Training date and Test data splition in followed chapers


