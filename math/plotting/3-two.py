#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# your code here

# titles for x,y
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.title("Exponential Decay of Radioactive Elements")
#create dashed line
plt.plot(x,y1,linestyle="dashed",color="r",label="C-14")
#create straight line
plt.plot(x,y2,"g",label="Ra-226")
# Add legend
plt.legend()
plt.show()
