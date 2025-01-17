#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
# your code here
# Create 3x2 grid plot view
fig, axes = plt.subplots(3, 2, figsize=(14, 7))

# Plot 1: Line plot
axes[0,0].plot(y0,color="red")

# Plot 2: Scatter plot
axes[0,1].scatter(x1,y1,color="magenta")
axes[0,1].set_title("Men's Height vs Weight",fontsize="x-small")
axes[0,1].set_ylabel("Weight (lbs)",fontsize="x-small")



# Plot 3: Semilogarithmic plot
axes[1,0].semilogy(x2,y2)
axes[1,0].set_xlim(0,28651)
axes[1,0].set_xlabel("Time Years",fontsize="x-small")
axes[1,0].set_ylabel("Fraction Remaining",fontsize="x-small")
axes[1,0].set_title("Exponential Decay of C-14",fontsize="x-small")

# Plot 4: Combined line plots
axes[1,1].set_xlabel("Time (years)",fontsize="x-small")
axes[1,1].set_ylabel("Fraction Remaining",fontsize="x-small")
axes[1,1].set_title("Exponential Decay of Radioactive Elements",fontsize="x-small")
#create dashed line
axes[1,1].plot(x3,y31,linestyle="dashed",color="r",label="C-14")
#create straight line
axes[1,1].plot(x3,y32,"g",label="Ra-226")
# Add legend
axes[1,1].legend()


# Plot 5: Histogram
axes[2,0].set_title("Project A",fontsize="x-small")
axes[2,0].set_ylabel("Number of Students",fontsize="x-small")
axes[2,0].set_xlabel("Grades",fontsize="x-small")
axes[2,0].set_xlim(0,100)
axes[2,0].set_ylim(0,30)
axes[2,0].hist(student_grades,edgecolor="black",bins=7)
# Adjust layout to avoid overlap and space for the title
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.suptitle("All In One", fontsize=16)
# remove unused graphs
fig.delaxes(axes[2,1])
plt.show()