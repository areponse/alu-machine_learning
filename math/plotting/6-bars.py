#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
# List of species
species = ["Farrah", "Fred", "Felicia"]
colors = ["red","yellow","#ff8000","#ffe5b4"]
labels = ["apples","bananas","oranges","peaches"]

# Create a figure and axis
fig, axs = plt.subplots()

# Initialize bottom array to stack bars on top of each other
bottom = np.random.randint(3)

# Iterate through each row in the fruit array (each represents a different boolean/species group)
for i, weight_count in enumerate(fruit):
    p = axs.bar(species, weight_count, width=0.5, label=labels[i], bottom=bottom, color=colors[i])
    bottom += weight_count  # Update bottom to stack the next bars

# Add title and legend
axs.set_title("Number of Fruit per Person")
axs.legend(loc="upper right")
axs.set_ylim(0,80)
axs.set_ylabel("Quantity of Fruit")

# Display the plot
plt.show()