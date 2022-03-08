import os
import matplotlib
from matplotlib import pyplot as plt 
import numpy as np 



def graphs(y1,y2,y3,y4,y5,y6):

    x=["None", "n/25","n/50" ]

    # Initialise the figure and axes.
    fig, ax = plt.subplots(1, figsize=(8, 6))

    # Set the title for the figure
    fig.suptitle('Item_based_recommendation', fontsize=15)

    # Draw all the lines in the same plot, assigning a label for each one to be
    # shown in the legend.
    ax.plot(x, y1, color="red", label="Pearson, >0 sim")
    ax.plot(x, y2, color="green", label="Pearson, 0.3 sim")
    ax.plot(x, y3, color="blue", label="Pearson, 0.5 sim")
    ax.plot(x, y4, color="yellow", label="Distance, >0 sim")
    ax.plot(x, y5, color="Orange", label="Distance, 0.3 sim")
    ax.plot(x, y6, color="Purple", label="Distance, 0.5sim")

    # Add a legend, and position it on the lower right (with no box)
    plt.legend(loc="lower right", title="Legend Title", frameon=False)
    plt.savefig("graphs/Item_based_MSE")
    plt.show()
    

def main():

    
    y1= [1.0997077956,0.7503563939,0.7503563939] #curve1: Pearson, >0 sim
    y2=[1.0854011983,0.7727069532,0.8308090518] #Curve2: Pearson, 0.3 sim
    y3=[1.0699380732,0.8102968499, 0.8886991324] #Curve3: Pearson, 0.5 sim
    y4=[0.1507575741,0.7165018242,0.7165018242] #Curve4: Distance, >0 sim
    y5=[0.1040306114, 0.8640133938, 0.7570739217] #Curve5: Distance, 0.3sim
    y6=[0, 0.9042446406, 0] #Curve6: Distance, 0.5sim

    list=[]

    list.extend([y1,y2,y3,y4,y5,y6])
    print(list)



    # graphs(y1,y2,y3,y4,y5,y6)
    

if __name__ == '__main__':
    main()