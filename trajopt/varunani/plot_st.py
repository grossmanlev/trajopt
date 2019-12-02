import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plot Stuff.')
parser.add_argument('-xlab','--xlabel',type=str,required=True)
parser.add_argument('-ylab','--ylabel',type=str,required=True)
parser.add_argument('-x','--xvals', nargs = '+', type=int, required=True) 
parser.add_argument('-y','--yvals', nargs = '+', type=float, required=True)

args = parser.parse_args()

plt.figure()
plt.plot(args.xvals,args.yvals)
plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)
plt.title('{} vs. {}'.format(args.ylabel,args.xlabel))
plt.savefig('{}_{}.png'.format(args.xlabel,args.ylabel))