import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time as timer
import numpy as np
import pickle

import argparse
import os
from datetime import datetime

class Model(nn.Module):
	def __init__(self,layers,nonlin=nn.Tanh()):
		super(Model, self).__init__()
		layerslist = []
		for i in range(len(layers)-1):
			layerslist.append(nn.Linear(layers[i],layers[i+1]))
			layerslist.append(nonlin)

		self.netwk = nn.Sequential(*layerslist)

	def forward(self,x):
		return self.netwk(x)

def bmark_time(model,x,y,fwd=True,ups=False):
	if fwd: # for forward
		rn = timer.time()
		outp = model(x)
		runtime = timer.time()-rn
	else: # for backward
		model.zero_grad()
		outp = model(x)
		loss = (outp-y)**2 # assuming least squares loss
		loss = loss.sum()
		if ups:
			optimizer = optim.SGD(model.netwk.parameters(),lr=0.001)
		
		rn = timer.time()
		loss.backward()
		if ups:
			optimizer.step()
		runtime = timer.time()-rn

	return runtime



parser = argparse.ArgumentParser(description='Benchmark network speed.')
parser.add_argument('-l','--layers', nargs='+', type=int, required=True) # layer sizes (including input and output)
parser.add_argument('-bs','--batch_size', type=int, required=True) # size of batch in forward pass
parser.add_argument('-i','--iterations', type=int, required=True) # number of iterations to average out over
parser.add_argument('-p','--passtype', type=str, required=True) # options: fwd, bwd, both
parser.add_argument('-u','--update_bwd',type=bool,required=False) # options: True, False
parser.add_argument('-a','--act_func',type=str,required=True) # options: tanh, ReLU, softmax
# Example command: python3 benchmark_time.py -l 14 128 128 1 -bs 32 -i 100 -p both -u False -a tanh

args = parser.parse_args()
layers = args.layers
bs = args.batch_size
iters = args.iterations
ups = False
if args.update_bwd:
	ups = args.update_bwd
if args.act_func == 'tanh':
	act_func = nn.Tanh()
elif args.act_func == 'ReLU':
	act_func = nn.ReLU()
elif args.act_func == 'softmax':
	act_func = nn.Softmax()

print("NEW RUN:", layers, bs, iters, ups, act_func)

model = Model(layers,nonlin=act_func)

runtimefwd = 0
runtimebwd = 0

for j in range(iters):
	X = torch.randn(bs,layers[0])
	Y = torch.randn(bs)

	if (args.passtype == 'fwd') or (args.passtype == 'both'):
		runtimefwd += bmark_time(model,X,Y,fwd=True,ups=ups)
	if (args.passtype == 'bwd') or (args.passtype == 'both'):
		runtimebwd += bmark_time(model,X,Y,fwd=False,ups=ups)

if (args.passtype == 'fwd') or (args.passtype == 'both'):
	print('Forward Time:',runtimefwd/iters)
if (args.passtype == 'bwd') or (args.passtype == 'both'):
	print('Backward Time:',runtimebwd/iters)


