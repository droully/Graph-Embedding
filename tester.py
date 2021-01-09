#%%
import itertools
import os
import pickle
#%matplotlib inline
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import backboning
from evalne.evaluation.evaluator import LPEvaluator, NCEvaluator
from evalne.evaluation.score import Scoresheet
from evalne.evaluation.split import EvalSplit
from evalne.methods.similarity import common_neighbours
from evalne.utils import preprocess as pp
from openne.dataloaders import Graph, create_self_defined_dataset
from openne.models import gae, gf, grarep, lap, line, lle, node2vec, sdne, vgae

# %%


"""Picklear datos"""
users = pickle.load(open("data/users.p", "rb"))
visits = pickle.load(open("data/visits.p", "rb"))
props = pickle.load(open("data/props.p", "rb"))

grafos = pickle.load(open("data/grafos.p", "rb"))
grafos_test = pickle.load(open("data/grafos_test.p", "rb"))
# %%





# %%

graph=create_self_defined_dataset(root_dir="",name_dict={},name="props_grafos", weighted=True, directed=False, attributed=True)()
graph.set_g(grafos.Users_f)



# %%
model=vgae.VGAE(output_dim=3,dim=3, hiddens=[10],save=True)
vectors=model(graph,epochs=10)


# %%


emb = {str(k): np.array(v) for k, v in vectors.items()}

df = pd.DataFrame(emb).T
X = df.T.to_dict("list")

X = {str(k): v for k, v in X.items()}

X = {str(k): np.array(v) for k, v in vectors.items()}
labels = np.array([[node,attrs["label"][0]] for node, attrs in graph.G.nodes(data=True)])
evaluator = NCEvaluator(graph.G, labels, nw_name="GPI",
                               num_shuffles=10, traintest_fracs=[0.8], trainvalid_frac=0,dim=model.dim)
                               
res=evaluator.evaluate_ne(X=X, method_name="vge",params={})

# %%
res[0].pretty_print()
res[1].pretty_print()
res[2].pretty_print()
res[3].pretty_print()

# %%
len(evaluator.shuffles[0])
# %%


emb = pd.read_csv("results/nc/users/lap/10.txt",sep=" ",header=None,skiprows=[0],index_col=0)
# %%

def test(kwargs1,**kwargs2):
    print("kwargs1")
    for d,v in kwargs1:
        print(d,v) 
    print("kwargs2")
    for d,v in kwargs2:
        print(d,v) 
# %%

from scipy import misc
def f(x):
    return np.sin(x)

def der(f,x,n=1,h=0.05,*args):
    if n==1:
        weights = [-1,1]
    if n==2:
        weights = [1,-2,1]

    val=0
    for k,w in enumerate(weights):
        val += w*f(x+(k-1)*h,*args)
    return val / h**n


# %%



def NewtonsMethod(f, x):
    step=1
    xs=[x]
    while step<10:
        alpha=1/step**(0.1)
        x1 = x -  der(f,x)/der(f,x,n=2)            
        x = x1
        step=step+1
        
        xs.append(x)
    return xs
# %%
xs=np.array(NewtonsMethod(f, 2))
xs
# %%
plt.plot(np.linspace(-3,3),f(np.linspace(-3,3)))
plt.scatter(xs,f(xs),c=range(len(xs)))
# %%
l=np.linspace(-10,10)

# %%


from scipy import misc

def der(f,x,n=1,h=0.05,**kwargs):
    if n==1:
        weights = [-1,1]
    if n==2:
        weights = [1,-2,1]

    val=0
    for k,w in enumerate(weights):
        val += w*f(x+(k-1)*h,**kwargs)
    return val / h**n


def NewtonsMethod(f, x, *kwargs):
    xs=[x]
    for step in tqdm(range(10)):
        #alpha=1/step**(0.1)
        x1 = x -  misc.derivative(f, x,args=kwargs) /misc.derivative(f, x,n=2,args=kwargs)         
        x = x1
        
        xs.append(x)
    return xs
    
def std(power,on):
    grafos.filter_weights(on, power=power)
    hist=grafos.get_histogram(on,filtered=True)

    return np.std(hist)

std_dict={}
