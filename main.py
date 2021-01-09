# %%
"""Imports"""
import importlib
import os
import pickle
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
#import model
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp

from evalne.evaluation.split import EvalSplit
import tuning
from h3 import h3
from tqdm import tqdm
from scipy.spatial import distance_matrix

import visualizer
import funciones as fn

# %%
"""Picklear datos"""
users = pickle.load(open("data/users.p", "rb"))
visits = pickle.load(open("data/visits.p", "rb"))
props = pickle.load(open("data/props.p", "rb"))

grafos = pickle.load(open("data/grafos.p", "rb"))
grafos_test = pickle.load(open("data/grafos_test.p", "rb"))

data_type = {"users": users, "props": props}
graph_type = {"props": grafos.Props_f, "users": grafos.Users_f,"bipartite":grafos.B_f}

nodesnuevos=[n for n in grafos_test.B_f if n not in grafos.B_f]
grafos_test.B_f.remove_nodes_from(nodesnuevos)
edgesnuevos=[e for e in grafos_test.B_f.edges if e not in grafos.B_f.edges]
grafos_test.B_f.remove_edges_from(list(grafos_test.B.edges))
grafos_test.B_f.add_edges_from(edgesnuevos)
"""
grafos.set_features("users",mode="adj")
grafos.set_features("props",mode="adj")
grafos.set_features("bipartite",mode="adj")
"""
# %%

dict_method={"gae":"GAE","gae2":"GAE*","vgae":"VGAE","vgae2":"VGAE*", "node2vec":"Node2Vec",
            "sdne":"SDNE","grarep": "GRAREP","gf":"Graph Fact.", "lap": "Mapeo Lap.", "line": "LINE"}
dict_edgemb={"weighted_l1":"L1","weighted_l2":"L2","hadamard":"Hadamard","average":"Promedio"}    
dict_tipo={"props":"Propiedades","users":"Usuarios"}        
# %%

importlib.reload(visualizer)

tipo="users"
method="gae"
dim="30"
emb = pd.read_csv("results/nc/"+tipo+"/"+method+"/"+dim+" 512 0.txt",sep=" ",header=None,skiprows=[0],index_col=0)

G=graph_type[tipo]
#perplexity

# %%
importlib.reload(visualizer)
visual = visualizer.NodeVisualizer(G, emb,proyection_type="TSNE")

ax,scatter=visual.plot_from_graph(alpha=0.7,s=5)

#plt.legend(handles=scatter.legend_elements()[0], labels=['Casa, Venta', 'Casa, Arriendo', 'Depto, Venta','Depto, Arriendo'],ncol=4)
ax.set(title="{}, {} dimensiones".format(dict_method[method],dim))
#plt.savefig("plots/emb"+tipo+"/"+method+dim+".png")

# %%
"""NC chico"""




nodes_chico=users[(users["id_modalidad"]==2)&(users["id_tipo_propiedad"]==2)].index
G_chico=G.subgraph(nodes_chico)

importlib.reload(visualizer)
visual = visualizer.NodeVisualizer(G_chico, emb,proyection_type="TSNE")
ax.set(title="{}, Precio ".format(dict_tipo[tipo]))

caracteristica="valor_uf"

ax,scatter=visual.plot_from_df(users,caracteristica,mode="quantile",alpha=0.7,s=5)


cb=plt.colorbar(scatter, ax=ax)
plt.savefig("plots/emb"+tipo+"/precio"+method+".png")

 # %%
"LP"


importlib.reload(visualizer)

tipo="bipartite"
method="gae"
dim="30"
emb = pd.read_csv("results/lp/bipartite/"+method+"/"+dim+" 64 0.txt",sep=" ",header=None,skiprows=[0],index_col=0)

G=graph_type[tipo]
G_test=grafos_test.B_f

# %%

importlib.reload(visualizer)
edgeemb="hadamard"
visual = visualizer.EdgeVisualizer(G, emb,G_test, proyection_type="skTSNE",emb_method=edgeemb,n=3000)

ax,scatter=visual.plot_from_graph(alpha=0.7,s=5)
ax.set(title="{}, {} dimensiones, {}".format(dict_method[method],dim,dict_edgemb[edgeemb],))
#plt.legend(handles=scatter.legend_elements()[0], labels=['Enlaces Falsos','Enlaces Conocidos','Enlaces Predichos'])
#plt.savefig("plots/emblp/"+method+dim+edgeemb+".png")


# %%
embsamp=emb.sample(n=500)


dist_matrix=distance_matrix(embsamp,embsamp)

# %%
importlib.reload(visualizer)
ordered_dist_mat, res_order, res_linkage = visualizer.compute_serial_matrix(dist_matrix,"complete")
# %%

N = len(dist_matrix)
plt.pcolormesh(ordered_dist_mat)
plt.xlim([0,N])
plt.ylim([0,N])
plt.show()
# %%
"""
presentacion
"""

tester=tuning.LinkPredictionTuning(grafos.B_f,grafos_test.B_f)
emb_dict = emb.T.to_dict("list")
X = {str(k): np.array(v) for k, v in emb_dict.items()}
# %%
tester.evaluator.evaluate_ne(tester.split, X=X,method="method_name",edge_embed_method="hadamard",params={"nw_name":"GPI"})


# %%
tr_edge_embeds, te_edge_embeds = tester.evaluator.compute_ee(tester.split, X, "hadamard")
train_pred, test_pred = tester.evaluator.compute_pred(tester.split, tr_edge_embeds, te_edge_embeds)


# %%

lr=tester.evaluator.lp_model
# %%
user=-1563824

uprops_test=list(dict(G_test[user]).keys())
uprops=[node for node,d in G.nodes(data=True) if d["bipartite"]==0]

ebunch=[(str(user), str(prop)) for prop in  uprops]

edgeemb= fn.hadamard(X,ebunch)
uprops_train=list(dict(G[user]).keys())
# %%
res=lr.predict_proba(edgeemb)

resdf= pd.DataFrame(res,index=uprops)
resdf=resdf.merge(props,left_index=True, right_index=True)
resdf["train"]=np.where(resdf.index.isin(uprops_train), 1, 0)
resdf["test"]=np.where(resdf.index.isin(uprops_test), 1, 0)
resdf=resdf.drop(0,axis=1)
#resdf=resdf[resdf["train"]==0]

# %%
resdfn=resdf.nlargest(20,1)
resdfn.groupby(["test"]).count()
# %%
resdf.hist()
# %%
[node for node in G_test.nodes if node in G.nodes and G_test.nodes[node]["bipartite"]==1 and len(G_test[node])>10 and len(G[node])>10]
# %%
h=[len(G[node]) for node in G_test.nodes if node in G.nodes and G_test.nodes[node]["bipartite"]==1]
# %%
plt.hist(h)
# %%
