#%%
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%%
task,tipo="nc","users"

df=pd.DataFrame()
for root, dirs, files in os.walk(r".\results\{}\{}".format(task,tipo)):
    for name in files:
        if name=="dimf1.csv":
            method=root.split('\\')[-1]
            df1=pd.read_csv(os.path.join(root, name))
            df1["method"]=method
            d_dim={0:10,1:30,2:50,3:100,4:300,5:500}
            df1["dim"]=df1['Unnamed: 0'].map(d_dim)
            df1=df1.drop(["Unnamed: 0"],axis=1)
            
            df=pd.concat([df,df1])
del df1
# %%

dict_method={"gae":"GAE","gae2":"GAE*","vgae":"VGAE","vgae2":"VGAE*", "node2vec":"Node2Vec",
            "sdne":"SDNE","grarep": "GRAREP","gf":"Graph Fact.", "lap": "Mapeo Lap.", "line": "LINE"}

methods=pd.unique(df["method"])
methods=[  'node2vec', 'line', 'lap','gf', 'grarep', 'sdne','gae', 'vgae']

edge_emb="hadamard"
# %%
"NC"

for method in methods:
    toplot=df[df["method"]==method]
    plt.plot(toplot["dim"].astype('str'),toplot["score"])

#plt.legend([dict_method[m] for m in methods],ncol=2)

dic_title_task={"nc":"Clasificacion de Nodos","lp": "Prediccion de Enlaces"}
dic_title_tipo={"bipartite":"Bipartito", "users":"Usuarios","props":"Propiedades"}
plt.xlabel("Dimensión del Embedding")
plt.ylabel("Puntaje F1 Ponderado")
dict_ylim={"users":(0.8,0.97),"props":(0.9,0.99)}
plt.ylim(dict_ylim[tipo])
plt.title("{} {}".format(dic_title_task[task],dic_title_tipo[tipo]))

plt.savefig("plots/results/"+tipo+".png")
# %%


df_score=df
df_score=df_score[["method","score",'dim']]
df_score=df_score.groupby(["dim",'method'])["score"].aggregate('first').unstack()
df_score.columns=[dict_method[m]for m in df_score.columns]
df_score=df_score.T
df_score.to_csv("f1"+tipo+".csv",float_format='%.5f')

# %%
"LP"

fig, axs = plt.subplots(2, 2,figsize=(12.8, 9.6))

for edge_emb,n in zip(["l1","l2","hadamard","average"],[[0,0],[0,1],[1,0],[1,1]]):
    ax=axs[n[0], n[1]]
    for method in methods:
        toplot=df[df["method"]==method]
        
        ax.plot(toplot["dim"].astype('str'),toplot[edge_emb])



    dic_title_task={"nc":"Clasificacion de Nodos","lp": "Prediccion de Enlaces"}
    dic_title_tipo={"bipartite":"Bipartito", "users":"Usuarios","props":"Propiedades"}
    dict_ylim={"l1":(0.5,0.9),"l2":(0.5,0.9),"hadamard":(0.5,0.92),"average":(0.5,0.78)}

    title_dict={"l1":"L1","l2":"L2","hadamard":"Hadamard","average":"Promedio"}
    ax.set(xlabel='Dimensión del Embedding', ylabel='Puntaje AUROC',ylim=dict_ylim[edge_emb],
        title="Prediccion de Enlaces "+title_dict[edge_emb])
fig.legend([dict_method[m] for m in methods],ncol=4,loc='lower center',
           fancybox=True, shadow=True)

plt.savefig("plots/results/"+tipo+".png")
# %%
