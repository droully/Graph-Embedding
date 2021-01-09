# %%
"""Imports"""
import importlib
import pickle
import random

import matplotlib.pyplot as plt
#import model
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
from h3 import h3
from matplotlib.ticker import PercentFormatter
from networkx.algorithms import bipartite
from tqdm import tqdm

import funciones as fn
import visitconstructor

# %%

def get_visits(desde_users="2019-02-01",hasta_users="2019-06-01",desde_visits="2019-02-01",hasta_visits="2019-06-01"):
    """Carga de Usuarios y Propiedades de SQL"""
    SQL = visitconstructor.FromSQL()
    propquery = ('SELECT id, id_modalidad,id_tipo_propiedad, ubicacion_latitud, ubicacion_longitud, valor_uf,habitaciones, banos '
                'FROM propiedad '
                'JOIN propiedad_valor_uf AS uf ON uf.id_propiedad=id '
                'WHERE id_region = 13 '
                'AND id_tipo_propiedad !=3 '
                'AND ubicacion_latitud BETWEEN -35 and -32 '
                'AND ubicacion_longitud BETWEEN -71.5 and -70 '
                )

    props = SQL.query2df(propquery, index="id")
    
    props1 = fn.dropoutliers(props[props["id_modalidad"] == 1], "valor_uf", left=0.02, right=0.99, verbose=1)
    props2 = fn.dropoutliers(props[props["id_modalidad"] == 2], "valor_uf", left=0.02, right=0.99, verbose=1)
    props = pd.concat([props1, props2])
    del props1, props2

    userquery = ('SELECT id '
                'FROM usuario '
                'WHERE id_empresa is NULL '

                'AND created_at >= \'' + desde_users+'\'  '
                'AND created_at < \'' + hasta_users+'\'  '
                )

    users = SQL.query2df(userquery, index="id")


    """Carga de Visitas de SQL"""
    visits = SQL.visits_from_query(id_usuarios=users.index, id_propiedades=props.index,min_time=desde_visits, max_time=hasta_visits,save=False)

    #visits = SQL.visits_from_query(id_propiedades=propsnunoa.index,min_time="2019-01-01", max_time="2019-02-01" )

    visits = visits.drop("created_at", axis=1)
    visits = visits.merge(users, left_on="id_usuario", right_index=True)

    visits = visits[visits["id_usuario"].duplicated(keep=False)]
    visits = visits[visits["id_entidad"].duplicated(keep=False)]

    props = props[props.index.isin(visits["id_entidad"].values)]
    users = users[users.index.isin(visits["id_usuario"].values)]

    """Asignacion al usuario una caracteristica a partir de las propiedades que visita"""
    importlib.reload(visitconstructor)
    users=visitconstructor.add_user_attributes(users,props,visits)
    return users,props,visits

# %%

users,props,visits=get_visits()
users_test,props_test,visits_test=get_visits(hasta_visits="2019-08-01")
# %%
importlib.reload(visitconstructor)

grafos=visitconstructor.GraphConstructor(users,props,visits,threshold=0.578)
grafos_test=visitconstructor.GraphConstructor(users_test,props_test,visits_test,threshold=0.578)
# %%
nodesnuevos=[n for n in grafos_test.B_f if n not in grafos.B_f]
grafos_test.B_f.remove_nodes_from(nodesnuevos)


edgesnuevos=[e for e in grafos_test.B_f.edges if e not in grafos.B_f.edges]
grafos_test.B_f.remove_edges_from(list(grafos_test.B.edges))
grafos_test.B_f.add_edges_from(edgesnuevos)


# %%
grafos.project("users",power=0.25,threshold=0.822)
grafos.project("props",power=0.3,threshold=0.884)

# %%
grafos.set_features_from_graph("users",mode="adj")
grafos.set_features_from_graph("props",mode="adj")
grafos.set_features_from_graph("bipartite",mode="adj")

# %%
pickle.dump(users, open("data/users.p", "wb" ) )
pickle.dump(visits, open("data/visits.p", "wb" ) )
pickle.dump(props, open("data/props.p", "wb" ) )

pickle.dump(grafos, open("data/grafos.p", "wb" ) )
pickle.dump(grafos_test, open("data/grafos_test.p", "wb" ) )

# %%

"""Picklear datos"""
users = pickle.load(open("data/users.p", "rb"))
visits = pickle.load(open("data/visits.p", "rb"))
props = pickle.load(open("data/props.p", "rb"))

grafos = pickle.load(open("data/grafos.p", "rb"))
grafos_test = pickle.load(open("data/grafos_test.p", "rb"))
# %%

users,props,visits=get_visits(desde_users="2019-02-01",hasta_users="2019-03-01",desde_visits="2019-02-01",hasta_visits="2019-03-01")

# %%

importlib.reload(visitconstructor)
grafos=visitconstructor.GraphConstructor(users,props,visits,threshold=0.578)

# %%

# %%

def plot_histogram(grafos, on, filtered,weighted=False, threshold=None, power=1,**kwargs):

    dic_name={"users":"Usuarios","props":"Propiedades","bipartite": "Bipartito"}
    
    grafos.filter_weights(on,threshold,power)

    hist=grafos.get_histogram(on,filtered)
    

    fig, ax = plt.subplots()
    if weighted:
        weights=np.ones(len(hist))/len(hist)
    else:
        weights=None

    histo=ax.hist(hist,weights=weights,**kwargs)

    ax.set_xlabel("Peso")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribucion Pesos Grafo "+dic_name[on])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) 
    return ax, histo
# %%
hist=grafos.get_histogram("bipartite",filtered=True)
title="Distribucion Pesos Grafo Bipartito"
plot_histogram(grafos,"bipartite",filtered=True,threshold=0,power=1,weighted=True,range=(1,10))

#plt.savefig("plots/bipartito1.png")
# %%
title="Distribucion Pesos Grafo Bipartito"
for i in [0,0.3,0.5,0.7,0.9]:
    plot_histogram(grafos,"bipartite",filtered=True,threshold=i,power=1,title=title,range=(1,10),label=r"$\alpha$="+str(i))
plt.gca().yaxis.set_major_formatter(PercentFormatter(len(hist)))
plt.savefig("plots/bipartitohists.png")



# %%

hist=grafos.get_histogram("users",filtered=True)
title="Indice de Jaccard, Usuarios"
plot_histogram(grafos,"users",filtered=True,threshold=None,power=1,title=title,weighted=True,bins=100,range=(0,1))
plt.savefig("plots/jac_users1.png")
plt.close()

hist=grafos.get_histogram("props",filtered=True)
title="Indice de Jaccard, Propiedades"
plot_histogram(grafos,"props",filtered=True,threshold=None,power=1,title=title,weighted=True,bins=100,range=(0,1))
plt.savefig("plots/jac_props1.png")


# %%

title="Indice de Jaccard, Usuarios"

for p in [0.01,0.05,0.1,0.3,0.5,1]:
    plot_histogram(grafos,"users",filtered=True,threshold=None,power=p,title=title,weighted=True,bins=100,range=(0,1),label="p="+str(p))

plt.savefig("plots/jac_users.png")



# %%

title="Indice de Jaccard, Propiedades"

for p in [0.01,0.05,0.1,0.3,0.5,1]:
    plot_histogram(grafos,"props",filtered=True,threshold=None,power=p,title=title,weighted=True,bins=100,range=(0,1),label="p="+str(p))

plt.savefig("plots/jac_props.png")

# %%
"Encontrar p usuarios"
std=[]
tipo="users"
l=[0.1,0.15,0.2,0.25,0.3,0.35,0.4]
for i in l:    
    grafos.filter_weights(tipo,power=i)
    hist=grafos.get_histogram(tipo,filtered=True)

    std.append(np.std(hist))

i=3
plt.plot(l,std)
plt.scatter(l[i],std[i],c="red")
plt.legend(["p="+str(l[i])],loc='lower right')
plt.title("Desviacion Estandar Usuarios")
plt.xlabel("p")
plt.ylabel("Desviacion Estandar")

plt.savefig("plots/std_jac_"+tipo+".png")
    

# %%
"Encontrar p propiedades"
std=[]
tipo="props"
l=[0.1,0.15,0.2,0.25,0.3,0.35,0.4]
for i in l:    
    grafos.filter_weights(tipo,power=i)
    hist=grafos.get_histogram(tipo,filtered=True)

    std.append(np.std(hist))

i=4
plt.plot(l,std)
plt.scatter(l[i],std[i],c="red")
plt.legend(["p="+str(l[i])],loc='lower right')
plt.title("Desviacion Estandar Propiedades")
plt.xlabel("p")
plt.ylabel("Desviacion Estandar")

plt.savefig("plots/std_jac_"+tipo+".png")
  # %%
  

grafos.filter_weights("users",power=0.25,threshold=0.822)
grafos.filter_weights("props",power=0.3,threshold=0.884)

# %%

"""cobertura"""

tipo="bipartite"
tipo_dict={"bipartite":grafos.B,"users":grafos.Users,"props":grafos.Props}


edgtotal=len(tipo_dict[tipo].edges)
nodetotal=len([node for node in tipo_dict[tipo].nodes if tipo_dict[tipo].degree(node) > 1])


l_dic={"bipartite":-1,"users":-1.5,"props":-1.5}

l=np.logspace(l_dic[tipo],0,num=9)
l=np.insert(l,0,0)
l=1-l

coverages=[]
edges=[]
nodes=[]

for alpha in tqdm(l):
    grafos.filter_weights(tipo,threshold=alpha)
    tipo_f_dict={"bipartite":grafos.B_f,"users":grafos.Users_f,"props":grafos.Props_f}

    non_isolated=len([node for node in tipo_f_dict[tipo].nodes if tipo_dict[tipo].degree(node) > 1])
    coverage=non_isolated/nodetotal
    edges.append(len(tipo_f_dict[tipo].edges))
    nodes.append(len(tipo_f_dict[tipo]))
    coverages.append(coverage)

# %%
dic_name={"bipartite":"Bipartito","users":"Usuarios","props":"Propiedades"}
#plt.plot(l,coverages,linewidth=1,linestyle='dashed')
#plt.scatter(l,coverages,c=l, cmap="rainbow")
edge_relativo=[i/edgtotal for i in edges]
i_dic={"bipartite":6,"users":5,"props":4}
i=i_dic[tipo]
plt.scatter(edge_relativo[i],coverages[i],s=200,c="red")
plt.legend(["$\\alpha$="+str(l[i])[:5]],loc='lower right' )

plt.text(edge_relativo[i], coverages[i]-0.1, '({:.4f}, {:.4f})'.format(edge_relativo[i], coverages[i]))
plt.scatter(edge_relativo,coverages,c=l, cmap="winter")
plt.plot(edge_relativo,coverages,linewidth=1,linestyle='dashed')
plt.xlabel("Vertices Relativos")
plt.ylabel("Cobertura")
plt.title("Filtrado "+dic_name[tipo])
plt.colorbar()
#plt.savefig("plots/filter"+tipo+".png")

plt.show()



# %%

hist=grafos.get_histogram("users",filtered=True)
title="Usuarios Final"
plot_histogram(grafos,"users",filtered=True,threshold=0.822,power=0.25,title=title,weighted=True,bins=100,range=(0,1))
plt.savefig("plots/usersfinal.png")
plt.close()

hist=grafos.get_histogram("props",filtered=True)
title="Propiedades Final"
plot_histogram(grafos,"props",filtered=True,threshold=0.884,power=0.3,title=title,weighted=True,bins=100,range=(0,1))
plt.savefig("plots/propsfinal.png")


# %%
hist=grafos.get_histogram("bipartite",filtered=True)
title="Bipartito Final"
plot_histogram(grafos,"bipartite",filtered=True,threshold=0.578,power=1,title=title,weighted=True,bins=10,range=(0,10))
plt.savefig("plots/bipartitofinal.png")

# %%

users["id_modalidad"].hist(bins=10, grid=False)


# %%

n=len(users["id_modalidad"])
plt.hist(users["id_modalidad"]-1,weights=np.ones(n)/n,bins=20)

plt.title('Distribucion Modalidad Usuarios')
plt.xlabel('Modalidad')
plt.ylabel('Frecuencia')
#plt.savefig("plots/usersmodalidad.png")
plt.show()
# %%

n=len(users["id_tipo_propiedad"])
plt.hist(users["id_tipo_propiedad"]-1,weights=np.ones(n)/n,bins=20)

plt.title('Distribucion Tipo Usuarios')
plt.xlabel('Tipo')
plt.ylabel('Frecuencia')
plt.savefig("plots/userstipo.png")
plt.show()
    # %%


# %%

n=len(props["id_modalidad"])
plt.hist(props["id_modalidad"]-1,weights=np.ones(n)/n,bins=20)

plt.title('Distribucion Modalidad Propiedades')
plt.xlabel('Modalidad')
plt.ylabel('Frecuencia')
plt.savefig("plots/propsmodalidad.png")
plt.show()
# %%

n=len(props["id_tipo_propiedad"])
plt.hist(props["id_tipo_propiedad"]-1,weights=np.ones(n)/n,bins=20)

plt.title('Distribucion Tipo Propiedades')
plt.xlabel('Tipo')
plt.ylabel('Frecuencia')
plt.savefig("plots/propstipo.png")
plt.show()
# %%

len((users[(users["id_modalidad"]<1.95) & (users["id_modalidad"]>1.05)]))
# %%
len(users)
# %%# %%
