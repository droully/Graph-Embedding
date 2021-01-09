"""
Modulo que construye, fitea y visualiza embeddings de nodos
"""



import time

import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import funciones as fn
try:
    from openTSNE import TSNE
except:
    pass
from sklearn.decomposition import PCA

from tqdm import tqdm

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import  linkage
import random
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE as skTSNE



def GetNegativeEdges(G, n,G_test=None):
    r"""
    Funcion auxiliar que construye los enlaces negativos

    """
            
    prop_nodes=[n for n, d in G.nodes(data=True) if d['bipartite']==0]
    user_nodes=[n for n, d in G.nodes(data=True) if d['bipartite']==1]

    non_edges=[]

    while len(non_edges) <=n:
        random_prop = random.choice(prop_nodes)
        random_user = random.choice(user_nodes)
        edge=(random_prop,random_user) 

        if G_test is not None:
            if G.has_edge(*edge) or G_test.has_edge(*edge):
                continue
            else: 
                non_edges.append(edge)
        else:
            if G.has_edge(*edge):
                continue
            else: 
                non_edges.append(edge)
    return non_edges

"""
    def distance_matrix(self,X, save=False, savefile=None):
        
        construccion de la matriz de distancias para visualizar en kepler
        save: bool, si guardar o no
        savefile: str, nombre del archivo de salida
      
        
        emb=self.predict(X)
        dist=sp.spatial.distance.pdist(emb)
        square=sp.spatial.distance.squareform(dist)


        distancedf = pd.DataFrame(data=square, columns=X, index=X)

        #distancedf = pd.merge(users, distancedf, left_index=True, right_index=True)

 

        if save:
            if savefile is None:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                distancedf.to_csv("output/embedding/distmatrix"+timestr+".csv")
            else:
                distancedf.to_csv(savefile)
        return distancedf

"""




class Visualizer():
    """
    Clase general que construye una proyeccion de enlaces o nodos, utilizando PCA o TSNE
    """

    def __init__(self, G, emb, proyection_type="TSNE", **kwargs):

        self.tipo=G.graph["tipo"]
        self.G=G
        self.emb=emb

        #self.df=data_type[self.tipo]
        self.setproyection(proyection_type,**kwargs)

    def setproyection(self, proyection_type="TSNE",**kwargs):
        r"""
        Calcular proyeccion de los datos

        Parameters
        ----------
        proyection_type: str
            Tipo de proyeccionhay tres opciones: TSNE, implementado con OpenTSNE;
                skTSNE, implementado por sklearn; y PCA, implementado por sklearn. 
        kwargs: dict
            Argumentos para la proyeccion (Perplexity, etc)
        """
        if self.emb.shape[1]==2:
            X_proyected = self.emb.values
        elif  proyection_type=="PCA":
            X_proyected = PCA(n_components=2,**kwargs).fit_transform(self.emb)
        elif  proyection_type=="skTSNE":
            X_proyected = skTSNE(n_components=2,**kwargs).fit_transform(self.emb)
        elif  proyection_type=="TSNE":
            X_proyected = pd.DataFrame(TSNE(n_components=2,n_jobs=8,**kwargs).fit(self.emb.values),index=self.emb.index,columns=["xdim","ydim"])
        


        self.ids = self.emb.index
        self.proyected = pd.DataFrame(X_proyected, columns=["xdim", "ydim"], index=self.ids)

    def plot_data(self,node_color=None, quantile=True ,**kwargs):
        r"""
        Plotea los datos proyectados

        Parameters
        ----------
        node_color: DataFrame
            Dataframe con indices los datos y columna el color
        quantile: bool
            Si colorear los valores en si mismo, o los cuantiles, esto con el fin de reducir el efecto
            de outliers.   
            
        kwargs: dict
            Argumentos para scatter

        Returns
        -------
        ax: axes
            eje de pyplot
        scatter:
            scatter de pyplot
        """
        
        proyecion = self.proyected
        df=proyecion.merge(node_color,left_index=True, right_index=True)

        n=len(pd.unique(node_color["node_color"]))
        if n<10:
            cmap = matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap("tab10").colors[:n])
        else:
            cmap ="plasma"
            if quantile: 
                df['node_color']=df['node_color'].rank()
                df['node_color']=100*df['node_color']/df['node_color'].max()
                
        fig, ax = plt.subplots()
        scatter=ax.scatter(df['xdim'], df['ydim'],  c=df['node_color'],cmap=cmap,**kwargs)

        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_axis_off()
        return ax,scatter

class NodeVisualizer(Visualizer):
    r"""
    Clase encargada de visualizar nodos.

    Parameters
    ----------
    G: NetworkX graph
        Grafo.
    emb: DataFrame
        Dataframe con indices los nodos y valores el embedding
    proyection_type: str
        Tipo de proyeccionhay tres opciones: TSNE, implementado con OpenTSNE;
            skTSNE, implementado por sklearn; y PCA, implementado por sklearn. 
    kwargs: dict
        Argumentos para la proyeccion (Perplexity, etc)
    """
    def __init__(self,G, emb, proyection_type="TSNE", **kwargs):        
        emb=emb.loc[list(G.nodes)]
        super(NodeVisualizer, self).__init__(G, emb, proyection_type=proyection_type, **kwargs)

    def plot_from_graph(self, feature="label",  quantile=True ,**kwargs):
        r"""
        Plotear la proyeccion coloreando los nodos a partir de informacion del grafo.

        Parameters
        ----------
        feature: str
            Atributo del nodo del grafo con el que colorear
        quantile: bool
            Si colorear los valores en si mismo, o los cuantiles, esto con el fin de reducir el efecto
            de outliers.
        kwargs: dict
            Argumentos para scatter

        Returns
        -------
        ax: axes
            eje de pyplot
        scatter:
            scatter de pyplot
        """
        try:
            node_color={node:attrs[feature][0] for node, attrs in self.G.nodes(data=True)}
        except:
            node_color={node:attrs[feature] for node, attrs in self.G.nodes(data=True)}
        node_color=pd.DataFrame.from_dict(node_color, orient='index',columns=["node_color"])
        ax,scatter=self.plot_data(node_color=node_color, quantile=quantile, **kwargs)
        return ax,scatter
        
    def plot_from_df(self, X, columns_to_plot,  quantile=True ,**kwargs):
        r"""
        Plotear la proyeccion coloreando los nodos a partir de informacion de un dataframe.

        Parameters
        ----------
        X: DataFrame
            Dataframe con indices los nodos y columnas atributos

        columns_to_plot: str
            Columna que plotear del dataframe, TO DO: que aguante mas de una columna.

        quantile: bool
            Si colorear los valores en si mismo, o los cuantiles, esto con el fin de reducir el efecto
            de outliers.
        kwargs: dict
            Argumentos para scatter
        Returns
        -------
        ax: axes
            eje de pyplot
        scatter:
            scatter de pyplot
        """

        node_color=pd.DataFrame({"node_color":X[columns_to_plot]})

        ax,scatter=self.plot_data(node_color=node_color, quantile=quantile,**kwargs)

        return ax,scatter

class EdgeVisualizer(Visualizer):
    r"""
    Clase encargada de visualizar enlaces.

    Parameters
    ----------
    G: NetworkX graph
        Grafo de entrenamiento.
    emb: DataFrame
        Dataframe con indices los nodos y valores el embedding
    G_test: NetworkX graph
        Grafo de testeo.
    emb_method: str
        Metodo de embedding de enlace, "weighted_l1", "weighted_l2", "hadamard", "average".
    proyection_type: str
        Tipo de proyeccionhay tres opciones: TSNE, implementado con OpenTSNE;
        skTSNE, implementado por sklearn; y PCA, implementado por sklearn. 
    n: int
        Cantidad de enlaces por clase para muestrear.
    kwargs: dict
        Argumentos para la proyeccion (Perplexity, etc)
    """
    def __init__(self,G, emb,G_test=None, emb_method="hadamard", proyection_type="TSNE",n=None, **kwargs):
        if G_test is None:
            G_test=nx.Graph()

        emb_dict=emb.T.to_dict('list')
        emb_dict={node:np.array(emb) for node, emb in emb_dict.items()}

        train_edges=G.edges()
        edge_emb=fn.compute_edge_embeddings(emb_dict,train_edges,emb_method)
        self.dftrainedges=pd.DataFrame(edge_emb,train_edges)

        test_edges=G_test.edges()
        edge_emb=fn.compute_edge_embeddings(emb_dict,test_edges,emb_method)
        self.dftestedges=pd.DataFrame(edge_emb,test_edges)

        if n is not None:
            self.dftrainedges=self.dftrainedges.sample(n=n)
            try:
                self.dftestedges=self.dftestedges.sample(n=n)
            except:
                pass



        non_edges=GetNegativeEdges(G, len(self.dftrainedges),G_test=G_test)
        non_edge_emb=fn.compute_edge_embeddings(emb_dict,non_edges,emb_method)
        self.dfnonedges=pd.DataFrame(non_edge_emb,non_edges)
        edgemb=pd.concat([self.dfnonedges,self.dftrainedges,self.dftestedges])
        super(EdgeVisualizer, self).__init__(G, edgemb, proyection_type=proyection_type, **kwargs)
        
    def plot_from_graph(self,**kwargs):
        r"""
        Plotear la proyeccion coloreando los enlace a partir de informacion del grafo.

        Parameters
        ----------

        kwargs: dict
            Argumentos para scatter

        Returns
        -------
        ax: axes
            eje de pyplot
        scatter:
            scatter de pyplot
        """  
        
        
        edg0=pd.DataFrame(index=self.dfnonedges.index)
        edg0["node_color"]=0
        edg1=pd.DataFrame(index=self.dftrainedges.index)
        edg1["node_color"]=1
        edg2=pd.DataFrame(index=self.dftestedges.index)
        edg2["node_color"]=2
        
        node_color=pd.concat([edg0,edg1,edg2])
        ax,scatter=self.plot_data(node_color=node_color,**kwargs)
        return ax,scatter


""" visualizacion de matriz de distancia"""
def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
    
def compute_serial_matrix(dist_mat,method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage

