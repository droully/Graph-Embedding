import pandas as pd
import numpy as np


def weighted_jaccard(G,u,v):
    r"""
    Calcula la similitud jaccard ponderada para nodos u y v del grafo G

    Parameters
    ----------
    
    G: NetworkX graph
        Grafo.
    u: node
        Primer nodo.
    v: node
        Segundo nodo.
    Returns
    -------
    index: float
        Indice de jaccard entre u y v
    """

    propsu = [k for k in G[u].keys()]
    propsv = [k for k in G[v].keys()]
    
    totalprops = list(set(propsu+propsv))
    x = [G[u].get(prop,dict()).get("weight",0) for prop in totalprops]
    x = np.array(x)
    y = [G[v].get(prop,dict()).get("weight",0) for prop in totalprops]
    y = np.array(y)
    
    xnorm = np.linalg.norm(x,ord=1)
    ynorm = np.linalg.norm(y,ord=1)
    subnorm = np.linalg.norm(x-y,ord=1)
    
    index = (xnorm+ynorm-subnorm)/(xnorm+ynorm+subnorm)    
    return index

def weighted_maximum(G,u,v):
    r"""
    Calcula la similitud maximo para nodos u y v del grafo G

    Parameters
    ----------
    
    G: NetworkX graph
        Grafo.
    u: node
        Primer nodo.
    v: node
        Segundo nodo.
    Returns
    -------
    index: float
        Indice del maximo entre u y v
    """

    propsu = [k for k in G[u].keys()]
    propsv = [k for k in G[v].keys()]
    totalprops = list(set(propsu+propsv))
    x = [G[u].get(prop,dict()).get("weight",0) for prop in totalprops]
    x = np.array(x)
    y = [G[v].get(prop,dict()).get("weight",0) for prop in totalprops]
    y = np.array(y)
    
    xnorm = np.linalg.norm(x,ord=1)
    ynorm = np.linalg.norm(y,ord=1)
    subnorm = np.linalg.norm(x-y,ord=1)
    
    index = (xnorm+ynorm-subnorm)/(2*max(xnorm,ynorm))       
    return index
    

def weighted_jaccard_pair(x,y):

    x = np.array(x)
    y = np.array(y)
    
    xnorm = np.linalg.norm(x,ord=1)
    ynorm = np.linalg.norm(y,ord=1)
    subnorm = np.linalg.norm(x-y,ord=1)
    
    index = (xnorm+ynorm-subnorm)/(xnorm+ynorm+subnorm)    
    return index

def weighted_maximum_pair(x,y):
    x = np.array(x)
    y = np.array(y)
    
    xnorm = np.linalg.norm(x,ord=1)
    ynorm = np.linalg.norm(y,ord=1)
    subnorm = np.linalg.norm(x-y,ord=1)
    
    index = (xnorm+ynorm-subnorm)/(2*max(xnorm,ynorm))       
    return index

def trunc(x,n):
    #funcion auxiliar para cortar los valores de banos y habitaciones
    if x>=n:
        return n
    else:
        return x
    

def dropoutliers(df,column,left=0.05, right=0.95,verbose=0):
    r"""
    funcion que toma un DF y una columna y dropea los outliers respecto a esa columna
    
    df: DF de entrada
    column: columna que dropear
    left: percentil izquierdo
    right: percentil derecho
    """
    
    ql=df[column].quantile(left)
    qr=df[column].quantile(right)
    
    df=df[(df[column]>ql)&(df[column]<qr)]
    
    if verbose:
        print("cuantil %.2f : %.2f, cuantil %.2f: %.2f"%(left,ql,right,qr))
    return df


def average(X, ebunch):
    r"""
    Compute the edge embeddings all node pairs (u,v) in ebunch as the average of the embeddings of u and v.

    Parameters
    ----------
    X : dict
        A dictionary where keys are nodes in the graph and values are the node embeddings.
        The keys are of type str and the values of type array.
    ebunch : iterable of node pairs
        The edges, as pairs (u,v), for which the embedding will be computed.

    Returns
    -------
    edge_embeds : matrix
        A Numpy matrix containing the edge embeddpreserve_inputings in the same order as ebunch.
    """
    edge_embeds = np.zeros((len(ebunch), len(X[list(X.keys())[0]])))
    i = 0
    for edge in ebunch:
        edge_embeds[i] = (X[edge[0]] + X[edge[1]]) / 2.0
        i += 1
    return edge_embeds


def hadamard(X, ebunch):
    r"""
    Compute the edge embeddings all node pairs (u,v) in ebunch as the hadamard distance of the embeddings of u and v.

    Parameters
    ----------
    X : dict
        A dictionary where keys are nodes in the graph and values are the node embeddings.
        The keys are of type str and the values of type array.
    ebunch : iterable of node pairs
        The edges, as pairs (u,v), for which the embedding will be computed.

    Returns
    -------
    edge_embeds : matrix
        A Numpy matrix containing the edge embeddings in the same order as ebunch.
    """
    edge_embeds = np.zeros((len(ebunch), len(X[list(X.keys())[0]])))
    i = 0
    for edge in ebunch:
        edge_embeds[i] = X[edge[0]] * X[edge[1]]
        i += 1
    return edge_embeds


def weighted_l1(X, ebunch):
    r"""
    Compute the edge embeddings all node pairs (u,v) in ebunch as the weighted l1 distance of the embeddings of u and v.

    Parameters
    ----------
    X : dict
        A dictionary where keys are nodes in the graph and values are the node embeddings.
        The keys are of type str and the values of type array.
    ebunch : iterable of node pairs
        The edges, as pairs (u,v), for which the embedding will be computed.

    Returns
    -------
    edge_embeds : matrix
        A Numpy matrix containing the edge embeddings in the same order as ebunch.
    """
    edge_embeds = np.zeros((len(ebunch), len(X[list(X.keys())[0]])))
    i = 0
    for edge in ebunch:
        edge_embeds[i] = np.abs(X[edge[0]] - X[edge[1]])
        i += 1
    return edge_embeds


def weighted_l2(X, ebunch):
    r"""
    Compute the edge embeddings all node pairs (u,v) in ebunch as the weighted l2 distance of the embeddings of u and v.

    Parameters
    ----------
    X : dict
        A dictionary where keys are nodes in the graph and values are the node embeddings.
        The keys are of type str and the values of type array.
    ebunch : iterable of node pairs
        The edges, as pairs (u,v), for which the embedding will be computed.

    Returns
    -------
    edge_embeds : matrix
        A Numpy matrix containing the edge embeddings in the same order as ebunch.
    """
    edge_embeds = np.zeros((len(ebunch), len(X[list(X.keys())[0]])))
    i = 0
    for edge in ebunch:
        edge_embeds[i] = np.power(X[edge[0]] - X[edge[1]], 2)
        i += 1
    return edge_embeds


def compute_edge_embeddings(X, ebunch, method='hadamard'):
    r"""
    Helper method to call any of the edge embedding methods using a simple string parameter.

    Parameters
    ----------
    X : dict
        A dictionary where keys are nodes in the graph and values are the node embeddings.
        The keys are of type str and the values of type array.
    ebunch : iterable of node pairs
        The edges, as pairs (u,v), for which the embedding will be computed.
    method : string, optional
        The method to be used for computing the embeddings. Options are: average, hadamard, l1 or l2.
        Default is hadamard.

    Returns
    -------
    edge_embeds : matrix
        A Numpy matrix containing the edge embeddings in the same order as ebunch.
    """
    if method == 'hadamard':
        return hadamard(X, ebunch)
    elif method == 'average':
        return average(X, ebunch)
    elif method == 'weighted_l1':
        return weighted_l1(X, ebunch)
    elif method == 'weighted_l2':
        return weighted_l2(X, ebunch)
    else:
        raise ValueError("Unknown method!")