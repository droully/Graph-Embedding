"""
Modulo de carga de datos a partir de SQL o un CSV, y construccion de grafos

"""
import time

import networkx as nx
import numpy as np
import pandas as pd

import psycopg2
from h3 import h3
from networkx.algorithms import bipartite
from tqdm import tqdm

import backboning
import funciones as fn

import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter

USER = "USER"
PASSWORD = "PASSWORD"
DBNAME = "DBNAME"
HOST = "HOST"
PORT = "PORT"


class FromSQL():

    r"""
    Clase encargada de cargar a partir de queries de sql

    Parameters
    ----------

    user: str
        Usuario
    password: str
        Password
    dbname: str
        Database Name
    host: str
        Host
    port: str
        Port
    """

    def __init__(self, user=USER, password=PASSWORD, dbname=DBNAME, host=HOST, port=PORT):

        self.user = user
        self.password = password
        self.dbname = dbname
        self.host = host
        self.port = port

    def query2df(self, query, index=None):
        r"""

        Convierte una query de sql a un dataframe

        Parameters
        ----------

        query: str
            Query
        index: str
            Que columna de la tabla se convertira en el indice del df

        Returns
        -------
        data: DataFrame
            Dataframe con los datos
        """
        conn = psycopg2.connect(dbname=self.dbname,
                                user=self.user,
                                host=self.host,
                                password=self.password,
                                port=self.port)
        dat = pd.read_sql_query(query, conn)
        conn.close()
        if not index is None:
            dat = dat.set_index(index)
        return dat

    def visits_from_query(self, id_propiedades=None,
                          id_usuarios=None,
                          min_time=None, max_time=None,
                          save=False, savefile=None):
        r"""
        Construye un dataframe con las visitas tales que  sean usuarios de id_usuarios,
        propiedades de id_propiedades y entre min_time y max_time.

        Parameters
        ----------

        id_propiedades: lista de str
            Propiedades de las que obtener las visitas.
        id_usuarios: lista de str
            Usuarios de los que obtener las visitas.
        min_time: str
            Tiempo minimo,  con forma Y%m%d, ie. 2013-03-21, incluyente (>=)
        max_time:  str
            Tiempo maximo,  con forma Y%m%d, ie. 2013-03-21, excluyente (<)
        save: bool
            Guardar el DF
        -------
        data: DataFrame
            Dataframe con las visitas
        """

        query = ('SELECT id, id_entidad, id_usuario, created_at '
                 'FROM log_acciones '
                 'WHERE id_accion!=1 '
                 'AND id_usuario IS NOT NULL '
                 )

        if not id_propiedades is None:
            strids = str(tuple(id_propiedades))
            query = query+('AND id_entidad in ' + strids+' ')

        if not id_usuarios is None:
            strids = str(tuple(id_usuarios))
            query = query+('AND id_usuario in ' + strids+' ')

        if not min_time is None:
            query = query+('AND created_at >= \'' + min_time+'\' ')

        if not max_time is None:
            query = query+('AND created_at < \'' + max_time+'\' ')

        visits = self.query2df(query, index="id")
        visits = visits.sort_values(by="created_at")
        if save:
            if savefile is None:
                visits.to_csv("visitas"+min_time+max_time+".csv")
            else:
                visits.to_csv(savefile)
        return visits


def visits_from_csv(savefile):
    r"""
    Carga un dataframe de visitas de un .csv

    Parameters
    ----------

    savefile: str
        Ubicacion del archivo.

    Returns
    -------
    data: DataFrame
        Dataframe con las visitas
    """
    visits = pd.read_csv(savefile, index_col="id")
    visits['created_at'] = pd.to_datetime(visits['created_at'])
    return visits


class GraphConstructor():
    def __init__(self, users, props, visits, feats=None, threshold=0, power=1, min_degree=5,
                 filter_method=backboning.disparity_filter):
        r"""
        Clase encargada de la construccion de los grafos

        Parameters
        ----------        
        users: DataFrame
            Dataframe con usuarios.
        props: DataFrame
            Dataframe con propiedades.
        visits: DataFrame con visitas
            Dataframe con visitas. Extension: podrian no ser visitas, solo basta que sea un
             Dataframe con una columna ids de usuarios, y otra ids de propiedades.
        feats:  str
            TO DO: caracteristicas que asignarle a un nodo.
        threshold: float o None
            threshold con el que filtrar al grafo bipartito.
        power: float
            potencia a la que elevar los enlaces del grafo bipartito. (no es necesario para el grafo bipartito)
        min_degree: int
            Grado minimo que deben tener los nodos. 
        filter_method: metodos de backboning
            Que metodo de filtrado usar. (recomiendo usar el por defecto)


        Atributes
        ---------
        B: NetworkX Graph        
            Grafo bipartito original.    
        Users: NetworkX Graph
            Grafo proyectado de usuarios original.     
        Props: NetworkX Graph
            Grafo proyectado de propiedades original.     
        B_f: NetworkX Graph
            Grafo bipartito filtrado.  
        Users_f: NetworkX Graph
            Grafo proyectado de usuarios filtrado.     
        Props_f: NetworkX Graph
            Grafo proyectado de propiedades filtrado.     
        users_nodes: list
            Lista de nodos de usuarios.
        prop_nodes: list
            Lista de nodos de propiedades.
        """

        def labeler(node, tipo):
            """
            Funcion que a cada nodo, le da un label diciendo si  casa/depto, venta/arriendo

            (quizas se puede sacar y utilizar set_features_from_df)
            """
            data_type = {0: props, 1: users}
            df = data_type[tipo]
            dic_labeler = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

            def f(t):
                if 1.5 <= t and t <= 2:
                    return 1
                if 1 <= t and t <= 1.5:
                    return 0

            if tipo == 0:
                x = f(df.loc[node]["id_tipo_propiedad"])
                y = f(df.loc[node]["id_modalidad"])
            if tipo == 1:
                x = f(df.loc[-1*node]["id_tipo_propiedad"])
                y = f(df.loc[-1*node]["id_modalidad"])
            return dic_labeler[x, y]

        visits = visits.groupby(
            ["id_entidad", 'id_usuario']).size().reset_index()
        # evita coliciones de ID entre usuarios y propiedades
        visits["id_usuario"] = -1*visits["id_usuario"]
        self.B = nx.Graph()

        self.B.add_nodes_from(pd.unique(visits["id_entidad"]), bipartite=0)
        self.B.add_nodes_from(pd.unique(visits["id_usuario"]), bipartite=1)

        feat_dict = {node: [labeler(node, attrs["bipartite"])]
                     for node, attrs in self.B.nodes(data=True)}

        nx.set_node_attributes(self.B, values=feat_dict, name="label")

        if feats is not None:
            #nx.set_node_attributes(self.B, values=0, name="feature")
            pass

        self.B.add_weighted_edges_from(visits.values)
        # quitar nodos con grado menro a min_degree
        dangling = [node for node in self.B.nodes if self.B.degree(
            node) <= min_degree]
        self.B.remove_nodes_from(dangling)

        self.B = max(nx.connected_component_subgraphs(self.B), key=len)
        self.B.graph['tipo'] = "bipartite"

        self.prop_nodes = [n for n, d in self.B.nodes(
            data=True) if d['bipartite'] == 0]
        self.user_nodes = [n for n, d in self.B.nodes(
            data=True) if d['bipartite'] == 1]

        self.B_f = nx.Graph()
        self.Props = nx.Graph()
        self.Users = nx.Graph()

        self.Props_f = nx.Graph()
        self.Users_f = nx.Graph()

        self.filter_weights("bipartite", threshold=threshold,
                            power=power, filter_method=filter_method)

    def project(self, project_on, threshold=0, power=1, weight_function="jaccard",
                filter_method=backboning.disparity_filter):
        r"""
        Obtiene la proyecciones del grafo bipartito para usuarios o propiedades

        Parameters
        ----------

        project_on: str
            Proyectar en usuarios o propiedades
        threshold: float o None
            Umbral de filtro de enlaces, entre 0 y 1. para 0 no filtra y para 1 filtra todo los enlaces
        power: float
            Potencia a la que elevar los pesos de los enlaces.
        weight_function: str
            Funcion de peso para calcular peso de enlace, puede ser "jaccard" o "maximo
        filter_method: metodos de backboning
            Que metodo de filtrado usar. (recomiendo usar el por defecto)
        """

        function_dict = {"jaccard": fn.weighted_jaccard,
                         "maximo": fn.weighted_maximum}

        nodes_dict = {"users": self.user_nodes, "props": self.prop_nodes}
        nodes = nodes_dict[project_on]

        D = bipartite.projection.generic_weighted_projected_graph(
            self.B, nodes, weight_function=function_dict[weight_function])
        D = max(nx.connected_component_subgraphs(D), key=len)

        D.graph['tipo'] = project_on
        D.graph['function'] = weight_function
        if project_on == "users":
            mapping = {node: -1*node for node in D}
            self.Users = D
            self.Users = nx.relabel_nodes(self.Users, mapping=mapping)

        if project_on == "props":
            self.Props = D

        self.filter_weights(project_on, threshold=threshold,
                            power=power, filter_method=filter_method)

    def filter_weights(self, filter_on, threshold=None, power=1, filter_method=backboning.disparity_filter):
        r"""
        Filtra el grafo bipartito o una de sus proyecciones.

        Parameters
        ----------        
        filter_on: str
            Filtrar el grafo de usuarios, de propiedades o bipartito
        threshold: float o None
            Umbral de filtro de enlaces, entre 0 y 1. para 0 no filtra y para 1 filtra todo los enlaces
        power: float
            Potencia a la que elevar los pesos de los enlaces.
        filter_method: metodos de backboning
            Que metodo de filtrado usar. (recomiendo usar el por defecto)
        """

        # source: grafo original, filtered: grafo filtrado u objetivo
        dic_source = {"bipartite": self.B,
                      "users": self.Users, "props": self.Props}
        source = dic_source[filter_on]
        dic_filtered = {"bipartite": self.B_f,
                        "users": self.Users_f, "props": self.Props_f}
        filtered = dic_filtered[filter_on]

        # filtrado
        # D: grafo dummy

        if threshold is None:
            # si threshold es None, no hay que filtrar, y solo hay que actualizar la potencia de los pesos
            D = filtered

        if threshold is not None:
            # Para filtrar se le necesita entregar a backboning una dataframe con forma
            # src,trg, nij; donde src y trg son nodos y nij el peso
            D = source

            table = nx.to_pandas_edgelist(D)
            table.columns = ["src", "trg", "nij"]
            nc_table = filter_method(table, undirected=True)
            nc_backbone = backboning.thresholding(nc_table, threshold)
            nc_backbone.columns = ["src", "trg", "weight", "score"]
            D = nx.from_pandas_edgelist(nc_backbone, "src", "trg", "weight")

        # elevar
        for *edge, data in source.edges(data=True):
            if edge in D.edges():
                weight = data['weight']
                weight = weight**power
                D.add_edge(*edge, weight=weight)

        if len(D) != 0:
            D = max(nx.connected_component_subgraphs(D), key=len)

        D.graph['tipo'] = filter_on

        attrs_dict = {node: att for node, att in source.nodes(data=True)}
        # seteo
        if filter_on == "bipartite":
            self.B_f = D
            nx.set_node_attributes(self.B_f, values=attrs_dict)
            self.B_f.graph["threshold"] = threshold
            self.B_f.graph["power"] = power

        if filter_on == "users":
            self.Users_f = D
            nx.set_node_attributes(self.Users_f, values=attrs_dict)
            self.Users_f.graph["threshold"] = threshold
            self.Users_f.graph["power"] = power

        if filter_on == "props":
            self.Props_f = D
            nx.set_node_attributes(self.Props_f, values=attrs_dict)
            self.Props_f.graph["threshold"] = threshold
            self.Props_f.graph["power"] = power

    def get_histogram(self, hist_on, filtered=True):
        r"""
        Obtension del histograma de pesos de los enlaces

        Parameters
        ----------        
        hist_on: str
            histograma del grafo de usuarios, de propiedades o bipartito
        filtered: bool
            si es el caso filtrado o no

        """
        dic = {True: {"bipartite": self.B_f, "users": self.Users_f, "props": self.Props_f},
               False: {"bipartite": self.B, "users": self.Users, "props": self.Props}}
        D = dic[filtered][hist_on]

        hist = [edge[2]['weight'] for edge in D.edges(data=True)]
        return hist

    def set_features_from_graph(self, on, filtered=True, mode="adj", name="feature"):
        r"""
        Setea caracteristicas de un nodo a partir de propiedades del grafo: matriz de adyacencia, 
        onehot vector o topologicas.

        Parameters
        ----------        
        on: str
            Setear en grafo de usuarios, de propiedades o bipartito.
        filtered: bool
            Si es el caso filtrado o no.
        mode: str
            Que caracteristica asignarle: adj, onehot, topos.
        name: str
            Nombre a la que llamar a la caracteristica.
        """

        dic = {True: {"bipartite": self.B_f, "users": self.Users_f, "props": self.Props_f},
               False: {"bipartite": self.B, "users": self.Users, "props": self.Props}}

        D = dic[filtered][on]

        if mode == "onehot":
            def dirac(i, j):
                if i == j:
                    return 1
                else:
                    return 0

            for i, node in enumerate(D.nodes):
                D.nodes[node][name] = [dirac(i, j) for j in range(len(D))]

        elif mode == "adj":
            A = nx.adjacency_matrix(D).todense().A

            for i, node in enumerate(D.nodes):
                D.nodes[node][name] = list(A[i])

        elif mode == "topos":

            bb = nx.betweenness_centrality(D, k=100)
            deg = nx.degree(D)
            pr = nx.pagerank(D)
            clust = nx.clustering(D)
            eig = nx.eigenvector_centrality(D)

            for i, node in enumerate(D.nodes):
                D.nodes[node][name] = [bb[node], deg[node],
                                       pr[node], clust[node], eig[node]]

        #nx.set_node_attributes(self.B, values=0, name="feature")

    def set_features_from_df(self, on, filtered=True, df=pd.DataFrame(), name="feature"):
        r"""
        Setea caracteristicas de un nodo a partir de un dataframe

        Parameters
        ----------        
        on: str
            Setear en grafo de usuarios, de propiedades o bipartito.
        filtered: bool
            Si es el caso filtrado o no.
        df: DataFrame
            Dataframe con indices los nodos y columnas los valores que asignarles.
        name: str
            Nombre a la que llamar a la caracteristica.
        """

        dic = {True: {"bipartite": self.B_f, "users": self.Users_f, "props": self.Props_f},
               False: {"bipartite": self.B, "users": self.Users, "props": self.Props}}

        D = dic[filtered][on]

        for i, node in enumerate(D.nodes):
            D.nodes[node][name] = df.loc[node].values
        #nx.set_node_attributes(self.B, values=0, name="feature")

    def set_graphs(self, G, on, filtered):
        r"""
        Setea los grafos a unos que ya se han obtenido.

        Parameters
        ----------        
        G: NetworkX Graph
            Grafo con que se va a substituir
        on: str
            Grafo de usuarios, de propiedades o bipartito.
        filtered: bool
            Si es el caso filtrado o no.
        """
        if filtered:
            if on == "bipartite":
                self.B_f = G
            if on == "users":
                self.Users_f = G
            if on == "props":
                self.Props_f = G

        if not filtered:
            if on == "bipartite":
                self.B = G

                self.prop_nodes = {n for n, d in self.B.nodes(
                    data=True) if d['bipartite'] == 0}
                self.user_nodes = {n for n, d in self.B.nodes(
                    data=True) if d['bipartite'] == 1}

            if on == "users":
                self.Users = G
            if on == "props":
                self.Props = G
    
    def find_best_filter(self,on):
        pass
    def find_best_power(self,on):
        pass

    def plot_histogram(self, on, filtered=True, weighted=True, **kwargs):
        r"""
        Ploteo del histograma de enlaces de un grafo

        Parameters
        ----------        
        on: str
            Histograma del grafo de usuarios, de propiedades o bipartito.
        filtered: bool
            Si es el caso filtrado o no.
        weighted: bool            
            Si el grafico debe estar normalizado. (deberia se True la mayoria de las veces)
        """
        dic_name={"users":"Usuarios","props":"Propiedades","bipartite": "Bipartito"}


        hist=self.get_histogram(on,filtered)
        

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
        return fig, ax, histo
        
    def plot_std(self, on,  powers=np.linspace(0,1,11)):
        r"""
        Ploteo de la desviacion estandar de los pesos de enlaces de un grafo, para distintas potencias

        Parameters
        ----------        
        on: str
            Histograma del grafo de usuarios, de propiedades o bipartito.
        powers: Array np.
            Potencias que plotear:
            
        """
        dic_name={"users":"Usuarios","props":"Propiedades","bipartite": "Bipartito"}
        std=[]
        for p in tqdm(powers):
            self.filter_weights(on,power=p)
            hist=self.get_histogram(on,filtered=True)

            std.append(np.std(hist))
            
        fig, ax = plt.subplots()
        plot=ax.plot(powers,std)
        ax.set_title("Desviacion Estandar "+dic_name[on])
        ax.set_xlabel("p")
        ax.set_ylabel("Desviacion Estandar")

        return fig, ax, plot
    
    def plot_filters(self,on, filters=None):
        r"""
        Ploteo de la covertura vs enlaces relativos para distintos valores del filtro.

        Parameters
        ----------        
        on: str
            Histograma del grafo de usuarios, de propiedades o bipartito.
        filters: Array np.
            Filtros que plotear, recomiendo que este en escala logaritmica y que
            contenga al 1 y al 0
        """
        if filters is None:
            filters=np.logspace(-1,0,num=9)
            filters=np.insert(filters,0,0)
            filters=1-filters

        tipo_dict={"bipartite":self.B,"users":self.Users,"props":self.Props}
        dic_name={"users":"Usuarios","props":"Propiedades","bipartite": "Bipartito"}

        edgtotal=len(tipo_dict[on].edges)
        nodetotal=len([node for node in tipo_dict[on].nodes if tipo_dict[on].degree(node) > 1])
                
        coverages=[]
        edges=[]
        nodes=[]

        for alpha in tqdm(filters):
            self.filter_weights(on,threshold=alpha)

            tipo_f_dict={"bipartite":self.B_f,"users":self.Users_f,"props":self.Props_f}
            non_isolated=len([node for node in tipo_f_dict[on].nodes if tipo_dict[on].degree(node) > 1])
            coverage=non_isolated/nodetotal
            edges.append(len(tipo_f_dict[on].edges))
            nodes.append(len(tipo_f_dict[on]))
            coverages.append(coverage)
    
        edge_relativo=[i/edgtotal for i in edges]
        print(edge_relativo,coverages)
        fig, ax = plt.subplots()   
        scatter=plt.scatter(edge_relativo,coverages,c=filters, cmap="winter")
        plot=ax.plot(edge_relativo,coverages,linewidth=1,linestyle='dashed')

        ax.set_title("Filtrado "+dic_name[on])
        ax.set_xlabel("Enlaces Relativos")
        ax.set_ylabel("Cobertura")
        plt.colorbar()
        return fig, ax, scatter, plot
                
def add_user_attributes(users, props, visits, std=False):
    r"""
    Asignar caracteristicas de usuarios a partir de las propiedades que visito

    Parameters
    ----------        
    users: DataFrame
        Dataframe de usuarios.
    props: DataFrame
        Dataframe de propiedades.
    visits: DataFrame
        Dataframe de visitas.
    std: bool
        Si añadir las desviaciones estandar de las caracteristicas
    """
    visits_labeled = visits.merge(
        props, left_on="id_entidad", right_index=True)

    features = list(visits_labeled.columns)
    features.remove("id_entidad")
    features.remove("id_usuario")
    users_labeled = users
    for feat in features:
        grouped_visits_labeled = visits_labeled.groupby(
            "id_usuario")[feat].apply(np.mean).to_frame()
        users_labeled = users_labeled.merge(
            grouped_visits_labeled, right_index=True, left_index=True)
        if std:
            grouped_visits_labeled = visits_labeled.groupby(
                "id_usuario")[feat].apply(np.std).to_frame()
            users_labeled = users_labeled.merge(
                grouped_visits_labeled, right_index=True, left_index=True, suffixes=(None, '_std'))
    users = users_labeled
    visits = visits_labeled
    users["6h3"] = users.apply(lambda x: h3.geo_to_h3(
        x['ubicacion_latitud'], x['ubicacion_longitud'], 6), axis=1)
    users["8h3"] = users.apply(lambda x: h3.geo_to_h3(
        x['ubicacion_latitud'], x['ubicacion_longitud'], 8), axis=1)
    users["10h3"] = users.apply(lambda x: h3.geo_to_h3(
        x['ubicacion_latitud'], x['ubicacion_longitud'], 10), axis=1)
    return users


