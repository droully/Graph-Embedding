#%%
"""imports"""
import itertools
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from openne.models import gae, gf, grarep, lap, line, lle, node2vec, sdne, vgae
from tqdm import tqdm

from evalne.evaluation.evaluator import LPEvaluator, NCEvaluator
from evalne.evaluation.score import Scoresheet
from evalne.evaluation.split import EvalSplit
from evalne.utils import preprocess as pp
from openne.dataloaders import Graph, create_self_defined_dataset

# %%
"""

users = pickle.load(open("data/users.p", "rb"))

visits = pickle.load(open("data/visits.p", "rb"))
props = pickle.load(open("data/props.p", "rb"))

grafos = pickle.load(open("data/grafos.p", "rb"))
grafos_test = pickle.load(open("data/grafos_test.p", "rb"))

nodesnuevos=[n for n in grafos_test.B_f if n not in grafos.B_f]
grafos_test.B_f.remove_nodes_from(nodesnuevos)
edgesnuevos=[e for e in grafos_test.B_f.edges if e not in grafos.B_f.edges]
grafos_test.B_f.remove_edges_from(list(grafos_test.B.edges))
grafos_test.B_f.add_edges_from(edgesnuevos)

grafos.set_features("users",mode="adj")
grafos.set_features("props",mode="adj")
grafos.set_features("bipartite",mode="adj")
"""



# %%

class Tuning:
    r"""
    Clase general de entrenamiento 
    """

    def __init__(self,G,root):
        self.tipo = G.graph["tipo"]
        self.root = root
        self.methods_modules_dict = {
            "line": line.LINE, "node2vec": node2vec.Node2vec,"gf":gf.GraphFactorization,"lap":lap.LaplacianEigenmaps,
            "sdne": sdne.SDNE,"lle":lle.LLE,"grarep":grarep.GraRep,"gae":gae.GAE,"vgae":vgae.VGAE}
        self.method_dims_dict = {"line": "dim","gf": "dim", "node2vec": "dim", "lap":"dim","sdne":"encoder_layer_list","lle":"dim",
            "grarep":'dim',"gae":"output_dim","vgae":"output_dim"}
        self.methods_params_names_dict = {"line": ["dim", "negative_ratio","lr"],
                                          "node2vec": ["dim", "path_length", "num_paths",   "p", "q"],
                                          "gf":["dim"],
                                          "lap":["dim"],
                                          "sdne":["encoder_layer_list","alpha", "beta", "nu1","nu2"],
                                          "lle":["dim"],
                                          'grarep':["dim","kstep"],
                                          'gae':["output_dim","hiddens","max_degree"],
                                          'vgae':["output_dim","hiddens","max_degree"]
                                          }
        self.methods_params_types_dict = {"line": {"dim": "nat", "negative_ratio": "nat","lr":"ratio"},
                                          "node2vec": {"dim": "nat", "path_length": "nat", "num_paths": "nat",  "p": "ratio", "q": "ratio"},
                                          "gf":{"dim":"nat"},
                                          "lap":{"dim":"nat"},
                                          "sdne":{"encoder_layer_list":"ratio","alpha":"ratio", "beta":"gtone", "nu1":"ratio","nu2": "ratio"},
                                          "lle":{"dim":"nat"},
                                          "grarep":{"dim":"nat","kstep":"nat"},
                                          'gae':{"output_dim":"nat","hiddens":"ratio","max_degree":"nat"},
                                          'vgae':{"output_dim":"nat","hiddens":"ratio","max_degree":"nat"}
                                          }

        self.training_graph =None
        self.test_graph= None

        self.labels = np.array([[node,attrs["label"][0]] for node, attrs in G.nodes(data=True)])
        self.task=None
        self.scoresheet=Scoresheet()
        self.evaluator=None
        self.method=None

    def score(self, method_name,edge_method="hadamard"):
        r"""    
        metodo que entrega la metrica guardada en el scoresheet; para NC f1 ponderado, para LP auroc

        Parameters
        ----------
        method_name: str
            nombre del metodo
        edge_method: str
            metodo de embedding de enlace: "l1","l2","hadamard","average"
        """
        if self.task=="nc":
            return np.mean(self.scoresheet._scoresheet["GPI"][method_name+"_0.8"]["f1_weighted"])

        if  self.task=="lp":
            dic_edge_method={"l1":0,"l2":1,"hadamard":2,"average":3}
            # ['weighted_l1', 'weighted_l2', 'hadamard', 'average']
            return np.mean(self.scoresheet._scoresheet["GPI"][method_name]["auroc"][dic_edge_method[edge_method]])

    def time(self, method_name):
        r"""    
        metodo que entrega el tiempo guardado en el scoresheet

        Parameters
        ----------
        method_name: str
            nombre del metodo
        """
        
        if self.task=="nc":
            return self.scoresheet._scoresheet["GPI"][method_name+"_0.8"]['eval_time'][0]

        if  self.task=="lp":
            # ['weighted_l1', 'weighted_l2', 'hadamard', 'average']
            return self.scoresheet._scoresheet["GPI"][method_name]["eval_time"][0]

    def HCgen(self, seed, scale):
        r"""    
        generador de vecinos para hill climbing
        
        Parameters
        ----------
        seed: dict
            diccionario semilla, donde las llaves son los parametros y los valores sus valores
        scale: dict
            diccionario escala, donde las llaves son los parametros y los valores que tanto
            multiplicar o dividir 
        """
        
        aux = seed.copy()
        yield aux
        for k, v in seed.items():
            if k not in scale:
                continue

            if self.methods_params_types_dict[self.method][k] == "nat":
                aux.update({k: v+scale[k]})
                yield aux
                aux = seed.copy()

                if v-scale[k] > 0:
                    aux.update({k: v-scale[k]})
                    yield aux
                    aux = seed.copy()
            if self.methods_params_types_dict[self.method][k] == "ratio":
                aux.update({k: v*scale[k]})
                yield aux
                aux = seed.copy()

                aux.update({k: v/scale[k]})
                yield aux
                aux = seed.copy()
                
            if self.methods_params_types_dict[self.method][k] == "gtone":
                aux.update({k: v+scale[k]})
                yield aux
                aux = seed.copy()
                if v-scale[k] > 1:
                    aux.update({k: v-scale[k]})
                    yield aux
                    aux = seed.copy()
  
    def TestModel(self,*args):
        return 0

    def TrainModel(self, method,  d, savefile=None, **kwargs):
        r"""
        Entrena un modelo de embeddings usando un metodo  dado por OpenNE

        OJO IMPORTANTE: 
        algunos modelos sus hiperparametros se les da en la definicion de la instancia:
        model=loquesea(parametro)
         
        y otros cuando se calculan los embeddings: vectors = model(parametro)

        cual  caso es depende del metodo y del parametro!!! (culpa de openNE)

        Parameters
        ----------
        method: str
            Metodo de embedding a usar, puede ser "line","node2vec","gf","lap","sdne",'grarep','gae','vgae'.
        d: int
            Dimension del embedding.
        savefile: str
            Direccion de guardado del embedding, si es None no se guarda.
        **kwargs: dict
            Parametros para el metodo.
            
        Returns
        -------
        emb: dict
            Diccionario donde las llaves son los nodos y los valores su embedding.
        time: float
            Tiempo de entrenamiento.
        """
        start = time.time()


        if method=="node2vec" and not nx.is_directed(self.training_graph.G) :

            dictostr = {node: str(node) for node in self.training_graph.G}
            self.training_graph.G = nx.relabel_nodes(self.training_graph.G, dictostr)
            self.training_graph.G = self.training_graph.G.to_directed()

            model = self.methods_modules_dict[method](dim=d,save=False, **kwargs)
            vectors = model(self.training_graph,**kwargs)

            dictoint = {node: int(node) for node in self.training_graph.G}
            self.training_graph.G = nx.relabel_nodes(self.training_graph.G, dictoint)
            self.training_graph.G = self.training_graph.G.to_undirected()



        elif method=="grarep":
            model = self.methods_modules_dict[method](kstep=kwargs["kstep"],dim=d,save=False)
            vectors = model(self.training_graph,dim=d,**kwargs)

        elif method=="gae" or method=="vgae" :   
            kwargs["hiddens"]=[int(kwargs["hiddens"])]

            model = self.methods_modules_dict[method](output_dim=d,dim=d,save=False,**kwargs)
            vectors = model(self.training_graph,epochs=kwargs["epochs"])

        elif method=="sdne":   
            kwargs["encoder_layer_list"]=[int(kwargs["encoder_layer_list"]),d]
            model = self.methods_modules_dict[method](**kwargs,save=False)
            vectors = model(self.training_graph,**kwargs)

        else:   
            model = self.methods_modules_dict[method](dim=d,save=False, **kwargs)
            vectors = model(self.training_graph,**kwargs)

        end = time.time()


        emb = {str(k): np.array(v) for k, v in vectors.items()}
        if savefile is not None:
            model.save_embeddings(savefile)
        return emb, end-start

    def TabuSearchParams(self, method, dim, seed={}, scale={}, iters=2, tabu_lenght=2, **kwargs):
        r"""

        Busca los mejores hiperparametros de un modelo aplicando lista tabu
    
        ----------
        method: str
            Metodo de embedding a usar, puede ser "line","node2vec","gf","lap","sdne",'grarep','gae','vgae'.
        dim: int
            Dimension del embedding.
        seed: dict
            Diccionario con los hiperparametros con los que se inicia la busqueda, las llaves son el nombre y los valores su valor
        scale: dict
            Diccionario con la escala de cada hiperparametro con la que se determinara el 
        iters: int
            Numero de iteraciones de la lista tabu
        tabu_lenght: int
            Largo de la lista tabu
        **kwargs: dict
            Parametros extra para el modelo que no se tunearan
        """
        self.method = method
        self.path = "{}{}/{}/".format(self.root, self.tipo, self.method)

        os.makedirs(self.path, exist_ok=True)
        scorehist = pd.DataFrame(columns=self.methods_params_names_dict[self.method]+['f1'])
        
        if self.method in ["gf", "lap", "lle"]:
            iters=1 
            
        if self.method=="node2vec":
            dictostr = {node: str(node) for node in self.training_graph.G}
            self.training_graph.G = nx.relabel_nodes(self.training_graph.G, dictostr)
            self.training_graph.G = self.training_graph.G.to_directed()
                    
        try:
            self.scoresheet = pickle.load(open(self.path+"scorenc", "rb"))
        except:
            self.scoresheet = Scoresheet()

        tabu_list = tabu_lenght*[" "]

        best = seed
        best_score=0

        bestCandidate = seed
        bestCandidateValues = tuple(bestCandidate.values())
        #tabu_list.append(bestCandidateValues)
        #tabu_list = tabu_list[1:]

        score_dict = {(-1,): 0}
        for _ in range(iters):
            bestCandidate_ = bestCandidate
            bestCandidate = {"dummy": -1}           
            bestCandidateValues=tuple(bestCandidate.values())
            bestCandidateScore=score_dict[bestCandidateValues]
            
            for candidate in self.HCgen(bestCandidate_, scale):
                candidateValues = tuple(candidate.values())
                params_str = " {:n}"*(len(candidateValues)+1)
                params_str = params_str.format(dim, *candidateValues)
                params_str=params_str[1:]
                method_name ="{} {}".format(self.method,params_str) 
                try:
                    score_dict.update({candidateValues: self.score(method_name)})
                except:

                    args = candidate.copy()
                    args.update(kwargs)

                    emb,time=self.TrainModel(self.method,  dim, self.path+params_str+".txt", **args)
                    res = self.TestModel(emb,time, method_name)
                    #self.scoresheet.log_results(res)
                    self.scoresheet.write_pickle(self.path+"scorenc")

                    try:
                        os.remove(self.path+"scorenc.txt")
                    except:
                        pass
                    self.scoresheet.write_all(self.path+"scorenc.txt",repeats="all")

                    score_dict.update({candidateValues: self.score(method_name)})
                if (candidateValues not in tabu_list) and (score_dict[candidateValues] >= bestCandidateScore):
                    bestCandidate = candidate.copy()
                    bestCandidateValues=tuple(bestCandidate.values())
                    bestCandidateScore=score_dict[tuple(bestCandidate.values())]

            scorehist = scorehist.append({**{self.method_dims_dict[self.method]:dim}, **bestCandidate, **{"f1": bestCandidateScore}}, ignore_index=True)

            if score_dict[bestCandidateValues] > best_score:
                best = bestCandidate
                best_score = score_dict[tuple(best.values())]

            tabu_list.append(bestCandidateValues)
            tabu_list = tabu_list[1:]



            types2format = {"nat": '%d',"gtone": '%d', "ratio": '%E'}
            fmt = [types2format[self.methods_params_types_dict[self.method][v]]
                   for v in self.methods_params_names_dict[self.method]]+['%1.6f']
            np.savetxt(self.path+str(dim)+" scorehist.txt",scorehist.values, fmt=fmt)


        if self.method=="node2vec":
            dictoint = {node: int(node) for node in self.training_graph.G}
            self.training_graph.G = nx.relabel_nodes(self.training_graph.G, dictoint)
            self.training_graph.G = self.training_graph.G.to_undirected()


        bestvalues=tuple(best.values())
        params_str = " {:n}"*(len(bestvalues)+1)
        params_str = params_str.format(dim, *bestvalues)
        params_str=params_str[1:]
        best_method_name="{} {}".format(self.method,params_str) 
        best_time= self.time(best_method_name)
        if self.task=="nc":
            return best, best_score, best_time
        if self.task=="lp":
            best_l1=self.score(best_method_name,edge_method="l1")
            best_l2=self.score(best_method_name,edge_method="l2")
            best_hadamard=self.score(best_method_name,edge_method="hadamard")
            best_average=self.score(best_method_name,edge_method="average")
            return best, best_l1, best_l2, best_hadamard, best_average, best_time


class LinkPredictionTuning(Tuning):
    r"""

    Clase general de entrenamiento y testeo de embeddings de grafos para la tarea de prediccion de enlaces.

    Parameters
    ----------
    G: NetworkX graph
        Grafo de entrenamiento.
    G_test: NetworkX graph
        Grafo de testeo.
    root: str
        directorio en el que se guardaran los resultados
    """
    def __init__(self, G,G_test, root="results/lp/"):
        super(LinkPredictionTuning, self).__init__(G, root=root)        
        self.task="lp"

        train_E=G.edges
        train_E_false=self.GetNegativeEdges(G,len(train_E))

        test_E=G_test.edges
        test_E_false=self.GetNegativeEdges(G_test,len(test_E))
        
        self.split = EvalSplit()
        self.split.set_splits(train_E, train_E_false=train_E_false, test_E=test_E, test_E_false=test_E_false, TG=G)

        self.training_graph = create_self_defined_dataset(root_dir="",name_dict={},name="training "+self.tipo, weighted=True, directed=False, attributed=True)()
        self.training_graph.set_g(G)

        self.evaluator = LPEvaluator(self.split)

    def GetNegativeEdges(self, G, n):
        r"""

        Metodo auxiliar que muestrea enlaces negativos.

        Parameters
        ----------
        G: NetworkX graph
           Grafo bipartito.
        n: int
            cantidad de enlaces que muestrear.
        """

        prop_nodes=[n for n, d in G.nodes(data=True) if d['bipartite']==0]
        user_nodes=[n for n, d in G.nodes(data=True) if d['bipartite']==1]

        non_edges=[]

        while len(non_edges) <=n:
            random_prop = random.choice(prop_nodes)
            random_user = random.choice(user_nodes)
            edge=(random_prop,random_user) 
            if G.has_edge(*edge):
                continue
            else: 
                non_edges.append(edge)
        return non_edges

    def TestModel(self, emb,time=-1, method_name="method_name"):
        r"""

        Testea un embedding y lo guarda en el scoresheet.

        Parameters
        ----------
        emb: dict
            diccionario de embeddings, llaves son los nodos y los valores una lista con el embedding
        time: float
            tiempo de ejecucion del metodo, para guardar en el scoresheet
        method_name: str
            nombre del metodo con el que guardar.
        """
        df = pd.DataFrame(emb).T
        X = df.T.to_dict("list")
        X = {str(k): np.array(v) for k, v in X.items()} # tiene que ser array por que se hacen sumas      


        self.evaluator.dim=df.shape[1]

        reslp=[]
        for edge_method in ["weighted_l1","weighted_l2","hadamard","average"]:
            #TO DO que no evalue en los 4 embeddings de enlaces
            res = self.evaluator.evaluate_ne(self.split, X=X,method=method_name,edge_embed_method=edge_method,params={"nw_name":"GPI"})
            res.params.update({'eval_time': time})
            reslp.append(res)
        self.scoresheet.log_results(reslp)
        return reslp

class NodeClassificationTuning(Tuning):
    r"""

    Clase general de entrenamiento y testeo de embeddings de grafos para la tarea de clasificacion de nodos.

    Parameters
    ----------
    G: NetworkX graph
        Grafo.
    root: str
        directorio en el que se guardaran los resultados
    """
    def __init__(self, G, root="results/nc/",**kwargs):
        super(NodeClassificationTuning, self).__init__(G, root=root)
        self.task="nc"
       
        self.training_graph=create_self_defined_dataset(root_dir="",name_dict={},name="test "+self.tipo, weighted=True, directed=False, attributed=True)()
        self.training_graph.set_g(G)

        self.evaluator = NCEvaluator(self.training_graph.G, self.labels, nw_name="GPI",
                               num_shuffles=5, traintest_fracs=[0.8], trainvalid_frac=0)

    def TestModel(self, emb,time=-1, method_name="method_name"):
        r"""

        Testea un embedding y lo guarda en el scoresheet.

        Parameters
        ----------
        emb: dict
            diccionario de embeddings, llaves son los nodos y los valores una lista con el embedding
        time: float
            tiempo de ejecucion del metodo, para guardar en el scoresheet
        method_name: str
            nombre del metodo con el que guardar.
        """
        df = pd.DataFrame(emb).T
        X = df.T.to_dict("list")
        X = {str(k):  np.array(v) for k, v in X.items()}# tiene que ser array por que se hacen sumas   

        self.evaluator.dim=df.shape[1]
        
        resnc = self.evaluator.evaluate_ne(X=X, method_name=method_name)
        for res in resnc:
            res.params.update({'eval_time': time})
            
        self.scoresheet.log_results(resnc)
        return resnc


# %%
def tabu_search(G,task, method,G_test=None,seed={}, scale={}, dims=[10, 30, 50, 100, 300, 500],**kwargs):
    r"""
    funcion auxiliar que repite la busqueda tabu para un grafo, tarea, metodo, semilla/escala, y dimensiones
    especificas

    Parameters
    ----------
    G: NetworkX graph
        Grafo.
    task: str
        tarea: "nc" o "lp"
    G_test: NetworkX graph
        Grafo de testeo (solo para lp)
    seed: dict
        diccionario de parametros semilla
    scale: dict
        diccionario de parametros escala
    dims: list
        lista de dimensiones con las probar
    **kwargs
        parametros de los metodos que no se quieren tunera
    """
    if task =="nc":
        tester=NodeClassificationTuning(G)
        
        df=pd.DataFrame(columns=["name","score","time"])
    if task =="lp":
        tester=LinkPredictionTuning(G,G_test)
        df=pd.DataFrame(columns=["name","l1","l2","hadamard","average","time"])

    best_dict={}
    for d in dims:
        if task =="nc":
            best, best_f1, best_time=tester.TabuSearchParams(method=method,dim=d,seed=seed,scale=scale,**kwargs)
            best_dict.update({"name":best,"score":best_f1,"time":best_time})
        if task =="lp":
            best, best_l1, best_l2, best_hadamard, best_average, best_time=tester.TabuSearchParams(method=method,dim=d,seed=seed,scale=scale,**kwargs)
            best_dict.update({"name":best,"l1":best_l1,"l2":best_l2,"hadamard":best_hadamard,"average":best_average,"time":best_time})

        df=df.append(best_dict,ignore_index=True)
    df.index=dims
    
    df.to_csv("results/"+task+"/"+G.graph["tipo"]+"/"+method+"/dimf1.csv")
    return df

#%%
"""
for grafo,task in [(grafos.Users_f,"nc")]:
    tabu_search(grafo,task,"gae",seed={"hiddens":128,"max_degree":0},scale={"hiddens":2,"max_degree":1},iters=2,epochs=200)
    tabu_search(grafo,task,"vgae",seed={"hiddens":128,"max_degree":0},scale={"hiddens":2,"max_degree":1},iters=2,epochs=200)
    #tabu_search(grafo,task,"grarep",seed={"kstep": 5}, iters=1)
    #tabu_search(grafo,task,"gf")
    #tabu_search(grafo,task,"lap")
    #tabu_search(grafo,task,"node2vec",seed={"path_length": 20, "num_paths": 10,  "p": 0.1, "q": 0.1},scale={"path_length": 5, "num_paths": 5,  "p": 10, "q": 10},iters=2, window=4)
    #tabu_search(grafo,task,"line",seed={"negative_ratio": 15,"lr":0.001 }, iters=1,epochs=10)
    #tabu_search(grafo,task,"sdne",seed={"encoder_layer_list":128,"alpha":1e-6, "beta":10, "nu1":1e-8,"nu2": 1e-4},scale={"encoder_layer_list":2,"alpha":10, "beta":5, "nu1":10,"nu2":10},epochs=200, iters=3)
    
#%%
tabu_search(grafos.B_f,"lp","lap",G_test=grafos_test.B_f)
tabu_search(grafos.B_f,"lp","node2vec",seed={"path_length": 20, "num_paths": 10,  "p": 0.1, "q": 0.1},scale={"path_length": 5, "num_paths": 5,  "p": 10, "q": 10}, window=4, G_test=grafos_test.B_f)
tabu_search(grafos.B_f,"lp","line",seed={"negative_ratio": 15,"lr":0.001 }, iters=1,epochs=10,G_test=grafos_test.B_f)
tabu_search(grafos.B_f,"lp","gf",G_test=grafos_test.B_f)
tabu_search(grafos.B_f,"lp","sdne",seed={"encoder_layer_list":128,"alpha":1e-6, "beta":10, "nu1":1e-8,"nu2": 1e-4},scale={"encoder_layer_list":2,"alpha":10, "beta":5, "nu1":10,"nu2":10},epochs=200, iters=3,G_test=grafos_test.B_f)
tabu_search(grafos.B_f,"lp","grarep",seed={"kstep": 5}, iters=1,G_test=grafos_test.B_f)
# %%

tabu_search(grafos.B_f,"lp","gae",seed={"hiddens":128,"max_degree":0},scale={"hiddens":2,"max_degree":1},G_test=grafos_test.B_f, epochs=200,iters=1)
tabu_search(grafos.B_f,"lp","vgae",seed={"hiddens":128,"max_degree":0},scale={"hiddens":2,"max_degree":1},G_test=grafos_test.B_f,epochs=200, iters=1)


"""
# %%
