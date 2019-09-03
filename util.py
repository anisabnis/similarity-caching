""" This file contains the various utilities which can be used to simulate a similarity cache """

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
from lshash import *

# new random covar
def new_random_cov(n):
    if n == 1:
        return random.uniform(0.05, 0.1)

    # random matrix generation
    A = []
    for i in range(0, n):
        A.append([])
        for j in range(0, n):
            A[i].append([])

    for i in range(0, n):
        for j in range(0, n):
            A[i][j] = np.random.triangular(-0.1, 0, 0.1)

    A = np.array(A)
    B = np.matmul(A, A.transpose())
    C = np.zeros((n, n))

    for i in range(0, n):
        diago = 1 / math.sqrt(B[i][i])
        C[i][i] = diago
    first_mol = np.matmul(C, B)
    C = np.matmul(first_mol, C)
    D = np.zeros((n, n))

    for i in range(0, n):
        diago = np.random.uniform(0.05, 0.1)
        D[i][i] = diago
    first_mol = np.matmul(D, C)
    res = np.matmul(first_mol, D)
    return res


""" All attributes of a cache object """
class CacheObject:
    counter = 0

    def __init__(self, id, vec, size=1):
        CacheObject.counter += 1
        self.id = CacheObject.counter
        self.pos = vec
        self.size = size


""" All attributes and functions of a cache """
class Cache:
    def __init__(self, capacity, dim, learning_rate, integral=False, grid_s=[313,313]):
        #self.cache = np.random.rand(capacity, dim)

        self.cache = {}
        self.grid = grid_s

        for index in range(capacity):
            if integral == False:
                v = np.random.rand(dim)
                self.cache[index] = v
            else:
                ii = np.random.randint(0, grid_s[0])
                jj = np.random.randint(0, grid_s[1])
                self.cache[index] = np.array([ii,jj])

        self.alpha = learning_rate
        self.capacity = capacity
        self.initializeLSH(dim)        

    def getAllPoints(self):
        return self.lsh.get_all_points()
        
    def initializeLSH(self, dim):        
        self.lsh = LSHash(3, dim, 10)
        for index in range(self.capacity):
            v = self.cache[index]
            self.lsh.index(v)

    def updateCache(self, src_obj_pos, dst_obj):
        self.lsh.delete_vector(src_obj_pos)
        self.lsh.index(dst_obj.pos)
    
    def updateCacheDict(self, src_obj_pos, dst_obj):
        self.updateCache(src_obj_pos, dst_obj)
        #self.cache.pop(src_object_id, None)
        #self.cache[dst_obj.id] = dst_obj.pos

    def findNearest(self, vec):
        ## Loop through the cache and find the nearest
        nearest_point = []
        min_dst = 10000
        min_id = 0

        for index in self.cache:
            if np.linalg.norm(self.cache[index] - vec) < min_dst:
                nearest_point = self.cache[index]
                min_dst = np.linalg.norm(self.cache[index] - vec)
                min_id = index

        return [nearest_point, min_id]            
            

    def findNearestANN(self, vec):
        K = self.lsh.query(vec)
        nearest_point = K[0][0]
        min_dst = K[0][1]
        return nearest_point


""" Object Catalogue """
class ObjectCatalogueUniform:
    def __init__(self, no_objects, alpha, dim):
        
        self.catalogue = []

        ## generate objects at random locations        
        for i in range(no_objects):
            loc = np.random.rand(dim)
            c_obj = CacheObject(i, loc)
            self.catalogue.append(c_obj)

        N = no_objects
        x = np.arange(1, N)
        a = alpha
        weights = x ** (-a)
        weights /= weights.sum()
        self.bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))

    def objective(self, cache):
        obj = 0
        for c_obj in self.catalogue:
            min_dst = 10
            for c in cache:
                if np.linalg.norm(c_obj.pos - cache[c]) < min_dst:
                    min_dst =  np.linalg.norm(c_obj.pos - cache[c])

            obj += min_dst
        return obj
    
    def objective_l1(self, cache):
        return 0

    def getRequest(self):
        obj = self.catalogue[self.bounded_zipf.rvs(size=1)[0]]
        return obj
      

class ObjectCatalogueGrid:
    def __init__(self, dim_x, dim_y):
        self.catalogue = []
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.means = []

        self.obj_id = 0

        for i in range(dim_x):
            for j in range(dim_y):

                pos = np.array([i,j])
                c_obj = CacheObject(self.obj_id, pos)                
                self.catalogue.append(c_obj)
                self.means.append(pos)
                self.obj_id += 1
                
    def getRequest(self):
        index = np.random.randint(0, self.obj_id)
        obj = self.catalogue[index]
        return obj
        
    def objective(self, cache):
        obj = 0
        for c_obj in self.catalogue:
            min_dst = 100000
            for c in cache:
                if np.linalg.norm(c_obj.pos - c) < min_dst:
                    min_dst =  np.linalg.norm(c_obj.pos - c)
            obj += min_dst
        return obj


    def objective_l1(self, cache):
        obj = 0
        for c_obj in self.catalogue:
            K = cache.lsh.query(c_obj.pos)
            obj += K[0][1]
            #min_dst = 10000
            #for c in cache:
            #    if sum(abs(c_obj.pos - c)) < min_dst:
            #        min_dst = sum(abs(c_obj.pos - c))
            #obj += min_dst
            
        return obj

  
""" Object Catalogue """
class ObjectCatalogueGaussian:
    def __init__(self, no_objects, centers, dim):
        
        self.catalogue = []
        self.means = []
        self.covs = []
        self.no_objects = no_objects

        for i in range(centers):
            v = np.random.uniform(low=0.1, high=0.9, size=(dim,))
            c = new_random_cov(dim)
            self.means.append(v)
            self.covs.append(c)

        for i in range(no_objects):
            index = np.random.randint(0, centers)
            loc = np.random.multivariate_normal(self.means[index], self.covs[index])
            c_obj = CacheObject(i, loc)
            self.catalogue.append(c_obj)

    def getRequest(self):
        index = np.random.randint(0, self.no_objects)
        obj = self.catalogue[index]
        return obj

    def objective(self, cache):
        obj = 0
        for c_obj in self.catalogue:
            min_dst = 10
            for c in cache:
                if np.linalg.norm(c_obj.pos - cache[c]) < min_dst:
                    min_dst =  np.linalg.norm(c_obj.pos - cache[c])
            obj += min_dst
        return obj

""" DescentMethods """
class StochasticGradientDescent:
    def __init__(self, learning_rate):
        self.alpha = learning_rate

    def descent(self, nearest_object, current_object):

        def derivative_l2(nearest_object, current_object):
            return 2 * (nearest_object - current_object)

        def derivative_l1(nearest_object, current_object):            
            return np.array([-1 if nearest_object[i] - current_object[i] < 0 else 1 for i in range(len(nearest_object))])

        d = derivative_l1(nearest_object, current_object)
        n = nearest_object - self.alpha * d
        return n


""" Plot generation code """
class Plots:
    def __init__(self):
        pass

    def plot(self, time_series):
        plt.plot(time_series)
        plt.ylabel("Objective")
        plt.xlabel("iterations")
        plt.grid()
        plt.show()
        plt.clf()

    def plot_cache_pos(self, cache, obj_means, cache_init):
        cache_objs = list(cache.values())
        xs = [l[0] for l in cache_objs]
        ys = [l[1] for l in cache_objs]
        plt.scatter(xs, ys, marker='+', label="cache")
        
        obj_cata = obj_means
        xs = [l[0] for l in obj_cata]
        ys = [l[1] for l in obj_cata]
        plt.scatter(xs, ys, marker='*', label="obj_mean")

        cache_objs = list(cache_init.values())
        xs = [l[0] for l in cache_objs]
        ys = [l[1] for l in cache_objs]
        plt.scatter(xs, ys, marker='o', label="initial")
        plt.legend()
        plt.show()
        plt.clf()
        


    def plot_cache_pos_grid(self, cache, obj_means, cache_init):
        cache_objs = cache
        xs = [l[0] for l in cache_objs]
        ys = [l[1] for l in cache_objs]
        plt.scatter(xs, ys, marker='+', label="cache")
        
        obj_cata = obj_means
        xs = [l[0] for l in obj_cata]
        ys = [l[1] for l in obj_cata]
        #plt.scatter(xs, ys, marker='*', label="obj_mean")

        cache_objs = cache_init #list(cache_init.values())
        xs = [l[0] for l in cache_objs]
        ys = [l[1] for l in cache_objs]
        #plt.scatter(xs, ys, marker='o', label="initial")

        plt.legend()
        plt.show()
        plt.clf()
        
        


            


