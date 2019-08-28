""" This file contains the various utilities which can be used to simulate a similarity cache """

import numpy as np
import scipy.stats as stats
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections


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
    def __init__(self, capacity, dim, learning_rate):
        self.cache = np.random.rand(capacity, dim)
        print(self.cache)
        self.alpha = learning_rate
        self.capacity = capacity
        self.initializeLSH(dim)        
        
    def initializeLSH(self, dim):
        rbp = RandomBinaryProjections('rbp', 5)
        self.engine = Engine(dim, lshashes=[rbp])

        for index in range(self.capacity):
            v = self.cache[index]
            self.engine.store_vector(v, '%d' % index)

    def updateCache(self, src_object_id, dst_obj):
        self.engine.delete_vector(str(src_object_id))
        self.engine.store_vector(dst_obj.pos, '%d' % dst_obj.id)
    
    def findNearest(self, vec):
        K = self.engine.neighbours(vec)

        nearest_point = K[0][0]
        min_dst = np.linalg.norm(K[0][0]-vec)
        min_id = K[0][1]

        for k in range(len(K)):
            if np.linalg.norm(K[k][0] - vec) < min_dst:
                nearest_point = K[k][0]
                min_dst = np.linalg.norm(K[k][0]-vec)
                min_id = K[k][1]

        print("K : ", vec, nearest_point, min_dst, min_id)        
        return [nearest_point, min_id]


""" Object Catalogue """
class ObjectCatalogue:
    def __init__(self, no_objects, alpha, dim):
        
        self.catalogue = []

        ## generate objects at random locations        
        for i in range(no_objects):
            loc = np.random.rand(dim)
            c_obj = CacheObject(i, loc)
            self.catalogue.append(c_obj)

        N = no_objects
        x = np.arange(1, N+1)
        a = alpha
        weights = x ** (-a)
        weights /= weights.sum()
        self.bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))

    def getRequest(self):
        obj = self.catalogue[self.bounded_zipf.rvs(size=1)[0]]
        return obj
        

""" DescentMethods """
class StochasticGradientDescent:
    def __init__(self, learning_rate):
        self.alpha = learning_rate

    def descent(self, nearest_object, current_object):

        def derivative(nearest_object, current_object):
            return 2 * (nearest_object - current_object)

        d = derivative(nearest_object, current_object)
        return current_object - self.alpha * d


""" Plot generation code """
class Plots:
    def __init__(self):
        pass




            


