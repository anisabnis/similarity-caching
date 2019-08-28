""" This file contains the various utilities which can be used to simulate a similarity cache """

import numpy as np
import scipy.stats as stats
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections


""" All attributes of a cache object """
class CacheObject:
    def __init__(self, id, vec, size=1):
        self.id = id
        self.pos = vec
        self.size = size


""" All attributes and functions of a cache """
class Cache:
    def __init__(self, capacity, dim, learning_rate):
        self.cache = np.random.rand(capacity, dim)
        self.alpha = learning_rate
        self.initializeLSH(dim)        
        
    def initializeLSH(self, dim):
        rbp = RandomBinaryProjections('rbp', 10)
        self.engine = Engine(dim, lshashes=[rbp])

        for index in range(capacity):
            v = self.cache[index]
            engine.store_vector(v, 'data_%d' % index)

    def updateCache(self, src_vec, dst_vec):
        self.engine.delete_vector(src_vec)
        self.engine.store_vector(dst_vec)
    
    def findNearest(self, vec):
        return self.engine.neighbours(vec)[0][0]


""" Object Catalogue """
class ObjectCatalogue:
    def __init__(self, no_objects, alpha, dim):
        
        self.catalogue = []

        ## generate objects at random locations        
        for i in no_objects:
            loc = numpy.random.randn(dim)
            c_obj = CacheObject(i, loc)
            self.catalogue.append(c_obj)

        N = no_objects
        x = np.arange(1, N+1)
        a = alpha
        weights = x ** (-a)
        weights /= weights.sum()
        self.bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))

    def getRequest(self):
        obj = self.catalobue[bounded_zipf.rvs(size=1)[0]]
        return obj.pos
        


""" DescentMethods """
class StochasticGradientDescent:
    def __init__(self, learning_rate):
        self.alpha = learning_rate

    def descent(self, nearest_object, current_object):

        def derivative(nearest_object, current_object):
            return 2 * (nearest_object - current_object)

        d = self.derivative(nearest_object, current_object)
        return currect_object - self.learning_rate * derivative


""" Plot generation code """
class Plots:
    def __init__(self):
        pass




            


