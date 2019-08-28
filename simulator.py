from util import * 
import argparse

class Simulator:
    def __init__(self, dim, capacity, no_objects, alpha, iter, update_interval):
        self.obj_catalogue = ObjectCatalogue(no_objects, alpha, dim)
        self.cache = Cache(capacity, dim, learning_rate)
        self.iter = iter
        self.u_interval = update_interval
        self.descent = StochasticGradientDescent(learning_rate)
        
    def simulate(self):
        for i in range(self.iter):
            obj = self.objectCatalogue.getRequest()
            nearest_obj = self.cache.getNearest()
            
            if i % self.u_interval == 0:
                new_object = self.descent.descent(nearest_obj, obj)
                self.cache.updateCache(nearest_obj, new_object)

                
                
                



        