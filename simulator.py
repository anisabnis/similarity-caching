from util import * 
import argparse

class Simulator:
    def __init__(self, dim, capacity, no_objects, alpha, iter, update_interval, learning_rate):
        self.obj_catalogue = ObjectCatalogue(no_objects, alpha, dim)
        self.cache = Cache(capacity, dim, learning_rate)
        self.iter = iter
        self.u_interval = update_interval
        self.descent = StochasticGradientDescent(learning_rate)
        
    def simulate(self):
        for i in range(self.iter):
            obj = self.obj_catalogue.getRequest()
            [nearest_obj, nearest_object_id] = self.cache.findNearest(obj.pos)            

            if i % self.u_interval == 0:
                new_object_loc = self.descent.descent(nearest_obj, obj.pos)
                new_object = CacheObject(0, new_object_loc, 0)
                self.cache.updateCache(nearest_object_id, new_object)


s = Simulator(2, 1000, 10000, 0.8, 1000, 1, 0.001)
s.simulate()                
                
                



        
