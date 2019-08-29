from util import * 
import argparse

class Simulator:
    def __init__(self, dim, capacity, no_objects, alpha, iter, update_interval, learning_rate):
#        self.obj_catalogue = ObjectCatalogueGaussian(no_objects, 5, dim)
        self.obj_catalogue = ObjectCatalogueUniform(no_objects, 0.8, dim)
        self.cache = Cache(capacity, dim, learning_rate)
        self.iter = iter
        self.u_interval = update_interval
        self.descent = StochasticGradientDescent(learning_rate)
        self.plot =  Plots()
        
    def simulate(self):
        objective = [] 
        for i in range(self.iter):
            print("cache : ", self.cache.cache)
            obj = self.obj_catalogue.getRequest()
            [nearest_obj, nearest_object_id] = self.cache.findNearest(obj.pos)            
            if i % self.u_interval == 0:
                new_object_loc = self.descent.descent(nearest_obj, obj.pos)
                new_object = CacheObject(0, new_object_loc, 0)
                self.cache.updateCacheDict(nearest_object_id, new_object)

                objective.append(self.obj_catalogue.objective(self.cache.cache))

        self.plot.plot(objective)

s = Simulator(2, 6, 100, 0.8, 1000, 1, 0.01)
s.simulate()                
                
                



        
