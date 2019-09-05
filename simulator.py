from util import * 
import argparse
import copy

class Simulator:
    def __init__(self, dim, capacity, no_objects, alpha, iter, update_interval, learning_rate):
        #self.obj_catalogue = ObjectCatalogueGaussian(no_objects, 2, dim)
        #self.obj_catalogue = ObjectCatalogueUniform(no_objects, 0.8, dim)

        self.grid_d = 60

        self.obj_catalogue = ObjectCatalogueGrid(self.grid_d, self.grid_d)        

        #self.cache = Cache(capacity, dim, learning_rate)
        self.cache = Cache(capacity, dim, learning_rate, True, [self.grid_d, self.grid_d])
 
        self.iter = iter

        self.u_interval = update_interval

        self.descent = StochasticGradientDescent(learning_rate)

        self.plot =  Plots()

        self.initial_points = self.cache.getAllPoints()    
        #self.initial_points = [(p*self.grid_d) for p in self.cache.engine.get_all_points()]

    def simulate(self):
        objective = [] 

        count = 0
        for i in range(self.iter):

            obj = self.obj_catalogue.getRequest()
            pos = obj.pos
            nearest_obj = self.cache.findNearestANN(pos)                        

            if i % self.u_interval == 0:
                new_object_loc = self.descent.descent(nearest_obj, obj.pos)
                new_object = CacheObject(0, new_object_loc, 0)
                self.cache.updateCacheDict(nearest_obj, new_object)                

            if i % 100 == 0:
                print(i)
                objective.append(self.obj_catalogue.objective_l1(self.cache))
                self.plot.plot_cache_pos_grid(self.cache.getAllPoints(), self.obj_catalogue.means, self.initial_points, count)
                count += 1

        self.plot.plot(objective)

s = Simulator(2, 60, 100, 0.4, 40000, 1, 0.05)
s.simulate()                
                
                



        
