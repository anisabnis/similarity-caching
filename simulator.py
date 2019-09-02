from util import * 
import argparse
import copy

class Simulator:
    def __init__(self, dim, capacity, no_objects, alpha, iter, update_interval, learning_rate):
        #self.obj_catalogue = ObjectCatalogueGaussian(no_objects, 2, dim)
        #self.obj_catalogue = ObjectCatalogueUniform(no_objects, 0.8, dim)
        self.grid_d = 31
        self.obj_catalogue = ObjectCatalogueGrid(self.grid_d, self.grid_d)        
        #self.cache = Cache(capacity, dim, learning_rate)
        self.cache = Cache(capacity, dim, learning_rate, True, [self.grid_d, self.grid_d])
        self.iter = iter
        self.u_interval = update_interval
        self.descent = StochasticGradientDescent(learning_rate)
        self.plot =  Plots()
        self.initial_points = copy.deepcopy(self.cache.cache)

    def simulate(self):
        objective = [] 
        
        for i in range(self.iter):

            obj = self.obj_catalogue.getRequest()
            pos = np.array([float(x)/self.grid_d for x in obj.pos])
            [nearest_obj, nearest_object_id] = self.cache.findNearestANN(pos)            
            
            nearest_obj *= self.grid_d

            if i % self.u_interval == 0:
                new_object_loc = self.descent.descent(nearest_obj, obj.pos)
                new_object = CacheObject(0, new_object_loc, 0)
                self.cache.updateCacheDict(nearest_object_id, new_object)                

            if i%100 == 0:
                objective.append(self.obj_catalogue.objective(self.cache.cache))
                print(i)

        
        #self.plot.plot_cache_pos(self.cache.cache, self.obj_catalogue.means, self.initial_points)
        self.plot.plot(objective)
        #rint(self.obj_catalogue.means)
        print("cache : ", self.cache.cache)
        

s = Simulator(2, 100, 100, 0.4, 1800, 1, 0.001)
s.simulate()                
                
                



        
