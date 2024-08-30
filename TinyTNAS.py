import numpy as np
from datetime import datetime
from ModelBank import *


class TinyTNAS :
    def __init__(self,train_ds,val_ds,input_shape,num_class,learning_rate, constraints_specs= {"ram" : 20800, "flash" : 258400000  , "macc" : 2565454545 }):
        
        self.BuildModelwithSpecs = BuildModelwithSpecs
        self.ModelTraning = ModelTraning
        self.CheckFeasible = CheckFeasible
       
        
        self.max_acc_found = -1
        self.ahead = 3
        self.divier = 4
        self.k=4
        self.c = 3
        self.K=self.k
        self.C=self.c
        self.pendings = []
        self.RAM = -1
        self.FLASH=-1
        self.MACC = -1
        self.feasible_solutions = []
        self.explored_model_count = 0
        self.explored_model_configs = []
        self.constraints_specs = constraints_specs
        self.train_ds = train_ds 
        self.val_ds = val_ds
        self.input_shape = input_shape
        self.num_class = num_class
        self.learning_rate= learning_rate
        self.infeasible_configarations = []
        self.search_started_time = None
        self.search_time_minute = None
        
    def ExploreDepth(self,k,current_c,current_acc,constraints_specs,epochs,N=5, lossf=1):
        exploreable_cs = list(set(np.arange(N))-set([current_c]))
        exploreable_cs.sort()
        max_acc = -1
        qualified_c = -1
        qualified_specs = None
        exploration_count = 0
        for i in exploreable_cs :
            if ((datetime.now() - self.search_started_time).total_seconds()/60) < self.search_time_minute :
                model,ram,flash,macc = self.BuildModelwithSpecs(k= k,c = i, ds = self.train_ds, input_shape = self.input_shape, learning_rate = self.learning_rate , num_class= self.num_class, lossf=lossf)
                current_specs = {"ram" : ram  , "flash" : flash  , "macc" : macc }
                isFeasible = self.CheckFeasible(constraints_specs, current_specs)
                if  isFeasible :
                    acc = self.ModelTraning(model = model, train_ds = self.train_ds, val_ds = self.val_ds, epochs = epochs)
                    exploration_count = exploration_count + 1
                    self.explored_model_configs.append([k,  i, acc, ram, flash, macc])

                    if max_acc<acc :
                        max_acc = acc
                        qualified_c = i
                        qualified_specs = current_specs
                else :
                    self.infeasible_configarations.append([k,  i, -1, ram, flash, macc])
                    break

        if max_acc > current_acc :

            return max_acc , qualified_c,qualified_specs,exploration_count
        else :
            return -1,-1,-1,exploration_count
        
    def update_status(self, acc,specs):
        import math
        self.max_acc_found = acc
        self.pendings = []
        diff  = self.k*2 - self.k
        incr =  int(math.floor(diff/self.divier))
        if incr >= 1 :
            for i in range(self.divier) :
                self.pendings.append([self.k+i*incr, self.c])

       
        self.K = self.k
        self.C = self.c
        self.RAM = specs["ram"]
        self.FLASH = specs["flash"]
        self.MACC = specs["macc"]
        self.feasible_solutions.append([self.K,self.C,self.max_acc_found,self.RAM,self.FLASH,self.MACC])
        self.k=self.k*2
        
    def func_k(self,acc , epochs,lossf):
        
        suggested_acc,suggested_c,suggested_specs,exploration_count = self.ExploreDepth(self.k,self.c,acc,self.constraints_specs,epochs=epochs,N=5 , lossf=lossf)
        self.explored_model_count= self.explored_model_count + exploration_count
        if suggested_acc > self.max_acc_found :
            self.c = suggested_c
            self.update_status(suggested_acc,suggested_specs)
            return True, []

        if len(self.pendings) > 0 :
            self.k,self.c = self.pendings.pop()
            return True, []
        else :
            return False ,[self.K,self.C,self.max_acc_found,self.RAM,self.FLASH,self.MACC]

        
        
    def search(self , epochs = 3 , lossf=1 , search_time_minute = 5):
        self.search_time_minute = search_time_minute
        self.search_started_time = datetime.now()

        
        
        while True :
           
            if ((datetime.now() - self.search_started_time).total_seconds()/60) < self.search_time_minute :
    
                model,ram,flash,macc = self.BuildModelwithSpecs(k= self.k,c = self.c, ds = self.train_ds, num_class = self.num_class , input_shape = self.input_shape, learning_rate = self.learning_rate, lossf =lossf)

                current_specs = {"ram" : ram  , "flash" : flash  , "macc" : macc }
                isFeasible = self.CheckFeasible(self.constraints_specs, current_specs)

                acc = -1
                if isFeasible :
                    acc = self.ModelTraning(model = model, train_ds = self.train_ds, val_ds = self.val_ds, epochs = epochs)

                    self.explored_model_count= self.explored_model_count + 1
                    self.explored_model_configs.append([self.k, self.c, acc, ram, flash, macc])

                    if self.max_acc_found < acc :
                        self.update_status(acc,current_specs)
                        continue
                    else:
                        acc=-1
                else :
                    self.infeasible_configarations.append([[self.k, self.c, -1, ram, flash, macc]])
                is_continuable , results = self.func_k(acc, epochs= epochs , lossf=lossf)
                if is_continuable :
                    continue
                else:
                    return results
            else :
                
                results = [self.K,self.C,self.max_acc_found,self.RAM,self.FLASH,self.MACC]
                return results
                
                    
             