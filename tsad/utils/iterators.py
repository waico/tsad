"""
Very useful librray
"""
import numpy as np

# class MeshLoader:
#     def __init__(self, params):
#         """
# ЭТО ОКАЗЫВАЕТСЯ в NUMPY
# Only list in list, without оборачиваетелй numpy
# 		# mesh_alpha = np.arange(10)
# 		# mesh_beta = np.arange(10)
# 		# for k,z in enumerate(MeshLoader([[0.1,0.2,0.3],['a','b']])):
# 		#     print(z)
# 		"""
#         self.params = params
#         _my_array = np.ones(tuple(len(obj) for obj in params))
#         self.indexes = []
#         for index, values in np.ndenumerate(_my_array):
#             self.indexes.append(index)

#     def __iter__(self):
#         self.i = -1
#         return self

#     def __next__(self):
#         if self.i < len(self.indexes)-1:
#             self.i+=1
#             return tuple(self.params[from_parametr][self.indexes[self.i][from_parametr]] for from_parametr in range(len(self.params)))
#         else:
#             raise StopIteration

class Loader:
    def __init__(self, X,y, batch_size,shuffle=True, random_state = None):
        """
        X, y  are lists
        """
        if shuffle==True:
            np.random.seed(random_state)
            indices = np.array(range(len(X)))
            np.random.shuffle(indices)
            self.X = [X[ind] for ind in indices]
            self.y = [y[ind] for ind in indices]
        else:
            self.X = X
            self.y = y
        self.batch_size = batch_size

    def __iter__(self):
        self.i = - self.batch_size
        return self

    def __next__(self):
        if self.i+self.batch_size < len(self.X):
            self.i+=self.batch_size
            return self.X[self.i:self.i+self.batch_size], self.y[self.i:self.i+self.batch_size]
        elif self.i+2*self.batch_size < len(self.X):        
            return self.X[self.i:], self.y[self.i:]
        else:
            raise StopIteration
            
    def __len__(self):
        return len(np.arange(0,len(self.X),self.batch_size))
		        