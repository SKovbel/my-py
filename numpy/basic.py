import numpy as np

# a + b
# a * b
#
# ob.shape
# ob.ndim
# ob.dtype
# 
# np.sin
# np.sqrt
# np.log
# np.log2
# np.log10
#
# np.array
# np.full
# np.empty
# np.zero
# np.one
# np.arrange
# np.linspace
#
# np.append
# np.insert
# np.delete
#
# np.inf
# np.nan
# np.isnan
# np.isinf
#
# np.reshape
# np.resize
# np.flatten
# np.reval
#
# np.concatination
# np.stack
# np.vstack
# np.hstack
# np.split
#
# np.swapaxes   
# ob.transponse
# ob.T
#
# a.min
# a.max
# a.mean
# a.std
# a.sum
# np.median
#
# np.random.randint
# np.random.binomial
# np.random.normal
# np.random.choise
#
# np.save
# np.savetxt    
# np.load 
# np.loadtxt
#
a1 = np.array([1,2,3,4,5])
a2 = np.array([[1,2,3,4,5]])

a = np.concatenate((a1.reshape((1,a1.shape[0])), a2), axis=1)




print(a)

