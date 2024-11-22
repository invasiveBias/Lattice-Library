import numpy as np
import math


class Lattice_object(type):
    def __repr__(cls):
        typ = f"lattice.{cls.__name__}"
        typ = typ.lower()
        return typ


        
class array(metaclass= Lattice_object):
    def __init__(self,obj,arg=None,requires_grad = False, operation = None,tag=None,dtype=float):
        if type(obj) == list and type(obj[0]) == array:
            n_lst = []
            for i in obj:
                i = i._data if type(i) == array else i
                n_lst.append(i)
            obj = np.array(n_lst)
        self._data = np.array(obj,dtype=dtype) if type(obj) != np.ndarray else obj
        self.requires_grad = requires_grad
        self.operation = operation
        self.children = []
        self.shape = self._data.shape
        if self.requires_grad:
            self.grad = np.zeros_like(self._data)
        self.tag = tag
        self.scalar = False
        if len(self.shape) == 0:
            self.scalar = True
        self.dtype = dtype
    def __repr__(self):
        rg = f',requires_grad = {self.requires_grad})' if self.requires_grad == True else ''
        rep = f'{self._data} {rg}' if self.scalar == True else f'Lt_tensor({self._data} {rg})'
        return rep
    
    
    def backward(self, grad = None, z = None):
        ''' 
        Performs the backpropagation with gradient descent from current tensor.
        Will fill every tensor's "grad" attribute with gradients relative to "self" (current Tensor).
        To discuss in more detail how this function works, whenever it's called:
        1. it first confirms if computing gradients are set for the tensor and ends the operation with a message if not.
        2. if the 'grad' method wasn't given any gradient values for it's grad parameter when it was called (ie 'grad' parameter representing gradients reamins at default 'None') it creates placeholder gradient values of 1s matching the population/dimension of the tensor
        4. the values for the 'grad' parameter (either originally given when the method was called or set to 1s as placeholders ) are used to update the tensor's default gradient values (which were originally set to set to zeros).
           4b. these updated gradients become the outter-layer gradients for the final_tensor (ie dy/du) and when they are subsequently computed for precursor-parent tensors during the entire backpropagation process they are tagged "n minus k" layer gradients where n is the number of layers & k negatively sums by -1 as you go back a previous layer, so outtermost layer is n-1, the layer before that is n-2 etc
        5. it empties the tensor's 'children' attribute (which stores the children aka resulting variables of operations) as it is assumed that the surrent tensor is final output, since the backward computation is being done on it
        6. finally it rechecks that there are no more children nodes and sends the tensor's updated gradient and the tensor itself into the backward computation of the stored operation that created the tensor
        
        
        '''
        if not self.requires_grad:
            return "this tensor has requires_grad set to False"
        
        if grad is None:
            grad = np.ones_like(self._data)


        self.grad += grad
        
        
        if z is not None:
            self.children.remove(z)
        
        if self.operation:
            if not self.children:
                self.operation.backward(self.grad, self)
    
    
    def zero_grad(self):
        ''' Reset the Tensor's gradients to zero. '''
        self.grad = np.zeros_like(self._data)

    def zero_grad_tree(self):
        ''' Reset the gradients of this Tensor, and of all of the Tensors that led to it. '''
        self.zero_grad()
        if self.operation:
            for parent in self.operation.parents:
                parent.zero_grad_tree()
            self.operation = None
    
                
    def __add__(self, other):
        """ New = self + other """
        op = Add()
        other = array(other) if type(other) != type(self) else other
        return op.forward(self, other)
    
    
    
    def __radd__(self, other):
        """ New = other + self """
        op = Add()
        other = array(other) if type(other) != type(self) else other
        return op.forward(self, other)

    def __iadd__(self, other):
        """ self += other """
        op = Add()
        other = array(other) if type(other) != type(self) else other
        return op.forward(self, other)

    def __sub__(self, other):
        """ New = self - other """
        return self + -other

    def __rsub__(self, other):
        """ New = other - self """
        return other + -self

    def __isub__(self, other):
        """ self -= other """
        return self + -other
    
    def __neg__(self):
        """ self = -self """
        op = Neg()
        return op.forward(self) 
    
    
    def T(self):
        arr = self.reshape(self._data.T.shape)
        return arr
     
    def __mul__(self, other):
        """ New = self * other """
        op = Mul()
        other = array(other) if type(other) != type(self) else other
        return op.forward(self, other)
    
    
    def __rmul__(self, other):
        """ New = other * self """
        op = Mul()
        other = array(other) if type(other) != type(self) else other
        return op.forward(self, other)

    def __imul__(self, other):
        """ self *= other """
        op = Mul()
        other = array(other) if type(other) != type(self) else other
        return op.forward(self, other)
    
    def __pow__(self, other):
        op = Pow()
        other = array(other) if type(other) != type(self) else other
        return op.forward(self, other)
    
    def __rpow__(self, other):
        """ New = other ** power """
        op = Pow()
        other = array(other) if type(other) != type(self) else other
        return op.forward(other, self)
    
    def __truediv__(self, other):
        """ New = self / other """
        op = Div()
        other = array(other) if type(other) != type(self) else other
        return op.forward(self, other)
    
    def __rtruediv__(self, other):
        """ New = self / other """
        op = Div()
        other = array(other) if type(other) != type(self) else other
        return op.forward(other, self)
    
    def __itruediv__(self, other):
        """ New = self / other """
        op = Div()
        other = array(other) if type(other) != type(self) else other
        return op.forward(self, other)
    
    def __matmul__(self, other):
        """ New = self @ other """
        op = MatMul()
        other = array(other) if type(other) != type(self) else other
        return op.forward(self, other)
    
    
    def __len__(self):
        return len(self._data)
    

    def __gt__(self, other):
        """ New = self > other """
        other = array(other) if type(other) != type(self) else other
        return array(self._data > other._data ,requires_grad=self.requires_grad,dtype=bool)
    
    def __lt__(self, other):
        """ New = self > other """
        other = array(other) if type(other) != type(self) else other
        return array(self._data < other._data ,requires_grad=self.requires_grad,dtype=bool)
    
    def __ge__(self, other):
        """ New = self > other """
        other = array(other) if type(other) != type(self) else other
        return array(self._data >= other._data ,requires_grad=self.requires_grad,dtype=bool)
    
    def __le__(self, other):
        """ New = self > other """
        other = array(other) if type(other) != type(self) else other
        return array(self._data <= other._data ,requires_grad=self.requires_grad,dtype=bool)
    
    def __eq__(self,other):
        other = array(other) if type(other) != type(self) else other
        return array(self._data == other._data ,requires_grad=self.requires_grad,dtype=bool)
    
    def __ne__(self,other):
        other = array(other) if type(other) != type(self) else other
        return array(self._data != other._data ,requires_grad=self.requires_grad,dtype=bool)
    
    def __getitem__(self, index): 
        """ New = self[index] """
        op = Slice()
        return op.forward(self, index)
    
    def __iter__(self):
        self.ind = 0
        return self
    
    def __next__(self):
        if self.ind < self.shape[0]:
            val = self[self.ind]
            self.ind += 1
            return val
        else:
            raise StopIteration
    
    def __index__(self):
        return self.astype(int).item()
        
    def __setitem__(self, index, value):
        if type(index) == tuple and type(index[0]) == array:
            ind_ = index[0]._data.astype(int) if index[0].dtype == float else index[0]._data
            self._data[ind_,index[1]] = value._data if type(value) == array else value
        elif type(index) == tuple and type(index[1]) == array:
            ind_ = index[1]._data.astype(int) if index[1].dtype == float else index[1]._data
            self._data[index[0],ind_] = value._data if type(value) == array else value
        elif type(index) == array:
            index = index._data.astype(int) if index.dtype == float else index._data
            self._data[index] = value._data if type(value) == array else value
        else:
            self._data[index] = value._data if type(value) == array else value
    
    def __bool__(self):
        return self._data.tolist() != 0
    
    def reshape(self,*args):
        self._data = self._data.reshape(*args)
        self.shape = self._data.shape
        return self
    
    def flatten(self):
        arr = self._data.flatten()
        return array(arr)
    
    def T(self):
        arr = self._data.T
        return array(arr,requires_grad = self.requires_grad)
    
    def to_numpy(self,dtype=float):
        return np.array(self._data,dtype=dtype)
    
    def tolist(self):
        return self._data.tolist()
    
    def item(self):
        if self.scalar or self.shape == (1,):
            return self._data.item()
        else:
            raise Exception("Only scalar or tensor of shape (1,) has the item method")
        
    def astype(self,new_type):
        nt = new_type
        self._data = self._data.astype(nt)
        return self

    def max(self, dim= None, keepdims=False):
        """
        Returns the largest values across the "dim" dimention.
        Example: (B, T, D), dim = 1 -> (B, D).
        
        @param dim (int): dimention to be reduced (only largest remains).
        @param keepdims (bool): wether to broadcast result to same shape as input.
        """
        op = Max()
        return op.forward(self, dim, keepdims=keepdims)

    def sum(self, dim= None, keepdims=False):
        """
        Returns the sum of all values across the "dim" dimention.
        Example: (B, T, D), dim = 1 -> (B, D).
        
        @param dim (int): dimention to be summed across.
        @param keepdims (bool): wether to broadcast result to same shape as input.
        """
        op = Sum()
        return op.forward(self, dim, keepdims=keepdims)
    
    def log(self):
        op = Log()
        return op.forward(self)
    
    def exp(self):
        op = Exp()
        return op.forward(self)
    
    def sqrt(self):
        op = Sqrt()
        return op.forward(self)
    
    def mean(self, dim=None, keepdims=False):
        """
        Returns the mean of all values across the "dim" dimention.
        Example: (B, T, D), dim = 1 -> (B, D).

        @param dim (int): dimention to be averaged across.
        @param keepdims (bool): wether to broadcast result to same shape as input.
        """
        op = Mean()
        return op.forward(self, dim, keepdims=keepdims)

    def var(self, dim=None, keepdims=False):
        op = Var()
        return op.forward(self, dim, keepdims=keepdims)


""" The objects below represent the Operation classes for the methods of the Tensor class, they compute both the forward pass
(normal calculations) and the backward pass (gradient computations) for every parent tensor that contributed to the final output Tensor
for the forward pass;
1. The specified arithemtic operation is performed creating the child tensor, the 1st parameter value of the child tensor would allow it to compute gradients,while the 2nd would store the instance of the Operator class that led to it's formation for future reference purposes
2. the 2 tensors involved in the operation are logged in the Operator as parent tensors, after which each parent tensor logs the child tensor in their individual childern attributes then the parent tensors are cached by the operator to be directly used during the backward pass

To compute the gradients;
1. the 'backward' method of the Operator is initialized as the final process within the higher order 'backward' method of the final tensor itself (ie final_tensor.backward())
2. operator.backward() is given the outter-layer gradients (ie dy/du) from final_tensor.backward() and an instance of the final tensor itself as arguments
3. it inherits the cached parent_tensors used in the forward pass and computes the gradient for each parent w.r.t the outter-layer gradient (ie dy/du * du/d[parent_tensor])
4. To avoid dimensionality issues the newly calculated gradients are rebroadcasted in the same dimensions as the tensors used to compute them
5. Finally the parent tensors then call the higher order backward method & gives it their calculated gradient w.r.t the outter-layer gradient (dy/du * du/d[parent_tensor])) along with their child tensor as parameter values.
6. The reinitialization of the higher order backward method creates a recursive process that leads to initialization of the backward method of the n-2 layer operator at it's own end, this then leads again to the n-3 parent tensors reinitializing the higher order backward method & this will keep going until gradients are computed all the way back to the first layer.

"""

    
    
class Add:

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad
      
        # Get new Tensor's data:
        data = a._data + b._data
      
        # Create new Tensor:
        z = array(data, requires_grad=requires_grad, operation=self,tag= f"{a.tag} + {b.tag}") 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z
    
    def backward(self, dz, z): # dz represents dy/du (outter-layer gradients)
        a, b = self.cache
        
        
        # Find gradients relative to "a" ie du/da, and pass it downstream: 
        if a.requires_grad:
            da = dz  # using partial derivatives du/da in additon function a+b = 1 + 0 then according to chain rule dy/da = dz * 1=  dz

            # Rescale gradient to have the same shape as "a":
            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da, z)

        # Find gradients relative to "b" ie du/db, and pass it downstream:
        if b.requires_grad:
            db = dz # using partial derivatives du/da in additon function a+b = 0 + 1 then according to chain rule dy/da = dz * 1 =  dz

            # Rescale gradient to have the same shape as "b":
            grad_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db, z)


class Neg:

    def forward(self, a):
        requires_grad = a.requires_grad
   
        # Get new Tensor's data:
        data = - a._data 
   
        # Create new Tensor:
        z = array(data, requires_grad=requires_grad, operation=self) 
   
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)

        self.cache = a

        return z 
    
    def backward(self, dz, z):
        a = self.cache

        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            da = -dz # the derivative for a the negate of a variable is just the negative of the addition derivative
            a.backward(da, z)


class Mul:

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad
       
        # Get new Tensor's data:
        data = np.multiply(a._data, b._data)
       
        # Create new Tensor:
        z = array(data, requires_grad=requires_grad, operation=self) 
       
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z 
    
    def backward(self, dz, z):
        a, b = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # with chain rule dy/da = dz * du/da * b:
            da = dz * b._data

            # Rescale gradient to have the same shape as "a":
            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da, z)

        # Find gradients relative to "b", and pass it downstream:
        if b.requires_grad:
            # with chain rule dy/da = dz * a * du/db:
            db = dz * a._data

            # Rescale gradient to have the same shape as "b":
            grad_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db, z)


class Div:

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad
       
        # Get new Tensor's data:
        data = a._data / b._data
       
        # Create new Tensor:
        z = array(data, requires_grad=requires_grad, operation=self) 
       
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z  

    
    def backward(self, dz, z):
        a, b = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # with chain rule dy/da = dz * du/da * 1/b:
            da = dz * (1 / b._data)

            # Rescale gradient to have the same shape as "a":
            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da, z)

        # Find gradients relative to "b", and pass it downstream:
        if b.requires_grad:
            # with chain rule dy/db = dz * -a/b^2 :
            db = - dz * a._data / (b._data ** 2)

            # Rescale gradient to have the same shape as "b":
            grad_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db, z)
            

class Pow:
    def forward(self, tensor_a, tensor_b):
        requires_grad = tensor_a.requires_grad
        data = tensor_a._data ** tensor_b._data
        z = array(data, requires_grad=requires_grad, operation=self)
        self.parents = (tensor_a, tensor_b)
        tensor_a.children.append(z)
        tensor_b.children.append(z)
        self.cache = (tensor_a, tensor_b)
        return z
    
    def backward(self, dz, z):
        tensor_a, tensor_b = self.cache
        if tensor_a.requires_grad:
            da = dz * (tensor_b._data * tensor_a._data ** (tensor_b._data-1))
            grad_dim = len(da.shape)
            in_dim = len(tensor_a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            for n, dim in enumerate(tensor_a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            tensor_a.backward(da, z)
        if tensor_b.requires_grad:
            db = dz * tensor_a._data ** tensor_b._data * np.log(np.array(tensor_a._data,dtype=float))
            grad_dim = len(db.shape)
            in_dim = len(tensor_b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)
            for n, dim in enumerate(tensor_b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            tensor_b.backward(db, z)

class MatMul:

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad
     
        # Get new Tensor's data:
        data = a._data @ b._data
      
        # Create new Tensor:
        z = array(data, requires_grad=requires_grad, operation=self) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z  
    
    def backward(self, dz, z):
        a, b = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Backprop through the matmul:
            da = dz @ b._data.swapaxes(-1,-2)
            
            # Get difference between "a" size and upstream "da" size, to broadcast grad into "a":
            in_dim = len(a.shape)
            grad_dim = len(da.shape)

            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)

            a.backward(da, z)

        # Find gradients relative to "b", and pass it downstream:
        if b.requires_grad:
            # Backprop through the matmul:
            db = a._data.swapaxes(-1,-2) @ dz

            # Get difference between "b" size and upstream "db" size, to broadcast grad into "b":
            in_dim = len(b.shape)
            grad_dim = len(db.shape)


            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            b.backward(db, z)


class Exp:

    def forward(self, a):
        requires_grad = a.requires_grad
       
        # Get new Tensor's data:
        data = np.exp(np.array(a._data,dtype=float))
       
        # Create new Tensor:
        z = array(data, requires_grad=requires_grad, operation=self) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, data)

        return z
    
    def backward(self, dz, z):
        a, data = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # using the chain rule dy/d(E^a) = dz * du/d(E^a) <-> E^a
            da = dz * data
            a.backward(da, z)


class Log:

    def forward(self, a):
        requires_grad = a.requires_grad
        # Get new Tensor's data:
        data = np.log(np.array(a._data,dtype=float))
     
        # Create new Tensor:
        z = array(data, requires_grad=requires_grad, operation=self) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a)

        return z
    
    def backward(self, dz, z):
        a = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # using the chain rule dy/d(Log(a)) = dz * du/d(Log(a)) <-> (1/a)
            da = dz * (1 / a._data)
            a.backward(da, z)


class Sqrt:

    def forward(self, a):
        requires_grad = a.requires_grad
     
        # Get new Tensor's data:
        data = np.sqrt(np.array(a._data,dtype=float))
     
        # Create new Tensor:
        z = array(data, requires_grad=requires_grad, operation=self) 
     
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, data)

        return z
    
    def backward(self, dz, z):
        a, data = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # using chain rule dy/da = dz * du/d(sqrt(a)) <-> 1/2 * a ^ -(1/2) <-> (1/2) * (1/sqrt(a))
            da = (1 / 2) * (1 / data) * dz
            a.backward(da, z)


class Sum:

    def forward(self, a, dim, keepdims):
        requires_grad = a.requires_grad
     
        # Get new Tensor's data:
        data = a._data.sum(axis=dim, keepdims=keepdims)
     
        # Create new Tensor:
        z = array(data, requires_grad=requires_grad, operation=self) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a)

        return z
    
    def backward(self, dz, z):
        a =  self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            #  the derivative of a sum operation is dz * an array of ones who's dimensions = presum dimensions of 'a'  :
            da = np.ones(a.shape) * dz
            a.backward(da, z)

class Mean:

    def forward(self, a, dim, keepdims):
        requires_grad = a.requires_grad
    
        # Get new Tensor's data:
        data = a._data.mean(axis=dim, keepdims=keepdims)
      
        # Create new Tensor:
        z = array(data, requires_grad=requires_grad, operation=self) 
       
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, dim)

        return z
    
    def backward(self, dz, z):
        a, dim =  self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # the derivative of a mean operation on the array 'a' (du/d(mean(a))) is given as 1/n * du/d(a) where n is the constant number of values in the array of variables 'a', so dy/dx = dz *  du/d(mean(a))
            da = dz * np.ones(a.shape) / np.prod(np.array(a.shape)[dim])
            a.backward(da, z)

class Var:

    def forward(self, a, dim, keepdims):
        requires_grad = a.requires_grad
     
        # Get new Tensor's data:
        data = a._data.var(axis=dim, keepdims=keepdims)
      
        # Create new Tensor:
        z = array(data, requires_grad=requires_grad, operation=self) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, dim)

        return z
    
    def backward(self, dz, z):
        a, dim =  self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Propagate through the var(x) operation:
            da = np.ones(a.shape) * dz
            da = da * 2 * (a._data - a._data.mean(axis=dim, keepdims=True)) / np.prod(np.array(a.shape)[dim])
            a.backward(da, z)

class Max:

    def forward(self, a, dim, keepdims=False):
        requires_grad = a.requires_grad
      
        # Get new Tensor's data:
        data = np.max(a._data, axis=dim, keepdims=keepdims)
        if keepdims:
            data = np.ones(a.shape) * data

        # Create new Tensor:
        z = array(data, requires_grad=requires_grad, operation=self) 
     
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, data, dim)

        return z
    
    def backward(self, dz, z):
        a, data, dim =  self.cache

        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            max = data
            if a.shape != dz.shape: 
                # Brodcast upstream derivative to the size of "a":
                dz = np.expand_dims(dz, axis=dim)
                dz = dz * np.ones_like(a._data)
                # Brodcast upstream output (max) to the size of "a":
                max = np.expand_dims(data, axis=dim)
                max = max * np.ones_like(a._data)
            # Add upstream gradients to the [max] values:
            da = dz * np.equal(a._data, max)
            a.backward(da, z)
            

class Reshape:

    def forward(self, a, shape):
        requires_grad = a.requires_grad
      
        # Get new Tensor's data:
        data = a._data.reshape(*shape)
      
        # Create new Tensor:
        z = array(data, requires_grad=requires_grad, operation=self)
      
        # Add new Tensors to "children" and old Tensors to "parents": 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a)

        return z
    
    def backward(self, dz, z):
        a = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Reshape upstream gradients:
            da = dz.reshape(a.shape)
 
            a.backward(da, z)


class Transpose:

    def forward(self, a, *dims):
        requires_grad = a.requires_grad
       
        # Get new Tensor's data:
        data = a._data.swapaxes(*dims)
       
        # Create new Tensor:
        z = array(data, requires_grad=requires_grad, operation=self) 
       
        # Add new Tensors to "children" and old Tensors to "parents": 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, dims)

        return z
    
    def backward(self, dz, z):
        a, dims = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Transpose upstream gradients:
            da = dz.swapaxes(*dims)
 
            a.backward(da, z)


class Slice:

    def forward(self, a, index):
        requires_grad = a.requires_grad
        if type(index) == tuple and type(index[1]) == array:
            ind_ = index[1]._data.astype(int)
            data = a._data[index[0],ind_]
            
        elif type(index) == tuple and type(index[0]) == array:
            ind_ = index[0]._data.astype(int)
            data =  a._data[ind_,index[1]]
            
        elif type(index) == array:
            index = index._data.astype(int) if index.dtype == float else index._data
            data = a._data[index]
        else:
            data = a._data[index]
            
        
        # Create new Tensor:
        z = array(data, requires_grad=requires_grad, operation=self) 
       
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, index)

        return z
    
    def backward(self, dz, z):
        a, index =  self.cache
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Add upstream gradients to [index] part of da.
            da = np.zeros_like(a._data)
            da[index] = dz
            a.backward(da, z)



