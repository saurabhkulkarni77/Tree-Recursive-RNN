#Recursive RNN
def forwardProp(self,node):
	#Recursion
	node.h = np.dot(self.w,np.hstack([node.left.h, node.right.h])) + self.b
	#Relu
	node.h[node.h<0] = 0
	#softmax 
	node.probs = np.dot(self.Ws,node.h) + self.bs
	node.probs -= np.max(node.probs)
	node.probs = np.exp(node.probs)
	node.probs = node.probs/node.sum(node.probs)
def backProp(self,node,error=None):
	deltas = node.probs
	deltas[node.labels] -= 1.0
	self.dWs += np.outer(deltas,node.h)
	self.dbs += deltas
	deltas = np.dot(self.Ws.T,deltas)
if error is not None:
	 deltas += error

#f'(z) now:
deltas *= (node.h != 0)

#update word vector if leaf node:
if node.isLeaf:
	self.dL[node.word] += deltas
	return
	
#Recurresively backprop
if not node.isLeaf:
	self.dw += np.outer(deltas,np.hstack([node.left.h, node.right.h]))
	self.db += deltas
	# Error signal to children
	deltas = np.dot(self.W.T, deltas)
	self.backProp(node.left, deltas[:self.hiddenDm])
	self.backProp(node.right, deltas[self.hiddenDm:b])