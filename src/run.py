import numpy as np
import csv
import time

# MIN_ERROR = float(input("Please insert the maximum value for the error: "))
MIN_ERROR = 0.05

def dec2bin(dec, bits):
    if bits <= 0:
        return np.array([]).astype(int)
    bit = int(dec >= 2**(bits-1))
    return np.append(bit,dec2bin(dec - (2**(bits-1))*bit, bits-1)) 

def bin2dec(bin):
	return np.sum(bin*(2**np.arange(bin.shape[0]-1,-1,-1)))

def sigmoid(x):
	return (1.0/(1+np.exp(-x)))

def sigmoid_derivative(x):
    return x * (1.0 - x)


class NETWORK:
	def __init__(self, N_hidden, N_inputs, N_outputs, L_rate):
		# constants
		self.L_rate = L_rate

		# matrix with weights from inputs to hidden layer
		self.W_ij = np.random.rand(N_inputs+1,N_hidden)
		self.W_jk = np.random.rand(N_hidden+1, N_outputs)

	def test(self, inputs):
		inputs = np.concatenate((inputs,np.array([1]*inputs.shape[0])[:,np.newaxis]),axis=1)
		out_j = sigmoid(np.dot(inputs, self.W_ij))
		out_j = np.concatenate((out_j,np.array([1]*out_j.shape[0])[:,np.newaxis]),axis=1)
		return sigmoid(np.dot(out_j, self.W_jk))

	def train(self, inputs, outputs):
		inputs = np.concatenate((inputs,np.array([1]*inputs.shape[0])[:,np.newaxis]),axis=1)
		out_j = sigmoid(np.dot(inputs, self.W_ij))
		out_j = np.concatenate((out_j,np.array([1]*out_j.shape[0])[:,np.newaxis]),axis=1)
		out_k = sigmoid(np.dot(out_j, self.W_jk))
		self.W_jk += self.L_rate*np.dot(out_j.T, (2*(outputs - out_k) * sigmoid_derivative(out_k)))
		aux0 = np.dot(2*(outputs - out_k) * sigmoid_derivative(out_k), self.W_jk.T)
		aux1 = sigmoid_derivative(out_j[:,:-1])
		aux0 = np.dot(inputs.T,aux0[:,:-1]*sigmoid_derivative(out_j[:,:-1]))
		self.W_ij += self.L_rate*aux0
		
		return np.max(np.absolute(outputs - out_k))


def main(Lrate):
	print('Initializing training csv')
	dataset = open('impar.csv', 'r');
	data = list(csv.reader(dataset, delimiter=','))
	data = np.array(data).astype(int)
	dataset.close()

	print('Initializing network and data preprocessing')
	neural = NETWORK(N_hidden=4,N_inputs=4,N_outputs=1,L_rate=Lrate)

	for i in range(1000000):
		if(neural.train(data[:,1:],(data[:,0])[:,np.newaxis]) < MIN_ERROR):
			break

	print(f'Training finished with {i} iterations.')
	dataset = open('teste_impar.csv', 'r');
	data = list(csv.reader(dataset, delimiter=','))
	data = np.array(data).astype(int)
	dataset.close()

	print(neural.test(data[:,1:]))

#for i in range(5,50,5):
#    start = time.time()
#    Lrate = i*10**(-2)
#    print("Learning rate = %f" % (Lrate))
#    main(Lrate)
#    print("Tempo de execucao: %f \n" %(time.time()-start))

Lrate = input("Please insert the Learning Rate you'd like to test: ")
main(float(Lrate))
