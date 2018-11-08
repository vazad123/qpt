#! /usr/bin/env python3
#Protocol using kicked Ising dynamics for generating states with maximal multipartite entanglement
# programmer: Rachana S Yajur     email:rac.3570@gmail.com 

#FLOWCHART

#AIM to plot von nuemann entropy as a function of kicks

#1. Define the unitary evolution matrix (kick, after kick)

#2. Fast Walsch hadamard transformation

#3. Apply unitary on initial state

#4. Calculate the entanglement entropy
 

import numpy as np
import cmath
import math
import matplotlib.pyplot as plt
import random
import itertools
import qutip
from qutip import *
from scipy import linalg
from numpy import linalg as LA
from matplotlib.backends.backend_pdf import PdfPages



#Defining Pauli matrices
sigmax=np.array([[0,1],[1,0]],dtype=complex)
sigmay=np.array([[0,-1j],[1j,0]],dtype=complex)
sigmaz=np.array([[1,0],[0,-1]],dtype=complex)  
eye=np.eye(2)
sigmalist=[eye,sigmaz,sigmay,sigmax]                        # interchanged sigmax and sigmaz to against that in the paper

def matrix(ls):												#Funcion to create tensor products of Paulis
    if np.size(ls)==1:
        return sigmalist[ls[0]];
    for i in range(len(ls)-1):
        if (i==0):
            mat=np.kron(sigmalist[ls[0]],sigmalist[ls[1]])
        else:
            mat=np.kron(mat,sigmalist[ls[i+1]])
    return(mat)

def SZmSZn(m,n,size):										#Nearest neighbour terms
    ls=[0 for i in range(size)]
    ls[n]=3;ls[m]=3;
    H1t=matrix(ls)
    return(H1t)
    
def HJ(size,pbc=1):  										#nearest neighbour hamiltonian 
    if pbc==1:
        H=SZmSZn(N-1,0,size);
    else:
        dim=2**size;
        H=np.zeros((dim,dim));
    for i in range(1,size):
        H = H + SZmSZn(i-1,i,size)    
    return H;

def Hz(size):												#Magnetic field hamiltonian
    dim=2**size;
    H=np.zeros((dim,dim))
    for i in range(size):
        ls=[0 for i1 in range(size)];ls[i]=1;
        H=H+matrix(ls);
    return(H)


N = 4;														#Total number of spins
Nofkicks = 20; 

H0 = HJ(N);   												# Interaction Hamiltonian
val0, vec0 = LA.eigh(H0) 									#eigenvalues and eigen vectors of the interaction 																Hamiltonian 

Hkick =  Hz(N);   											# Kick term which adds as a delta function repeating after 																PI/4 time 

print(H0);

#Unitary evolution definition
Ukick = linalg.expm(-1j*(math.pi/4)*Hkick)     
Uafter = linalg.expm(-1j*(math.pi/4)*H0)
#Evolution for consecutive kicks is given by U = (Uafter*Ukick)

#Random number generation
def Random():
	probab = random.random()                                    #Generating random number betweeen 0 and 1
	t = 0; 
	if probab < 0.5:
		t = 1;
	else:
		t = -1;
	return(t);
	
###generating Thue-Morse sequence##

def genThueMorse():
	tm = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
	for i in range(Nofkicks+1):
		if tm[i] == 0:
			tm[i] = -1;
		else:
			tm[i] = 1;
	return(tm);



#Initial state as the first element of standard basis
initialstate = np.zeros((1,2**N));
initialstate[0][0] = 1.;
#initialstateinZZbasis=np.dot(np.transpose(np.conjugate(vec0)),initialstate);

#Unitary evolutiondifferent - easy method can handle max 10 spins in dcf pc
def Uevolutioneasy(state):
	evolvedstate = np.matmul(np.matmul(Uafter,Ukick),state);
	return(evolvedstate);
		

########entropy###################
ok = np.transpose(initialstate);
vnentropy = np.zeros(Nofkicks);
term = np.zeros(2**N);
for i in range(Nofkicks):
	ok = Uevolutioneasy(ok)
	densitymatrix = np.dot(np.conjugate(ok),np.transpose(ok))// np.linalg.norm(ok)**2;
	#partial trace
	
	eig = np.transpose(LA.eigvals(densitymatrix));
	#print(eig);
	
	for j in range(2**N):
	    if eig[j] < 10**-3:
	        term[j] = 0;
	    else:
	        term[j] = eig[j]*np.log2(eig[j]);
	    #print(term[j])
	vnentropy[i] = sum(term);
	
### not so BIG FOR LOOP contain only noisy things ###########
times = 1;
for i in range(times):
                                                 #Total number of kicks given 
	#kicks with noise
	dt = 3.14/8;
	tau = [];
	for i in range(Nofkicks + 1):
		tau.append(Random());

	def noisevolution(k):
		noiseUkick = linalg.expm(-1j*(math.pi/4 + (tau[k+1]-tau[k])*dt)*Hkick);   
		noiseUafter = linalg.expm(-1j*(math.pi/4 + tau[k]*dt)*H0);
		return(noiseUkick,noiseUafter);

	
	#Unitary noiseevolutiondifferent - easy method can handle max 10 spins in dcf pc
	def noiseUevolutioneasy(state,k):
		(noiseUkick,noiseUafter) = noisevolution(k);
		evolvedstate = np.matmul(np.matmul(noiseUafter,noiseUkick),state);
		return(evolvedstate);

	
	######## NOISE entropy###################
	def noiseentropy():
		ok = np.transpose(initialstate);
		noisevnentropy = np.zeros(Nofkicks);
		term = np.zeros(2**N);
		
		for i in range(Nofkicks):
			ok = noiseUevolutioneasy(ok,i)
			densitymatrix = np.dot(np.conjugate(ok),np.transpose(ok))// np.linalg.norm(ok)**2;
			eig = np.transpose(LA.eigvals(densitymatrix));
			for j in range(2**N):
				if eig[j] < 10**-3:
					term[j] = 0;
				else:
					term[j] = eig[j]*np.log2(eig[j]);
			noisevnentropy[i] = sum(term)
		return(noisevnentropy);
	en = noiseentropy();
	plt.plot(en, label = 'Random noise')


########THUE MORSE#############
TM = genThueMorse();
def TMevolution(k):
		TMUkick = linalg.expm(-1j*(math.pi/4 + (TM[k+1]-TM[k])*dt)*Hkick);   
		TMUafter = linalg.expm(-1j*(math.pi/4 + TM[k]*dt)*H0);
		return(TMUkick,TMUafter);

	
	#Unitary noiseevolutiondifferent - easy method can handle max 10 spins in dcf pc
def TMUevolutioneasy(state,k):
	(TMUkick,TMUafter) = TMevolution(k);
	evolvedstate = np.matmul(np.matmul(TMUafter,TMUkick),state);
	return(evolvedstate);

	
	######## NOISE entropy###################
def TMentropy():
	ok = np.transpose(initialstate);
	TMvnentropy = np.zeros(Nofkicks);
	term = np.zeros(2**N);
	
	for i in range(Nofkicks):
		ok = TMUevolutioneasy(ok,i)
		densitymatrix = np.dot(np.conjugate(ok),np.transpose(ok))// np.linalg.norm(ok)**2;
		eig = np.transpose(LA.eigvals(densitymatrix));
		for j in range(2**N):
			if eig[j] < 10**-3:
				term[j] = 0;
			else:
				term[j] = eig[j]*np.log2(eig[j]);
		TMvnentropy[i] = sum(term)
	return(TMvnentropy);
TMen = TMentropy();
plt.plot(TMen,label = 'ThueMorse sequence')
plt.xlabel('Number of kicks')
plt.ylabel('Von Neumann entropy')
plt.savefig('VNEntropy_%d_%d_%d.png' %(N,Nofkicks,times))	
plt.plot(vnentropy, label = 'dT = 0')
plt.legend();
plt.show();


