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
#sigmax=np.array([[0,1],[1,0]],dtype=complex)
#sigmay=np.array([[0,-1j],[1j,0]],dtype=complex)
#sigmaz=np.array([[1,0],[0,-1]],dtype=complex)  
#eye=np.eye(2)
sigmalist=[qeye(2),sigmaz(),sigmay(),sigmax()]                        # interchanged sigmax and sigmaz to against that in the paper

def matrix(ls):												#Funcion to create tensor products of Paulis
    if np.size(ls)==1:
        return sigmalist[ls[0]];
    for i in range(len(ls)-1):
        if (i==0):
            mat=tensor(sigmalist[ls[0]],sigmalist[ls[1]])
        else:
            mat=tensor(mat,sigmalist[ls[i+1]])
    return(mat)

def SZmSZn(m,n,size):										#Nearest neighbour terms
    ls=[0 for i in range(size)];
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
    H=0*HJ(N)
    for i in range(size):
        ls=[0 for i1 in range(size)];ls[i]=1;
        H=H+matrix(ls);
    return(H)

a = math.pi;
b = 0;

N = 8;														#Total number of spins
Nofkicks = 30; 

H0 = HJ(N);   												# Interaction Hamiltonian
#val0, vec0 = LA.eigh(H0) 									#eigenvalues and eigen vectors of the interaction 																Hamiltonian 

Hkick =  Hz(N);   											# Kick term which adds as a delta function repeating after 																PI/4 time 


#Unitary evolution definition
Ukick = (-1j*(math.pi/4)*Hkick).expm()   
Uafter = (-1j*(math.pi/4)*H0).expm()
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

def initialstate(theta,phi):
	initialstate = math.cos(theta/2)*basis(2,0)+np.exp(1j*phi)*math.sin(theta/2)*basis(2,1);
	for i in range(N-1):
		initialstate = tensor(initialstate,math.cos(theta/2)*basis(2,0)+np.exp(1j*phi)*math.sin(theta/2)*basis(2,1));
	return(initialstate)
#initialstateinZZbasis=np.dot(np.transpose(np.conjugate(vec0)),initialstate);

#Unitary evolutiondifferent - easy method can handle max 10 spins in dcf pc
def Uevolutioneasy(state):
	evolvedstate = Uafter*Ukick*state;
	return(evolvedstate);
		




########entropy###################
ok = initialstate(a,b);
vnentropy = np.zeros(Nofkicks);
term = np.zeros(2**N);
for i in range(Nofkicks):
	densitymatrix = ok.ptrace((0,1,2,3));
	eig = densitymatrix.eigenenergies();
	ok = Uevolutioneasy(ok)
	
	for j in range(16):
	    if eig[j] < 10**-3:
	        term[j] = 0;
	    else:
	        term[j] = eig[j]*np.log2(eig[j]);
	    #print(term[j])
	vnentropy[i] = -sum(term);
	
### not so BIG FOR LOOP contain only noisy things ###########
times = 50	;													# number of random samples to collect
avg = np.zeros(Nofkicks)
for i in range(times):
                                                 
	#kicks with noise
	dt = math.pi/4;
	tau = [];
	for i in range(Nofkicks + 1):
		tau.append(Random());

	######## NOISE entropy###################
	def noiseentropy():
		ok = initialstate(a,b);
		noisevnentropy = np.zeros(Nofkicks);
		term = np.zeros(16);
		
		for i in range(Nofkicks):
			noiseUkick = (-1j*(math.pi/4 + (tau[i+1]-tau[i])*dt)*Hkick).expm();   
			noiseUafter = (-1j*(math.pi/4 + tau[i]*dt)*H0).expm();
			densitymatrix = ok.ptrace((0,1,2,3));
			eig = densitymatrix.eigenenergies();
			ok = noiseUafter*noiseUkick*ok;
			for j in range(16):
				if eig[j] < 10**-4:
					term[j] = 0;
				else:
					term[j] = eig[j]*np.log2(eig[j]);
			noisevnentropy[i] = -sum(term)
		return(noisevnentropy);
	plt.plot(noiseentropy());
	avg  = avg + noiseentropy();
	
plt.plot(avg/times)




########THUE MORSE#############

#TM = genThueMorse();
def TMevolution(k):
		TMUkick = (-1j*(math.pi/4 + (TM[k+1]-TM[k])*dt)*Hkick).expm();   
		TMUafter = (-1j*(math.pi/4 + TM[k]*dt)*H0).expm();
		return(TMUkick,TMUafter);

	
	#Unitary noiseevolutiondifferent - easy method can handle max 10 spins in dcf pc
def TMUevolutioneasy(state,k):
	(TMUkick,TMUafter) = TMevolution(k);
	evolvedstate = TMUafter*TMUkick*state;
	return(evolvedstate);

	
	######## NOISE entropy###################
def TMentropy():
	ok = initialstate(a,b);
	TMvnentropy = np.zeros(Nofkicks);
	term = np.zeros(2**N);
	
	for i in range(Nofkicks):
		densitymatrix = ok.ptrace((0,1));
		eig = densitymatrix.eigenenergies();
		ok = TMUevolutioneasy(ok,i)
		for j in range(4):
			if eig[j] < 10**-3:
				term[j] = 0;
			else:
				term[j] = eig[j]*np.log2(eig[j]);
		TMvnentropy[i] = -sum(term)
	return(TMvnentropy);



#TMen = TMentropy();
#plt.plot(TMen,label = 'ThueMorse sequence')
plt.xlabel('Number of kicks')
plt.ylabel('Von Neumann entropy')	
plt.title('4+4, theta = pi, phi = 0')
plt.plot(vnentropy, label = 'dT = 0')
plt.savefig('pi_0_4_4_VNEntropy_%d_%d_%d.png' %(N,Nofkicks,times))
plt.legend();
plt.show();

	
