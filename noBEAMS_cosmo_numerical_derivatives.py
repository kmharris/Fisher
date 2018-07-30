import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib.patches import Ellipse

from astropy.cosmology.funcs1 import distmod
#I'll need to put in cosmology because I took it all out of the source code

import os


#best fit cosmological parameters


om = 0.298421
H = 100*0.779975
w = -0.998821
M = -19.1154

'''

#MEANS

om = 0.295909
H = 100*0.712292 
w = -1.03853 
M = -19.323 
'''


omega_b = 0.04

#lets do some mu calculations
from astropy.cosmology import Flatw0waCDM
#set a specific cosmology

cos = Flatw0waCDM(H, om, w0= w, Ob0 = omega_b)

data_file = os.path.join(os.path.expanduser('~/cosmosis/cosmosis-standard-library/supernovae/simplechi2/Data/Shafer2'), 'lcparam_DS17f_G10.txt')

datamat = np.loadtxt(data_file, unpack = True)

zhel = datamat[2] # redshift 
zcmb = datamat[1] # redshift
mb = datamat[4] 
dmb = datamat[5]




def f(mb, dmb, zcmb, M, H, om, w):
    cos = Flatw0waCDM(H, om, w0= w, Ob0 = omega_b)
    mu_theory = distmod(zcmb, cosmo=cos)  
    mu_theory = mu_theory.value

    
    mu_obs= mb - float(M)
    
    diffvec1a = (mu_obs-mu_theory)

      
    c1a = diffvec1a**2/dmb**2
    
    #print c1a
   
    rt = []
    for i in range(40):
        f1 = Decimal(1./(np.sqrt(2*np.pi)*dmb[i]))*Decimal(-c1a[i]/2).exp() 
        #print f1
 
        rt.append(float(f1))
    
    return np.array(rt)
 

def log_f(mb, dmb, zcmb, M, H, om, w):
    cos = Flatw0waCDM(H, om, w0= w, Ob0 = omega_b)
    mu_theory = distmod(zcmb, cosmo=cos)  
    mu_theory = mu_theory.value
    
    mu_obs= mb - float(M)
    
    diffvec1a = (mu_obs-mu_theory)
       

    c1a = diffvec1a**2/dmb**2

    
    rt = []
    for i in range(40):

        f1 = Decimal(1./(np.sqrt(2*np.pi)*dmb[i]))*Decimal(-c1a[i]/2).exp()
     
        rt.append(float(Decimal.ln(f1))) #maybe this is the float statement that needs to go?
    
    
    return np.array(rt)
    

#this naming convention is important, do dlogfd + whatever the name of the parameter is 
def dlogfdM(mb, dmb, zcmb): 
    h = 1e-6
    return (log_f(mb, dmb, zcmb, M + h, H, om, w ) - log_f(mb, dmb, zcmb,  M, H, om, w   ))/h
        
def dlogfdH(mb, dmb, zcmb): 
    h = 1e-6
    return (log_f(mb, dmb, zcmb, M , H +h, om, w   ) - log_f(mb, dmb, zcmb,  M, H, om, w   ))/h

def dlogfdom(mb, dmb, zcmb):
    h = 1e-6
    return (log_f(mb, dmb, zcmb, M , H, om + h, w ) - log_f(mb, dmb, zcmb,  M, H, om, w  ))/h

def dlogfdw(mb, dmb, zcmb):
    h=1e-6
    return (log_f(mb, dmb, zcmb, M, H, om, w + h  ) - log_f(mb, dmb, zcmb, M, H, om, w  ))/h
'''
         
# order is om, H, w, M,    , make sure order of funcs matches order of columns in emcee output
Ms = []
Hs = []
oms = []
ws = []
hs = np.logspace(-12, 2, 100, base = 10.)
for h in hs:
    Ms.append(np.trapz((eval('dlogfdM')(mb, dmb, zcmb))**2*f(mb, dmb, zcmb, M, H, om, w), mb))
    Hs.append(np.trapz((eval('dlogfdH')(mb, dmb, zcmb))**2*f(mb, dmb, zcmb, M, H, om, w), mb))
    oms.append(np.trapz((eval('dlogfdom')(mb, dmb, zcmb))**2*f(mb, dmb, zcmb, M, H, om, w), mb))
    ws.append(np.trapz((eval('dlogfdw')(mb, dmb, zcmb))**2*f(mb, dmb, zcmb, M, H, om, w), mb))

plt.plot(hs, Ms, 'ro')
plt.xscale('log')
plt.yscale('log')
plt.show()
'''

funcs = ['dlogfdom', 'dlogfdH','dlogfdw','dlogfdM']


F = np.zeros((len(funcs),len(funcs)))



for i in range(len(funcs)):
    for j in range(len(funcs)):
        F[i,j] = np.trapz(eval(funcs[i])(mb , dmb , zcmb )*eval(funcs[j])(mb , dmb , zcmb )*f(mb , dmb , zcmb , M, H, om, w), zcmb )


def jagged_integrate(x,y):
    '''
    This integrates a jagged function that has a certain number of outputs (y) and an
    equal number of independent variables, x.
    '''
    area = 0
    for i in range(0,(len(x) -1)):
        area += min(y[i], y[i+1])*(x[i+1] - x[i]) + 0.5*((x[i+1] - x[i]))*(max(y[i], y[i+1]) - min(y[i], y[i+1]))
        
    return area

        


ind = [0,2]
non_ind = [1,3]

def margMat(inds,non_inds, FMin):
 
    #We will use the F = A - B C B^-1 formalism from the appendix of 
    #Matsubara (arXiv:astro-ph/0408349) to obtain the marginalised matrix 


    size = np.shape(FMin)[0]
    A = np.zeros((len(inds),len(inds))) # the matrix of our parameters
    B = np.zeros((len(inds),size-len(inds)))
    C = np.zeros((size-len(inds), size-len(inds))) # the matrix of nuisance parameters
    
    
# Computing A, B, C
    print inds

    for i in range(len(inds)):
        for j in range(len(inds)):
            A[i,j] = FMin[int(inds[i]),int(inds[j])]
            
    for i in range(len(non_inds)):
        for j in range(len(non_inds)):
            C[i,j] = FMin[int(non_inds[i]), int(non_inds[j])]
                    
    for i in range(len(inds)):
        for j in range(len(non_inds)):
            B[i,j] = FMin[int(inds[i]),int(non_inds[j])]

    A = np.mat(A)
    B = np.mat(B)
    C = np.mat(C)


    FMout = A - B*C.I*B.T

    return FMout

FM_out = margMat(ind, non_ind, F)

C = np.linalg.inv(FM_out)
    



#redefine funcs to exclude whatever indices you marginalized over

# order is om, H, w, M,    , make sure order of funcs matches order of columns in emcee output

funcs = ['dlogfdom', 'dlogfdw']
#funcs = ['dlogfdom', 'dlogfdH','dlogfdw','dlogfdM']




def plotting(a,k): #where a and b correspond to the parameter you want in quotes ie. "u2"
    a = [i for i,s in enumerate(funcs) if a in s][0]
    k = [i for i,s in enumerate(funcs) if k in s][0]
    
    
    B = np.zeros((2,2))
    B[0,0] = C[a,a]
    B[1,0] = C[a,k]
    B[0,1] = C[k,a]
    B[1,1] = C[k,k]
    
    W,v=np.linalg.eigh(B)
    #angle = -156. #this is what it should be for w om
    
    angle=180*np.arctan2(v[1,0],v[0,0])/np.pi
    
   
    
    a_1s=np.sqrt(2.3*np.abs(W[0])) #68% confidence
    b_1s=np.sqrt(2.3*np.abs(W[1]))
    a_2s=np.sqrt(6.17*np.abs(W[0])) #95% confidence 
    b_2s=np.sqrt(6.17*np.abs(W[1]))
      
    centre = [eval(funcs[int(a)].split('dlogfd')[1]), eval(funcs[int(k)].split('dlogfd')[1])]
    
    print W

    
    e_1s=Ellipse(xy=centre,width=2*a_1s,height=2*b_1s,angle=angle,
                        facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='aqua', label = '$68$% $confidence$')
    #                     
    
    e_2s=Ellipse(xy=centre,width=2*a_2s,height=2*b_2s,angle=angle,
                        facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='blue',  label = '$95$% $confidence$')
    #                     facecolor=fc[i],linewidth=lw[i],linestyle=ls[i],edgecolor=lc[i])
    
   
    
    emcee_file = os.path.join(os.path.expanduser('~/cosmosis/'), 'emcee_noBEAMS.txt')
    
    emcee = np.loadtxt(emcee_file, unpack = True)
    
    emcee = emcee[ind]
    
    inside_sig1 = []
    inside_sig2 = []

    
    for i in range(0,len(emcee[a][24000:-1])):
        if ((emcee[a][i] - centre[0])/a_1s)**2 + ((emcee[k][i] - centre[1])/b_1s)**2 < 1:
            inside_sig1.append(i)  
  
    for i in range(0,len(emcee[a][24000:-1])):
        if ((emcee[a][i]- centre[0])/a_2s)**2 + ((emcee[k][i] - centre[1])/b_2s)**2 < 1:
            inside_sig2.append(i)
    
    print "1 sigma", float(len(inside_sig1))/len(emcee[a][24000:-1])
    print "2 sigma", float(len(inside_sig2))/len(emcee[k][24000:-1])
    
    
    ax = plt.gca()
    
    
    #this axis plotting prioritizes emcee
    
   
    #plt.axis([min(emcee[a]), max(emcee[a]), min(emcee[k]), max(emcee[k])]) 
    #plt.plot()
    
    plt.axis([0.1, 0.5, -2, -0.25]) 
    plt.plot()

   
    plt.xlabel(funcs[int(a)].split('dlogfd')[1])
    plt.ylabel(funcs[int(k)].split('dlogfd')[1])
    ax.scatter(emcee[a][19000:-1], emcee[k][19000:-1],  c = 'purple', s = 0.2, alpha = 0.1) #this plots the MCMC point from CosmoSIS if you have them to compare to
    ax.add_patch(e_1s)
    ax.add_patch(e_2s)
    
    
    ax.legend()
    
    plt.show()
    

