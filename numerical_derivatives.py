import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib.patches import Ellipse

from astropy.cosmology.funcs1 import distmod
#I'll need to put in cosmology because I took it all out of the source code

import os


#best fit cosmological parameters

M = -19.3714

om  =  0.277073 #omega_m

H  =  69.5532

w =  -0.978193

omega_b  =  0.04

b = 1.01035

sign1a = 0.0385345


#lets do some mu calculations
from astropy.cosmology import Flatw0waCDM
#set a specific cosmology

cos = Flatw0waCDM(H, om, w0= w, Ob0 = omega_b)

data_file = os.path.join(os.path.expanduser('~/cosmosis/cosmosis-standard-library/supernovae/simplechi2/data/Shafer2'), 'test_out.txt')

datamat = np.loadtxt(data_file, unpack = True)

zhel = datamat[2] # redshift 
zcmb = datamat[1] # redshift
mb = datamat[4] 
dmb = datamat[5]
p1a = datamat[-1]


def f(mb, dmb, zcmb, p1a, M, H, om, w, b, sign1a):
    cos = Flatw0waCDM(H, om, w0= w, Ob0 = omega_b)
    mu_theory = distmod(zcmb, cosmo=cos)  
    mu_theory = mu_theory.value
    
    mu_obs= mb - float(M)
    
    diffvec1a = (mu_obs-mu_theory)
       
    diffvecn1a = (mu_obs - mu_theory - float(b))

    c1a = diffvec1a**2/dmb**2
    cnon1a = diffvecn1a**2/float(sign1a)**2
    
    rt = []
    for i in range(40):
        ps = Decimal(p1a[i])
        f1 = Decimal(1./(np.sqrt(2*np.pi)*dmb[i]))*Decimal(-c1a[i]/2).exp()
        f2 = Decimal(1./(np.sqrt(2*np.pi)*float(sign1a)))*Decimal(-cnon1a[i]/2).exp()
        rt.append(float(ps*f1 + (Decimal(1.)-ps)*f2))
    
    return np.array(rt)

def log_f(mb, dmb, zcmb, p1a, M, H, om, w, b, sign1a):
    cos = Flatw0waCDM(H, om, w0= w, Ob0 = omega_b)
    mu_theory = distmod(zcmb, cosmo=cos)  
    mu_theory = mu_theory.value
    
    mu_obs= mb - float(M)
    
    diffvec1a = (mu_obs-mu_theory)
       
    diffvecn1a = (mu_obs - mu_theory - float(b))

    c1a = diffvec1a**2/dmb**2
    cnon1a = diffvecn1a**2/float(sign1a)**2
    #f1 = (1./(np.sqrt(2*np.pi)*dmb))*np.exp(-c1a/2)
    #f2 = (1./(np.sqrt(2*np.pi)*sign1a))*np.exp(-cnon1a/2)
    
    rt = []
    for i in range(40):
        ps = Decimal(p1a[i])
        f1 = Decimal(1./(np.sqrt(2*np.pi)*dmb[i]))*Decimal(-c1a[i]/2).exp()
        f2 = Decimal(1./(np.sqrt(2*np.pi)*float(sign1a)))*Decimal(-cnon1a[i]/2).exp()
        rt.append(float(Decimal.ln(ps) + Decimal.ln(f1) + Decimal.ln(Decimal(1.) + (Decimal(1.)-ps)*f2/(ps*f1))))
    
    
    return np.array(rt)


#this naming convention is important, do dlogfd + whatever the name of the parameter is 

def dlogfdM(mb): 
    h = 1e-8 #arbitrarily small step for analytical derivative
    return (log_f(mb, dmb, zcmb, p1a, M +h, H, om, w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, H, om, w, b, sign1a))/h

def dlogfdH(mb): 
    h = 1e-8 #arbitrarily small step for analytical derivative
    return (log_f(mb, dmb, zcmb, p1a, M , H +h, om, w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, H, om, w, b, sign1a))/h

def dlogfdom(mb): 
    h = 1e-8 #arbitrarily small step for analytical derivative
    return (log_f(mb, dmb, zcmb, p1a, M , H, om + h, w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, H, om, w, b, sign1a))/h

def dlogfdw(mb): 
    h = 1e-8 #arbitrarily small step for analytical derivative
    return (log_f(mb, dmb, zcmb, p1a, M, H, om, w + h, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, H, om, w, b, sign1a))/h

def dlogfdb(mb): 
    h = 1e-8 #arbitrarily small step for analytical derivative
    return (log_f(mb, dmb, zcmb, p1a, M, H, om, w, b + h, sign1a) - log_f(mb, dmb, zcmb, p1a, M, H, om, w, b, sign1a))/h

def dlogfdsign1a(mb): 
    h = 1e-8 #arbitrarily small step for analytical derivative
    return (log_f(mb, dmb, zcmb, p1a, M , H, om, w, b, sign1a + h) - log_f(mb, dmb, zcmb, p1a, M, H, om, w, b, sign1a))/h

# order is om, H, w, M, b, sign1a , make sure order of funcs matches order of columns in emcee output

funcs = ['dlogfdom', 'dlogfdH', 'dlogfdw','dlogfdM', 'dlogfdb', 'dlogfdsign1a']


I = np.zeros((len(funcs),len(funcs)))


for i in range(len(funcs)):
    for j in range(len(funcs)):
        I[i,j] = np.trapz(eval(funcs[i])(mb)*eval(funcs[j])(mb)*f(mb, dmb, zcmb, p1a, M, H, om, w, b, sign1a), mb)

F = np.linalg.inv(I)


def plotting(a,k): #where a and b correspond to the parameter you want in quotes ie. "u2"
    a = [i for i,s in enumerate(funcs) if a in s][0]
    k = [i for i,s in enumerate(funcs) if k in s][0]
    
    
    B = np.zeros((2,2))
    B[0,0] = F[a,a]
    B[1,0] = F[a,k]
    B[0,1] = F[k,a]
    B[1,1] = F[k,k]
    
    W,v=np.linalg.eigh(B)
    angle=180*np.arctan2(v[1,0],v[0,0])/np.pi
    
    a_1s=np.sqrt(2.3*W[0]) #68% confidence
    b_1s=np.sqrt(2.3*W[1])
    a_2s=np.sqrt(6.17*W[0]) #95% confidence 
    b_2s=np.sqrt(6.17*W[1])
      
    centre = [eval(funcs[int(a)].split('dlogfd')[1]), eval(funcs[int(k)].split('dlogfd')[1])]
    
    print a_1s
    print b_1s
    
    e_1s=Ellipse(xy=centre,width=2*a_1s,height=2*b_1s,angle=angle,
                        facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='aqua', label = '$68$% $confidence$')
    #                     
    
    e_2s=Ellipse(xy=centre,width=2*a_2s,height=2*b_2s,angle=angle,
                        facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='blue',  label = '$95$% $confidence$')
    #                     facecolor=fc[i],linewidth=lw[i],linestyle=ls[i],edgecolor=lc[i])
    
   
    
    emcee_file = os.path.join(os.path.expanduser('~/cosmosis/'), 'emcee_simplebeams.txt')
    
    emcee = np.loadtxt(emcee_file, unpack = True)
    
    
    ax = plt.gca()
    
    #this axis plotting prioritizes emcee
    
    plt.axis([min(emcee[a]), max(emcee[a]), min(emcee[k]), max(emcee[k])]) 
    plt.plot()
    
    
    #this plotting prioritizes the ellipses
    
    '''
    if -0.02 < np.cos(np.deg2rad(angle))  < 0.02:
        plt.axis([centre[0]- 2*b_2s, centre[0] + 2*b_2s, centre[1] - 2*a_2s, centre[1]+ 2*a_2s]) 
        plt.plot()
    elif -0.02 < np.sin(np.deg2rad(angle))  < 0.02 :
        plt.axis([centre[0]- 2*a_2s, centre[0] + 2*a_2s, centre[1] - 2*b_2s, centre[1]+ 2*b_2s]) 
        plt.plot()
        
    else:
        plt.axis([centre[0]- np.abs(2*np.cos(np.deg2rad(angle))*a_2s), centre[0] +np.abs(2*np.cos(np.deg2rad(angle))*a_2s), centre[1] - np.abs(2*np.cos(np.deg2rad(angle))*b_2s), centre[1]+ np.abs(2*np.cos(np.deg2rad(angle))*b_2s)]) 
        plt.plot()
    '''   
   
    
    

    plt.xlabel(funcs[int(a)].split('dlogfd')[1])
    plt.ylabel(funcs[int(k)].split('dlogfd')[1])
    ax.scatter(emcee[a][8000:-1], emcee[k][8000:-1],  c = 'purple', s = 0.2, alpha = 0.1) #this plots the MCMC point from CosmoSIS if you have them to compare to
    ax.add_patch(e_1s)
    ax.add_patch(e_2s)
    
    
    ax.legend()
    
    plt.show()
        
        
    
