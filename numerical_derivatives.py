import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib.patches import Ellipse

from astropy.cosmology.funcs1 import distmod
#I'll need to put in cosmology because I took it all out of the source code

import os


#best fit big_priors
'''
om   =     0.301852
H    =  100*0.719383
H = 71.9375
w=       -0.896407
M       = -19.4376
b       = 9.99243
sign1a      =  0.176808

#means

om  = 3.022695e-01   
H  = 100*7.137710e-01  
w   =-8.971287e-01  
M  =-1.945937e+01  
b  = 9.991584e+00  
sign1a =  1.905395e-01   


#best fit new_fixed4both
H = 70.
om     =   0.302944
w      =  -1.0085
M      = -19.5009
b      =  1.00877
sign1a   =     0.050195
'''

#MEANS


H = 70.
om =  3.027453e-01   
w   =-1.007951e+00   
M   =-1.950086e+01   
b   =1.007731e+00   
sign1a =  6.308897e-02   



omega_b = 0.04

#lets do some mu calculations
from astropy.cosmology import Flatw0waCDM

#set a specific cosmology

cos = Flatw0waCDM(H, om, w0= w, Ob0 = omega_b)

#data_file = os.path.join(os.path.expanduser('~/cosmosis/cosmosis-standard-library/supernovae/simplechi2/data/Shafer2'), 'new_6*dmb_fixed4both_gaussian_errors_test_out.txt')
data_file = os.path.join(os.path.expanduser('~/cosmosis/cosmosis-standard-library/supernovae/simplechi2/data/Shafer2'), 'new_test_out.txt')


datamat = np.loadtxt(data_file, unpack = True)
data_new = np.zeros((40, 4)) 
data_new[:,0] = datamat[4]
data_new[:,1] = datamat[5]
data_new[:,2] = datamat[1]
data_new[:,3] = datamat[-1]
datamat2 = data_new[np.lexsort(np.fliplr(data_new).T)]


zhel = datamat[2] # redshift 
zcmb = datamat[1] # redshift
mb = datamat[4]
dmb = datamat[5]
p1a = datamat[-1]
'''

mb = datamat2[:,0] 
dmb = datamat2[:,1]
p1a = datamat2[:,3]
zcmb = datamat2[:,2]
'''

mb_og = []
mb_shift = []

for i in range(0,len(mb)):
    if p1a[i]>0.2:
        mb_og.append(int((np.where(mb == mb[i]))[0]))
    else:
        mb_shift.append(int((np.where(mb == mb[i]))[0]))


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
    for i in range(len(mb)):
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
       
    diffvecn1a = (mu_obs - mu_theory- float(b))

    c1a = diffvec1a**2/dmb**2
    cnon1a = diffvecn1a**2/float(sign1a)**2
    #f1 = (1./(np.sqrt(2*np.pi)*dmb))*np.exp(-c1a/2)
    #f2 = (1./(np.sqrt(2*np.pi)*sign1a))*np.exp(-cnon1a/2)
    
    rt = []
    for i in range(len(mb)):
        ps = Decimal(p1a[i])
        f1 = Decimal(1./(np.sqrt(2*np.pi)*dmb[i]))*Decimal(-c1a[i]/2).exp()
        f2 = Decimal(1./(np.sqrt(2*np.pi)*float(sign1a)))*Decimal(-cnon1a[i]/2).exp()
        rt.append(float(Decimal.ln(ps) + Decimal.ln(f1) + Decimal.ln(Decimal(1.) + (Decimal(1.)-ps)*f2/(ps*f1)))) #maybe this is the float statement that needs to go?
    
    
    return np.array(rt)
    

#this naming convention is important, do dlogfd + whatever the name of the parameter is 
def dlogfdM(mb, dmb, zcmb, p1a): 
    h = 1e-6
    return (log_f(mb, dmb, zcmb, p1a, M + h, H, om, w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, H, om, w, b, sign1a))/h
        
def dlogfdH(mb, dmb, zcmb, p1a): 
     h = 1e-6
     return (log_f(mb, dmb, zcmb, p1a, M , H +h, om, w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, H, om, w, b, sign1a))/h

def dlogfdom(mb, dmb, zcmb, p1a):
     h = 1e-6
     return (log_f(mb, dmb, zcmb, p1a, M , H, om + h, w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, H, om, w, b, sign1a))/h

def dlogfdw(mb, dmb, zcmb, p1a):
     h = 1e-6
     return (log_f(mb, dmb, zcmb, p1a, M, H, om, w + h, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, H, om, w, b, sign1a))/h
        
def dlogfdb(mb, dmb, zcmb, p1a): 
     h = 1e-6
     return (log_f(mb, dmb, zcmb, p1a, M, H, om, w, b + h, sign1a) - log_f(mb, dmb, zcmb, p1a, M, H, om, w, b, sign1a))/(h)
        
def dlogfdsign1a(mb, dmb, zcmb, p1a):
     h = 1e-6
     return (log_f(mb, dmb, zcmb, p1a, M , H, om, w, b, sign1a + h) - log_f(mb, dmb, zcmb, p1a, M, H, om, w, b, sign1a))/(h)
        
# order is om, H, w, M, b, sign1a , make sure order of funcs matches order of columns in emcee output
'''
Ms = []
Hs = []
oms = []
ws = []
bs = []
sigs = []


hs = np.logspace(-9, 1, 100, base = 10.)
for h in hs:
    Ms.append(np.trapz((eval('dlogfdM')(mb , dmb , zcmb , p1a ))**2*f(mb , dmb , zcmb , p1a , M, H, om, w, b, sign1a), zcmb))
    Hs.append(np.trapz((eval('dlogfdH')(mb , dmb , zcmb , p1a ))**2*f(mb , dmb , zcmb , p1a , M, H, om, w, b, sign1a), zcmb))
    oms.append(np.trapz((eval('dlogfdom')(mb , dmb , zcmb , p1a ))**2*f(mb , dmb , zcmb , p1a , M, H, om, w, b, sign1a), zcmb))
    ws.append(np.trapz((eval('dlogfdw')(mb , dmb , zcmb , p1a ))**2*f(mb , dmb , zcmb , p1a , M, H, om, w, b, sign1a), zcmb))
    bs.append(np.trapz((eval('dlogfdb')(mb[mb_shift] , dmb[mb_shift]  , zcmb[mb_shift]  , p1a[mb_shift]  ))**2*f(mb[mb_shift] , dmb[mb_shift] , zcmb[mb_shift] , p1a[mb_shift] , M, H, om, w, b, sign1a), zcmb[mb_shift]))
    sigs.append(np.trapz((eval('dlogfdsign1a')(mb[mb_shift]  , dmb[mb_shift]  , zcmb[mb_shift]  , p1a[mb_shift]  ))**2*f(mb[mb_shift] , dmb[mb_shift] , zcmb[mb_shift] , p1a[mb_shift] , M, H, om, w, b, sign1a), zcmb[mb_shift]))


plt.plot(hs, Ms, 'ro')
plt.xscale('log')
plt.yscale('log')
plt.show()

'''

def jagged_integrate(x,y):
    '''
    This integrates a jagged function that has a certain number of outputs (y) and an
    equal number of independent variables, x.
    '''
    area = 0
    for i in range(0,(len(x) -1)):
        area += min(y[i], y[i+1])*(x[i+1] - x[i]) + 0.5*((x[i+1] - x[i]))*(max(y[i], y[i+1]) - min(y[i], y[i+1]))
        
    return area


funcs = ['dlogfdom','dlogfdw','dlogfdM', 'dlogfdb', 'dlogfdsign1a']


F = np.zeros((len(funcs),len(funcs)))



for i in range(len(funcs)):
    for j in range(len(funcs)):
        F[i,j] = np.trapz(eval(funcs[i])(mb , dmb , zcmb , p1a )*eval(funcs[j])(mb , dmb , zcmb , p1a )*f(mb , dmb , zcmb , p1a , M, H, om, w, b, sign1a), zcmb)

'''
for i in range(3,5):
    for j in range(3,5):
        F[i,j] = np.trapz(eval(funcs[i])(mb , dmb , zcmb , p1a )*eval(funcs[j])(mb , dmb , zcmb , p1a )*f(mb , dmb , zcmb , p1a , M, H, om, w, b, sign1a), mb )


for i in range(3,5):
    for j in range(3,5):
        F[i,j] = np.trapz(eval(funcs[i])(mb[mb_shift] , dmb[mb_shift] , zcmb[mb_shift] , p1a[mb_shift] )*eval(funcs[j])(mb[mb_shift] , dmb[mb_shift] , zcmb[mb_shift] , p1a[mb_shift] )*f(mb[mb_shift] , dmb[mb_shift] , zcmb[mb_shift] , p1a[mb_shift] , M, H, om, w, b, sign1a), mb[mb_shift] )
'''


ind = [0,1,3,4]
non_ind = [2]

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

C = np.linalg.inv(F)
    



#redefine funcs to exclude whatever indices you marginalized over

# order is om, H, w, M, b, sign1a , make sure order of funcs matches order of columns in emcee output

funcs = ['dlogfdom', 'dlogfdw','dlogfdM', 'dlogfdb', 'dlogfdsign1a']



def plotting(a,k): #where a and b correspond to the parameter you want in quotes ie. "u2"
    a = [i for i,s in enumerate(funcs) if a in s][0]
    k = [i for i,s in enumerate(funcs) if k in s][0]
    
    
    B = np.zeros((2,2))
    B[0,0] = C[a,a]
    B[1,0] = C[a,k]
    B[0,1] = C[k,a]
    B[1,1] = C[k,k]
    
    W,v=np.linalg.eigh(B)
    a#ngle = -164.5 #this is what it should be for w om
    
    print W
    angle=180*np.arctan2(v[1,0],v[0,0])/np.pi
    #angle = -154. #this is what it should be for w om
   
    
    a_1s=np.sqrt(2.3*W[0]) #68% confidence
    b_1s=np.sqrt(2.3*W[1])
    a_2s=np.sqrt(6.17*W[0]) #95% confidence 
    b_2s=np.sqrt(6.17*W[1])
    
    print a_1s, '', a_2s, '', b_1s, '', b_2s 
      
    centre = [eval(funcs[int(a)].split('dlogfd')[1]), eval(funcs[int(k)].split('dlogfd')[1])]
    
    
    
    e_1s=Ellipse(xy=centre,width=2*a_1s,height=2*b_1s,angle=angle,
                        facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='aqua', label = '$68$% $confidence$')
    #                     
    
    e_2s=Ellipse(xy=centre,width=2*a_2s,height=2*b_2s,angle=angle,
                        facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='blue',  label = '$95$% $confidence$')
    #                     facecolor=fc[i],linewidth=lw[i],linestyle=ls[i],edgecolor=lc[i])
    
   
    
    emcee_file = os.path.join(os.path.expanduser('~/cosmosis/'), 'new_fixed4both.txt') #old one was new_bigpriors_fixed4both.txt
    
    emcee = np.loadtxt(emcee_file, unpack = True)
    
    emcee = emcee
    
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
    
   
    plt.axis([min(emcee[a]), max(emcee[a]), min(emcee[k]), max(emcee[k])]) 
    plt.plot()

    
    #this plotting prioritizes the ellipses
 



    #if -0.02 < np.cos(np.deg2rad(angle))  < 0.02:
     #   plt.axis([centre[0]- 2*b_2s, centre[0] + 2*b_2s, centre[1] - 2*a_2s, centre[1]+ 2*a_2s]) 
      #  plt.plot()
    #elif -0.02 < np.sin(np.deg2rad(angle))  < 0.02 :
     #   plt.axis([centre[0]- 2*a_2s, centre[0] + 2*a_2s, centre[1] - 2*b_2s, centre[1]+ 2*b_2s]) 
      #  plt.plot()
        
    #else:
     #   plt.axis([centre[0]- np.abs(2*np.cos(np.deg2rad(angle))*a_2s), centre[0] +np.abs(2*np.cos(np.deg2rad(angle))*a_2s), centre[1] - np.abs(2*np.cos(np.deg2rad(angle))*b_2s), centre[1]+ np.abs(2*np.cos(np.deg2rad(angle))*b_2s)]) 
      #  plt.plot()
    

   
    plt.xlabel(funcs[int(a)].split('dlogfd')[1])
    plt.ylabel(funcs[int(k)].split('dlogfd')[1])
    ax.scatter(emcee[a][24000:-1], emcee[k][24000:-1],  c = 'purple', s = 0.2, alpha = 0.1) #this plots the MCMC point from CosmoSIS if you have them to compare to
    ax.add_patch(e_1s)
    ax.add_patch(e_2s)
    
    
    ax.legend()
    
    plt.show()
    


'''

#seeing how many points fall into the fisher ellipses

emcee_file = os.path.join(os.path.expanduser('~/cosmosis/'), 'new_fixed4both.txt') #old one was new_bigpriors_fixed4both.txt
    
emcee = np.loadtxt(emcee_file, unpack = True)

inside_sig1 = []
inside_sig2 = []

a_1s =0.022327647594901176
a_2s = 0.036569719814071734
b_1s = 0.031104235102512607 
b_2s = 0.05094460389950164
    
for i in range(0,len(emcee[0][24000:-1])):
    if ((emcee[4][i] - sign1a)/a_1s)**2 + ((emcee[3][i] - b)/b_1s)**2 < 1:
        inside_sig1.append(i)  
  
for i in range(0,len(emcee[0][24000:-1])):
    if ((emcee[4][i]- sign1a)/a_2s)**2 + ((emcee[3][i] - b)/b_2s)**2 < 1:
        inside_sig2.append(i)
    
print "1 sigma", float(len(inside_sig1))/len(emcee[4][24000:-1])
print "2 sigma", float(len(inside_sig2))/len(emcee[4][24000:-1])
'''

#for om and w

'''

#seeing how many points fall into the fisher ellipses

emcee_file = os.path.join(os.path.expanduser('~/cosmosis/'), 'new_fixed4both.txt') #old one was new_bigpriors_fixed4both.txt
    
emcee = np.loadtxt(emcee_file, unpack = True)

inside_sig1 = []
inside_sig2 = []

a_1s =0.0011737548728505934
a_2s =  0.0019224545106287796
b_1s = 0.0187162822298437 
b_2s = 0.030654783232192232
    
for i in range(0,len(emcee[0][24000:-1])):
    if ((emcee[0][i] - om)/a_1s)**2 + ((emcee[1][i] - w)/b_1s)**2 < 1:
        inside_sig1.append(i)  
  
for i in range(0,len(emcee[0][24000:-1])):
    if ((emcee[0][i]- om)/a_2s)**2 + ((emcee[1][i] - w)/b_2s)**2 < 1:
        inside_sig2.append(i)
    
print "1 sigma", float(len(inside_sig1))/len(emcee[4][24000:-1])
print "2 sigma", float(len(inside_sig2))/len(emcee[4][24000:-1])
'''  

