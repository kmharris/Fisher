import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib.patches import Ellipse

from astropy.cosmology.funcs1 import distmod
#I'll need to put in cosmology because I took it all out of the source code

import os


#cosmological parameters - means
M = -1.934060e+01

omega_m  =  2.742863e-01

h0  =  70.51156

w =  -9.811662e-01

omega_b  =  0.04

b = 1.012254e+00

sign1a = 4.547964e-02

#best fit 

M = -19.3714

omega_m  =  0.277073

h0  =  69.5532

w =  -0.978193

omega_b  =  0.04

b = 1.01035

sign1a = 0.0385345


#lets do some mu calculations
from astropy.cosmology import Flatw0waCDM
#set a specific cosmology

cos = Flatw0waCDM(h0, omega_m, w0= w, Ob0 = omega_b)

data_file = os.path.join(os.path.expanduser('~/cosmosis/cosmosis-standard-library/supernovae/simplechi2/data/Shafer2'), 'test_out.txt')

datamat = np.loadtxt(data_file, unpack = True)

zhel = datamat[2] # redshift 
zcmb = datamat[1] # redshift
mb = datamat[4] 
dmb = datamat[5]
p1a = datamat[-1]
''' 

#not working with covariance matrices right now
tcov = np.loadtxt('sys_DS17f_G10_0.txt', unpack=True) 
size=np.int(tcov[0]) # reading the size of the matrix from the first entry                                                    
cov1a = np.zeros((size,size))
count=1 # since the cov mat starts with the number of lines 

for i in range(size):
    for j in range(size):
        cov1a[i,j]=tcov[count]
        count +=1

for i in range(size):            
    cov1a[i,i]+=(dmb[i])**2




#this was testing the trapezoidal rule

b = np.arange(0,1.,0.0001)

def f(x):
   return x**2


t = np.trapz(f(b), b) # going to do it using this! nice, trapezoidal rule!
#as I decrease the dx b = np.arange(0,1.,0.0001), it approximates the actual mp_quad

t1 = mp_quad(f, [0, 1.0], args=())
'''

#this next part was the old way to get mu, in here for posterity

'''
#calculate true mu from look up table 
#mu_theory=np.interp(zcmb,z_model_table,mu_model_table)
    
#I don't actually know how this works, so I'm going to do the astropy and compare to the cosmosis one

z_model_table = z_model_table[1:]
mu_model_table = mu_model_table[1:]

# I know how it works now, bascially they make a function so you can pull out a mu value, but I'll set an actual cosmology

interpf = interp1d(z_model_table, mu_model_table, kind='cubic')
mu_theory  = interpf(zcmb)  



#calculate mu using astropy
mu_theory = distmod(zcmb, cosmo=cos)  
mu_theory = mu_theory.value

#calculate mu_obs from data and M0, alpha, beta

mu_obs= mb - M
    
diffvec1a = (mu_obs-mu_theory)
       
diffvecn1a = (mu_obs - mu_theory - b)




#what the mixed distribution should look like

p1a = 0.7
b = 3.0
sig1 = 0.5
sig2 = 0.5
x = np.arange(-5,5,0.1)
c1a = x**2/sig1**2
cnon1a = (x - b)**2/(sig2)**2
f1 = (1./(np.sqrt(2.*np.pi)*sig1))*np.exp(-c1a/2.)
f2 = (1./(np.sqrt(2.*np.pi)*sig2))*np.exp(-cnon1a/2.)
y = p1a*f1 + (1.-p1a)*f2

'''


def f(mb, dmb, zcmb, p1a, M, h0, omega_m, w, b, sign1a):
    cos = Flatw0waCDM(h0, omega_m, w0= w, Ob0 = omega_b)
    mu_theory = distmod(zcmb, cosmo=cos)  
    mu_theory = mu_theory.value
    
    mu_obs= mb - M
    
    diffvec1a = (mu_obs-mu_theory)
       
    diffvecn1a = (mu_obs - mu_theory - b)

    c1a = diffvec1a**2/dmb**2
    cnon1a = diffvecn1a**2/(sign1a)**2
    f1 = (1./(np.sqrt(2.*np.pi)*dmb))*np.exp(-c1a/2.)
    f2 = (1./(np.sqrt(2.*np.pi)*sign1a))*np.exp(-cnon1a/2.)
    return p1a*f1 + (1.-p1a)*f2

def log_f(mb, dmb, zcmb, p1a, M, h0, omega_m, w, b, sign1a):
    cos = Flatw0waCDM(h0, omega_m, w0= w, Ob0 = omega_b)
    mu_theory = distmod(zcmb, cosmo=cos)  
    mu_theory = mu_theory.value
    
    mu_obs= mb - M
    
    diffvec1a = (mu_obs-mu_theory)
       
    diffvecn1a = (mu_obs - mu_theory - b)

    c1a = diffvec1a**2/dmb**2
    cnon1a = diffvecn1a**2/(sign1a)**2
    f1 = (1./(np.sqrt(2.*np.pi)*dmb))*np.exp(-c1a/2.)
    f2 = (1./(np.sqrt(2.*np.pi)*sign1a))*np.exp(-cnon1a/2.)
    
    return np.log(p1a) + np.log(f1) + np.log(1. + (1.-p1a)*f2/(p1a*f1))


    
#this outputs forty points, which is why the trapezoidal rule will work but the others won't
   
#just comment out what I don't need


#this naming convention is important, do dlogfd + whatever the name of the parameter is 

def dlogfdu1(x): #u2
    h = 1e-8 #arbitrarily small step for analytical derivative
    return (log_f(x,u1 +h,u2,p1,sig1,sig2) - log_f(x,u1,u2,p1,sig1,sig2))/h

def dlogfdu2(x): #u2
    h = 1e-8 #arbitrarily small step for analytical derivative
    return (log_f(x,u1 ,u2 + h,p1,sig1,sig2) - log_f(x,u1,u2,p1,sig1,sig2))/h

def dlogfdp1(x): #u2
    h = 1e-8 #arbitrarily small step for analytical derivative
    return (log_f(x,u1 ,u2 ,p1 +h,sig1,sig2) - log_f(x,u1,u2,p1,sig1,sig2))/h

def dlogfdsig1(x): #u2
    h = 1e-8 #arbitrarily small step for analytical derivative
    return (log_f(x,u1 ,u2 ,p1,sig1 + h,sig2) - log_f(x,u1,u2,p1,sig1,sig2))/h

def dlogfdsig2(x): #u2
    h = 1e-8 #arbitrarily small step for analytical derivative
    return (log_f(x,u1 ,u2,p1,sig1,sig2 +h) - log_f(x,u1,u2,p1,sig1,sig2))/h    

funcs = ['dlogfdu1','dlogfdu2', 'dlogfdsig1', 'dlogfdsig2']

I = np.zeros((len(funcs),len(funcs)))


for i in range(len(funcs)):
    for j in range(len(funcs)):
        I[i,j] = np.trapz(eval(funcs[i])(x)*eval(funcs[j])(x)*f(x,u1,u2,p1,sig1,sig2), x)

F = np.linalg.inv(I)

 
def plotting(a,b): #where a and b correspond to the order of parameters you want in funcs
    C = np.zeros((2,2))
    C[0,0] = F[a,a]
    C[1,0] = F[a,b]
    C[0,1] = F[b,a]
    C[1,1] = F[b,b]
    
    w,v=np.linalg.eigh(C)
    angle=180*np.arctan2(v[1,0],v[0,0])/np.pi
    
    a_1s=np.sqrt(2.3*w[0]) #68% confidence
    b_1s=np.sqrt(2.3*w[1])
    a_2s=np.sqrt(6.17*w[0]) #95% confidence 
    b_2s=np.sqrt(6.17*w[1])
      
    centre = np.array([eval(funcs[a].split('dlogfd')[1]),eval(funcs[b].split('dlogfd')[1])])
    
    e_1s=Ellipse(xy=centre,width=2*a_1s,height=2*b_1s,angle=angle,
                        facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='aqua', label = '$68$% $confidence$')
    #                     
    
    e_2s=Ellipse(xy=centre,width=2*a_2s,height=2*b_2s,angle=angle,
                        facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='blue',  label = '$95$% $confidence$')
    #                     facecolor=fc[i],linewidth=lw[i],linestyle=ls[i],edgecolor=lc[i])
    
    print a_1s
    print b_1s
    
    ax = plt.gca()
    
    
    plt.axis([centre[0]-2*a_2s,centre[0] + 2*a_2s,centre[1] - 2*b_2s ,centre[1]+2*b_2s]) 
    plt.plot()
    
    plt.xlabel(funcs[a].split('dlogfd')[1])
    plt.ylabel(funcs[b].split('dlogfd')[1])
    #ax.scatter(gauss[1][8000:-1], gauss[0][8000:-1],  c = 'purple', s = 0.2, alpha = 0.1) #this plots the MCMC point from CosmoSIS if you have them to compare to
    ax.add_patch(e_1s)
    ax.add_patch(e_2s)
    
    
    ax.legend()
    
    plt.show()
        
    



#this is for M(x axis) and omega_m (y axis)
    
#Order of variables is M, h0, omega_m, w, b, sign1a

#first row 
    
def I_00(mb): #dm^2
    h = 2.842841e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m, w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M + h, h0, omega_m,   w, b, sign1a  ) - log_f(mb, dmb, zcmb, p1a, M - h , h0, omega_m,   w, b, sign1a))/(2.*h))**2

def I_01(mb): #dm*dh0
    hm = 2.842841e-03
    hh = 1.004367e-01
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,  w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M +hm , h0, omega_m,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M - hm , h0, omega_m ,   w, b, sign1a))/(2.*hm))*((log_f(mb, dmb, zcmb, p1a, M, h0 +hh, omega_m  ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 - hh, omega_m ,  w, b , sign1a ))/(2.*hh))
 
def I_02(mb): #dm*domega_m
    hm = 2.842841e-03
    hom = 4.880099e-03
    
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,  w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M +hm , h0, omega_m,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M - hm , h0, omega_m ,   w, b, sign1a))/(2.*hm))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m + hom ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m - hom , w, b , sign1a ))/(2.*hom))
 

def I_03(mb): #dm*dw
    hm = 2.842841e-03
    hw = 1.291811e-02
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,  w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M +hm , h0, omega_m,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M - hm , h0, omega_m ,   w, b, sign1a))/(2.*hm))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w + hw, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m  ,  w - hw, b , sign1a ))/(2.*hw))
 
def I_04(mb): #dm*db
    hm = 2.842841e-03
    hb = 1.380365e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m, w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M +hm , h0, omega_m,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M - hm , h0, omega_m ,   w, b, sign1a))/(2.*hm))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w , b + hb, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m  ,  w , b -hb, sign1a ))/(2.*hb))
 
def I_05(mb): #dm*dsign1a
    hm = 2.842841e-03
    hs = 1.011612e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,  w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M +hm , h0, omega_m,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M - hm , h0, omega_m ,   w, b, sign1a))/(2.*hm))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w , b , sign1a + hs) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m  ,  w , b, sign1a - hs ))/(2.*hs))
 
# second row

def I_10(mb): #dh0*dm
    return I_01(mb)

def I_11(mb): #dh0^2
    h = 1.004367e-01
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0 + h, omega_m,   w, b, sign1a  ) - log_f(mb, dmb, zcmb, p1a, M  , h0 - h, omega_m,   w, b, sign1a))/(2.*h))**2

def I_12(mb): #dh0*domega_m
    hh = 1.004367e-01
    hom = 4.880099e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m +hom,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m - hom,   w, b, sign1a))/(2.*hom))*((log_f(mb, dmb, zcmb, p1a, M, h0 +hh, omega_m  ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 - hh, omega_m ,  w, b , sign1a ))/(2.*hh))
  
def I_13(mb): #dh0*dw
    hh = 1.004367e-01
    hw = 1.291811e-02
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w + hw, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w - hw, b, sign1a))/(2.*hw))*((log_f(mb, dmb, zcmb, p1a, M, h0 +hh, omega_m  ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 - hh, omega_m ,   w, b , sign1a ))/(2.*hh))

def I_14(mb): #dh0*db
    hh = 1.004367e-01
    hb = 1.380365e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w, b +hb , sign1a ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w , b - hb, sign1a))/(2.*hb))*((log_f(mb, dmb, zcmb, p1a, M, h0 +hh, omega_m  ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 - hh, omega_m ,   w, b , sign1a ))/(2.*hh))

def I_15(mb): #dh0*dsign1a
    hh = 1.004367e-01
    hs = 1.011612e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w, b , sign1a + hs ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w , b, sign1a - hs))/(2.*hs))*((log_f(mb, dmb, zcmb, p1a, M, h0 +hh, omega_m  ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 - hh, omega_m ,   w, b , sign1a ))/(2.*hh))

#third row

def I_20(mb): #dM*domega_m
    return I_02(mb)

def I_21(mb): #dh0*domega_m
    return I_12(mb)

def I_22(mb): #domega_m^2
    h = 4.880099e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m + h,   w, b, sign1a  ) - log_f(mb, dmb, zcmb, p1a, M  , h0 , omega_m - h,   w, b, sign1a))/(2.*h))**2

def I_23(mb): #domega_m*dw
    hom = 4.880099e-03
    hw = 1.291811e-02
    
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w + hw, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w - hw, b, sign1a))/(2.*hw))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m + hom ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m - hom ,   w, b , sign1a ))/(2.*hom))
  
def I_24(mb): #domega_m*db
    hom = 4.880099e-03
    hb = 1.380365e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w , b + hb , sign1a ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w , b - hb, sign1a))/(2.*hb))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m + hom ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m - hom ,   w, b , sign1a ))/(2.*hom))
  
def I_25(mb): #domega_m*dsign1a
    hom = 4.880099e-03
    hs = 1.011612e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w , b  , sign1a +hs) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w , b , sign1a - hs))/(2.*hs))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m + hom ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m - hom ,   w, b , sign1a ))/(2.*hom))
  
#fourth row


def I_30(mb): #dM*dw
    return I_03(mb)

def I_31(mb): #dw*dh0
    return I_13(mb)

def I_32(mb): #dw*domega_m
    return I_23(mb)

def I_33(mb): #dw^2
    h = 1.291811e-02
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w + h, b, sign1a  ) - log_f(mb, dmb, zcmb, p1a, M  , h0 , omega_m,   w - h, b, sign1a))/(2.*h))**2

def I_34(mb): #dw*db
    hw = 1.291811e-02
    hb = 1.380365e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b +hb, sign1a ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w, b - hb, sign1a))/(2.*hb))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w + hw, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m  ,  w - hw, b , sign1a ))/(2.*hw))

def I_35(mb): #dw*dsign1a
    hw = 1.291811e-02
    hs = 1.011612e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a +hs) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w, b, sign1a - hs))/(2.*hs))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w + hw, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m  ,  w - hw, b , sign1a ))/(2.*hw))

#fifth row 
    
def I_40(mb): #dM*db
    return I_04(mb)

def I_41(mb): #dh0*db
    return I_14(mb)

def I_42(mb): #domega_m*db
    return I_24(mb)

def I_43(mb): #dw*db
    return I_34(mb)

def I_44(mb): #db^2
    h = 1.380365e-03
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w , b + h, sign1a  ) - log_f(mb, dmb, zcmb, p1a, M  , h0 , omega_m,   w , b - h, sign1a))/(2.*h))**2

def I_45(mb): #dsign1a*db
    hb = 1.380365e-03
    #hs = 1.011612e-03
    hs = 2.111612e-02
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w, b , sign1a +hs) - log_f(mb, dmb, zcmb, p1a, M  , h0, omega_m ,   w, b, sign1a - hs))/(2.*hs))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w , b + hb, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m , w , b -hb, sign1a ))/(2.*hb))
 
#sixth row 
 
def I_50(mb): #dsign1a*dM
    return I_05(mb)

def I_51(mb): #dsign1a*dh0
    return I_15(mb)

def I_52(mb): #dsign1a*domega_m
    return I_25(mb)

def I_53(mb): #dsign1a*dw
    return I_35(mb)

def I_54(mb): #dsign1a*db
    return I_45(mb)

def I_55(mb): #dsign1a^2
    #h = 1.011612e-03
    h = 9.833985e-04
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w , b, sign1a +h ) - log_f(mb, dmb, zcmb, p1a, M  , h0 , omega_m,   w , b, sign1a - h ))/(2.*h))**2



F = np.zeros((6,6)) #empty matrix that will become the fisher matrix


for i in range(6):
    for j in range(6):
        F[i,j] = np.trapz(eval('I_'+'{:1.0f}'.format(i) + '{:1.0f}'.format(j))(mb), mb) #uses trapz to integrate over data vector mb
    


'''
# this is for M (x axis) and omega_m (yaxis), test the cosmosis run with just these things

def I_00(mb):
    h = 1e-5
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m, w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M + h, h0, omega_m, w, b, sign1a  ) - log_f(mb, dmb, zcmb, p1a, M - h , h0, omega_m,   w, b, sign1a))/(2.*h))**2

def I_01(mb):
    h2 = 1e-5
    
    h1 = 1e-5
    #h2 = 1.45e-3
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M +h1 , h0, omega_m, w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M - h1 , h0, omega_m ,   w, b, sign1a))/(2.*h1))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m +h2 ,   w , b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0, omega_m -h2 ,   w , b , sign1a ))/(2.*h2))
 
def I_11(mb):
    h = 1e-5
    #h = 1.45e-3
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m +h , w , b , sign1a) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m - h,   w  , b , sign1a ))/(2.*h))**2


T = np.zeros((2,2))

T[0,0] = np.trapz(I_00(mb), mb) #not sure if I should be integrating with mb, I'll ask renee
T[0,1] = np.trapz(I_01(mb), mb) 
T[1,0] = T[0,1]
T[1,1] = np.trapz(I_11(mb), mb)

i = np.linalg.inv(T)
'''

C = np.linalg.inv(F)

C_M_omega_m = np.zeros((2,2))
C_M_omega_m[0,0] = C[0,0]
C_M_omega_m[0,1] = C[0,2]
C_M_omega_m[1,0] = C_M_omega_m[0,1]
C_M_omega_m[1,1] = C[2,2]

C_M_w = np.zeros((2,2))
C_M_w[0,0] = C[0,0]
C_M_w[0,1] = C[0,3]
C_M_w[1,0] = C_M_w[0,1]
C_M_w[1,1] = C[3,3]


C_w_omega_m = np.zeros((2,2))
C_w_omega_m[0,0] = C[3,3]
C_w_omega_m[0,1] = C[2,3]
C_w_omega_m[1,0] = C_w_omega_m[0,1]
C_w_omega_m[1,1] = C[2,2]

C_sign1a_b = np.zeros((2,2))
C_sign1a_b[0,0] = C[5,5]
C_sign1a_b[0,1] = C[4,5]
C_sign1a_b[1,0] = C_sign1a_b[0,1]
C_sign1a_b[1,1] = C[4,4]

C_sign1a_omega_m = np.zeros((2,2))

C_sign1a_omega_m[0,0] = C[5,5]
C_sign1a_omega_m[0,1] = C[2,5]
C_sign1a_omega_m[1,0] = C_sign1a_b[0,1]
C_sign1a_omega_m[1,1] = C[2,2]



C_b_omega_m = np.zeros((2,2))

C_b_omega_m[0,0] = C[4,4]
C_b_omega_m[0,1] = C[4,2]
C_b_omega_m[1,0] = C_b_omega_m[0,1]
C_b_omega_m[1,1] = C[2,2]

# change this, centre, range, and labels as needed

w,v=np.linalg.eigh(C_w_omega_m)

angle=180*np.arctan2(v[1,0],v[0,0])/np.pi

#

a_1s=np.sqrt(2.3*(w[0])) #68% #it seems like a is whatever the 2nd element of the array is, so sigy**2
b_1s=np.sqrt(2.3*w[1])
a_2s=np.sqrt(6.17*(w[0])) #95%
b_2s=np.sqrt(6.17*w[1])
a_3s = np.sqrt(11.8*w[0])
b_3s = np.sqrt(11.8*w[1])


#centre=np.array([params[i1].val,params[i2].val])


#make cuts 
w = -0.978193
centre = np.array([w,omega_m])

e_1s=Ellipse(xy=centre,width=2*a_1s,height=2*b_1s,angle=angle,
                    facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='aqua', label = '$68$% $confidence$')
#                     

e_2s=Ellipse(xy=centre,width=2*a_2s,height=2*b_2s,angle=angle,
                    facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='blue',  label = '$95$% $confidence$')
#                     facecolor=fc[i],linewidth=lw[i],linestyle=ls[i],edgecolor=lc[i])

e_3s=Ellipse(xy=centre,width=2*a_3s,height=2*b_3s,angle=angle,
                    facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='purple',  label = '99% confidence')


ax = plt.gca()

M_omega_m_emcee = np.loadtxt('M_omega_m_emcee_simplebeams.txt', unpack=True)  

full_emcee = np.loadtxt('full_emcee_simplebeams.txt', unpack=True)  

#plt.axis([0.94, 1.06, 0.1, 0.4] ) #b, omega_m
#plt.axis([0, 0.1, 0.94, 1.06] ) #sign1a, b
#plt.axis([0, 0.1, 0.1, 0.4] ) #sign1a, omega_m
#plt.axis([-19.39, -19.29, -1.4, -0.6] ) #M, w

plt.axis([-1.4,-0.6,0.1, 0.4]) #w, omega_m




plt.plot()

plt.xlabel('$w$')
plt.ylabel("$\Omega_{m}$")

ax.add_patch(e_1s)
ax.add_patch(e_2s)
#ax.add_patch(e_3s)

# order is omega_m, h0, w, M, b, sign1a 
#ax.scatter(full_emcee[3][8000:-1],full_emcee[2][8000:-1], c = 'purple', s = 0.2, alpha = 0.1)
ax.legend()
#plt.savefig('bam.pdf')

plt.show()


