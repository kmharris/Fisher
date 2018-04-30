import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib.patches import Ellipse

from astropy.cosmology.funcs1 import distmod
from statsmodels.stats import correlation_tools


#I'll need to put in cosmology for astropy because I took it all out of the source code

from astropy.cosmology import Flatw0waCDM

#cosmological parameters best estimates from Cosmosis or accepted values

m = -19.356

omega_m  =  0.272

h0  =  70.0

w =  -9.6e-01

omega_b  =  0.04

b = 1.012

sign1a = 0.0374

#loading in the data


datamat = np.loadtxt('test_out.txt', unpack=True)    
zhel = datamat[2] # redshift 
zcmb = datamat[1] # redshift
mb = datamat[4] 
dmb = datamat[5]
p1a = datamat[-1]
 

def f(mb, dmb, zcmb, p1a, M, h0, omega_m, w, b, sign1a): #creates the BEAMS posterior
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

def log_f(mb, dmb, zcmb, p1a, M, h0, omega_m, w, b, sign1a): #log BEAMS posterior
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
   
    
#Order of variables is M, omega_m, w, b, sign1a

#first row 
    
#these functions make up the elements of the fisher matrix before integration, and include numerical derivatives
#the h values are the steps taken in the numerical derivative, and are 0.1*cosmosis_error
    
def I_00(mb): #dm^2
    h = 1.3e-3
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m, w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M + h, h0, omega_m,   w, b, sign1a  ) - log_f(mb, dmb, zcmb, p1a, M - h , h0, omega_m,   w, b, sign1a))/(2.*h))**2

def I_01(mb): #dm*domega_m
    hm = 1.3e-3
    hom = 5.0e-3
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,  w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M +hm , h0, omega_m,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M - hm , h0, omega_m ,   w, b, sign1a))/(2.*hm))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m + hom ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m - hom , w, b , sign1a ))/(2.*hom))
 

def I_02(mb): #dm*dw
    hm = 1.3e-3
    hw = 1.2e-2
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,  w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M +hm , h0, omega_m,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M - hm , h0, omega_m ,   w, b, sign1a))/(2.*hm))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w + hw, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m  ,  w - hw, b , sign1a ))/(2.*hw))
 
def I_03(mb): #dm*db
    hm = 1.3e-3
    hb = 1.3e-3
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m, w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M +hm , h0, omega_m,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M - hm , h0, omega_m ,   w, b, sign1a))/(2.*hm))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w , b + hb, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m  ,  w , b -hb, sign1a ))/(2.*hb))
 
def I_04(mb): #dm*dsign1a
    hm = 1.3e-3
    hs = 9.4e-4
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,  w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M +hm , h0, omega_m,   w, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M - hm , h0, omega_m ,   w, b, sign1a))/(2.*hm))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w , b , sign1a + hs) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m  ,  w , b, sign1a - hs ))/(2.*hs))
 
# second row


def I_10(mb): #dM*domega_m
    return I_01(mb)

def I_11(mb): #domega_m^2
    h = 5.0e-3
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m + h,   w, b, sign1a  ) - log_f(mb, dmb, zcmb, p1a, M  , h0 , omega_m - h,   w, b, sign1a))/(2.*h))**2

def I_12(mb): #domega_m*dw
    hom = 5.0e-3
    hw = 1.2e-2
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w + hw, b , sign1a ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w - hw, b, sign1a))/(2.*hw))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m + hom ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m - hom ,   w, b , sign1a ))/(2.*hom))
  
def I_13(mb): #domega_m*db
    hom = 5.0e-3
    hb = 1.3e-3
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w , b + hb , sign1a ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w , b - hb, sign1a))/(2.*hb))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m + hom ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m - hom ,   w, b , sign1a ))/(2.*hom))
  
def I_14(mb): #domega_m*dsign1a
    hom = 5.0e-3
    hs = 9.4e-4
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w , b  , sign1a +hs) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w , b , sign1a - hs))/(2.*hs))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m + hom ,   w, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m - hom ,   w, b , sign1a ))/(2.*hom))
  
#third row


def I_20(mb): #dM*dw
    return I_02(mb)

def I_21(mb): #dw*domega_m
    return I_12(mb)

def I_22(mb): #dw^2
    h = 1.2e-2
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w + h, b, sign1a  ) - log_f(mb, dmb, zcmb, p1a, M  , h0 , omega_m,   w - h, b, sign1a))/(2.*h))**2

def I_23(mb): #dw*db
    hw = 1.2e-2
    hb = 1.3e-3
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b +hb, sign1a ) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w, b - hb, sign1a))/(2.*hb))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w + hw, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m  ,  w - hw, b , sign1a ))/(2.*hw))

def I_34(mb): #dw*dsign1a
    hw = 1.2e-2
    hs = 9.4e-4
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a +hs) - log_f(mb, dmb, zcmb, p1a, M , h0, omega_m ,   w, b, sign1a - hs))/(2.*hs))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w + hw, b, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m  ,  w - hw, b , sign1a ))/(2.*hw))

#fourth row 
    
def I_30(mb): #dM*db
    return I_03(mb)

def I_31(mb): #domega_m*db
    return I_13(mb)

def I_32(mb): #dw*db
    return I_23(mb)

def I_33(mb): #db^2
    h = 1.3e-3
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w , b + h, sign1a  ) - log_f(mb, dmb, zcmb, p1a, M  , h0 , omega_m,   w , b - h, sign1a))/(2.*h))**2

def I_34(mb): #dsign1a*db
    hb = 1.3e-3
    hs = 9.4e-4
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,   w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w, b , sign1a +hs) - log_f(mb, dmb, zcmb, p1a, M  , h0, omega_m ,   w, b, sign1a - hs))/(2.*hs))*((log_f(mb, dmb, zcmb, p1a, M, h0, omega_m  ,   w , b + hb, sign1a) - log_f(mb, dmb, zcmb, p1a, M, h0 , omega_m , w , b -hb, sign1a ))/(2.*hb))
 
#fifth row 
 
def I_40(mb): #dsign1a*dM
    return I_04(mb)

def I_41(mb): #dsign1a*domega_m
    return I_14(mb)

def I_42(mb): #dsign1a*dw
    return I_24(mb)

def I_43(mb): #dsign1a*db
    return I_34(mb)

def I_44(mb): #dsign1a^2
    h = 9.4e-4
    return f(mb, dmb, zcmb, p1a, M, h0, omega_m,  w, b, sign1a)*((log_f(mb, dmb, zcmb, p1a, M , h0, omega_m,   w , b, sign1a +h ) - log_f(mb, dmb, zcmb, p1a, M  , h0 , omega_m,   w , b, sign1a - h ))/(2.*h))**2


F = np.zeros((5,5)) #empty matrix that will become the fisher matrix

#this loop integrates the functions for the expectation value

for i in range(5):
    for j in range(5):
        F[i,j] = np.trapz(eval('I_'+'{:1.0f}'.format(i) + '{:1.0f}'.format(j))(mb), mb) #uses trapz to integrate over data vector mb
    



C = np.linalg.inv(F) #inverting the full matrix to get the covariance matrix

C = correlation_tools.cov_nearest(C, threshold = 1e-14) #this makes sure the inverted matrix is positive definite
#C = C + 1.37765529e-04

#these are the 2x2 matrices for plotting confidence ellipses

C_sign1a_b = np.zeros((2,2))

C_sign1a_b[0,0] = C[4,4] 
C_sign1a_b[0,1] = C[3,4]
C_sign1a_b[1,0] = C_sign1a_b[0,1]
C_sign1a_b[1,1] = C[3,3]


C_m_b = np.zeros((2,2))
C_m_b[0,0] = C[0,0] 
C_m_b[0,1] = C[0,3]
C_m_b[1,0] = C_sign1a_b[0,1]
C_m_b[1,1] = C[3,3]

C_w_omega_m = np.zeros((2,2))

C_w_omega_m[0,0] = C[2,2]
C_w_omega_m[0,1] = C[1,2]
C_w_omega_m[1,0] = C_w_omega_m[0,1]
C_w_omega_m[1,1] = C[1,1]

C_m_w = np.zeros((2,2))
C_m_w[0,0] = C[0,0]
C_m_w[0,1] = C[0,2]
C_m_w[1,0] = C_m_w[0,1]
C_m_w[1,1] = C[2,2]

C_sign1a_w = np.zeros((2,2))

C_sign1a_w[0,0] = C[4,4] 
C_sign1a_w[0,1] = C[2,4]
C_sign1a_w[1,0] = C_sign1a_w[0,1]
C_sign1a_w[1,1] = C[2,2]


# change the eignvectors, centre, range, and labels as needed

#plotting the ellipses!

w,v=np.linalg.eigh(C_sign1a_b)

angle=180*np.arctan2(v[1,0],v[0,0])/np.pi

#


a_1s=np.sqrt(2.3*(w[0])) #68% 
b_1s=np.sqrt(2.3*w[1])
a_2s=np.sqrt(6.17*(w[0])) #95%
b_2s=np.sqrt(6.17*w[1])
a_3s = np.sqrt(11.8*(w[0])) #99%
b_3s = np.sqrt(11.8*w[1])



#make cuts 
w = -0.97

centre = np.array([sign1a, b])

e_1s=Ellipse(xy=centre,width=2*a_1s,height=2*b_1s,angle=angle,
                    facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='red', label = '68% confidence')
#                     

e_2s=Ellipse(xy=centre,width=2*a_2s,height=2*b_2s,angle=angle,
                    facecolor='None',linewidth=1.0,linestyle='dashed',edgecolor='blue',  label = '95% confidence')
#                     facecolor=fc[i],linewidth=lw[i],linestyle=ls[i],edgecolor=lc[i])

e_3s=Ellipse(xy=centre,width=2*a_3s,height=2*b_3s,angle=angle,
                    facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='purple',  label = '99% confidence')


ax = plt.gca()


no_h0_emcee = np.loadtxt('no_h0_emcee_simplebeams.txt', unpack=True) 

#different axes for different ellipses
plt.axis([0, 0.1, 0.94, 1.06] ) #sign1a, b
#plt.axis([-19.39, -19.30, 0.94, 1.06] ) #m, b
#plt.axis([-1.3, -0.6, 0.1, 0.4] ) #w, omega_m
#plt.axis([-19.39, -19.29, -1.3, -0.6] ) #m, w
#plt.axis([0, 0.1, -1.3, -0.6] ) #sign1a, w

plt.plot()

plt.xlabel('m')
plt.ylabel("w")

ax.add_patch(e_1s)
ax.add_patch(e_2s)
ax.add_patch(e_3s)

#order in the file is omega_m, w, m, b, sign1a
ax.scatter(no_h0_emcee[4][8000:-1],no_h0_emcee[3][8000:-1], c = 'g', s = 0.2, alpha = 0.1)
#scatter plots MCMC poitns
ax.legend()

plt.show()

