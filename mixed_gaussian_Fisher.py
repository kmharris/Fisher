import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib.patches import Ellipse
import random as random
from statsmodels.stats import correlation_tools
from scipy.integrate import quad 

'''
This code generates two gaussian distributions
to test the functionality of the mixed gaussian
fisher matrix code
'''

data1a = np.random.normal(0.,0.1,40) #simple gaussian, acts as 1a population
datan1a = np.random.normal(0.3, 0.5, 40) #non-1a population, notice shifted mean and larger sigma

rand = random.sample(range(40),15)

data = np.zeros((40,2))

for i in range(40):
    if rand.count(i) == 1:
        data[i,0] = datan1a[i]
        data[i,1] = np.random.uniform(0,0.2) #assigning low probability to non-1a
    else:
        data[i,0] = data1a[i]
        data[i,1] = 1.0 #assigns high probability to 1a

data = data[np.lexsort(np.fliplr(data).T)]  #sorts data from smallest to largest 
sig1 = 0.1
sig2 = 0.5
u1 = 0
u2 = 0.3
p1 = data[:,1]    
x = data[:,0]


def f(xs,u1s,u2s,p1s,sig1s,sig2s): #function that represents the mixed gaussian, acts as BEAMS posterior

    rt = []
    for i in range(40):
        ps = Decimal(p1s[i])
        f1 = Decimal(1./(np.sqrt(2*np.pi)*(sig1s)))*Decimal(-(xs[i]-u1s)**2/(2*sig1s)).exp()
        f2 = Decimal(1./(np.sqrt(2*np.pi)*(sig2s)))*Decimal(-(xs[i]-u2s)**2/(2*sig2s)).exp()
        rt.append(float(ps*f1 + (Decimal(1.)-ps)*f2))
    
    return np.array(rt)

def log_f(xs,u1s,u2s,p1s,sig1s,sig2s): #log of that function
    rt = []
    for i in range(40):
        ps = Decimal(p1[i])
        f1 = Decimal(1./(np.sqrt(2*np.pi)*(sig1s)))*Decimal(-(xs[i]-u1s)**2/(2*sig1s)).exp()
        f2 = Decimal(1./(np.sqrt(2*np.pi)*(sig2s)))*Decimal(-(xs[i]-u2s)**2/(2*sig2s)).exp()
        rt.append(float(Decimal.ln(ps) + Decimal.ln(f1) + Decimal.ln(Decimal(1.) + (Decimal(1.)-ps)*f2/(ps*f1))))
    
    return np.array(rt)

#these three functions represent the elements of a 2x2 fisher matrix. The derivatives are done within the function


def I_00(x): #u2
    h = 4e-1 #arbitrarily small step for analytical derivative
    return f(x,u1,u2,p1,sig1,sig2)*((log_f(x,u1 ,u2 + h,p1,sig1,sig2) - log_f(x,u1,u2 - h,p1,sig1,sig2))/(2.*h))**2

def I_01(x):
    h = 4e-1
    return f(x,u1,u2,p1,sig1,sig2)*((log_f(x,u1,u2,p1,sig1,sig2+ h) - log_f(x,u1,u2 ,p1,sig1,sig2 - h))/(2.*h))*((log_f(x,u1,u2 + h,p1,sig1,sig2) - log_f(x,u1 ,u2 - h,p1,sig1,sig2 ))/(2.*h))

def I_11(x): #sig2
    h = 4e-1
    return f(x,u1,u2,p1,sig1,sig2)*((log_f(x,u1 ,u2 ,p1 ,sig1 ,sig2 + h) - log_f(x,u1,u2 ,p1 ,sig1 ,sig2 - h))/(2.*h))**2



C_sig2_u2 = np.zeros((2,2))
C_sig2_u2[0,0] = np.trapz(I_11(x), x) #this integrates the whole function, which is required for the expectation value
C_sig2_u2[0,1] = np.trapz(I_01(x), x)
C_sig2_u2[1,0] = C_sig2_u2[0,1]
C_sig2_u2[1,1] = np.trapz(I_00(x), x)



#C_sig2_u2 = correlation_tools.cov_nearest(C_sig2_u2)
C = np.linalg.inv(C_sig2_u2) # this inverts the matrix to get the covariance matrix

#this part plots the ellipses for different confidence intervals

w,v=np.linalg.eigh(C)

angle=180*np.arctan2(v[1,0],v[0,0])/np.pi

#

a_1s=np.sqrt(2.3*w[0]) #68% confidence
b_1s=np.sqrt(2.3*w[1])
a_2s=np.sqrt(6.17*w[0]) #95% confidence 
b_2s=np.sqrt(6.17*w[1])


centre = np.array([sig2,u2])

e_1s=Ellipse(xy=centre,width=2*a_1s,height=2*b_1s,angle=angle,
                    facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='aqua', label = '$68$% $confidence$')
#                     

e_2s=Ellipse(xy=centre,width=2*a_2s,height=2*b_2s,angle=angle,
                    facecolor='None',linewidth=1.0,linestyle='solid',edgecolor='blue',  label = '$95$% $confidence$')
#                     facecolor=fc[i],linewidth=lw[i],linestyle=ls[i],edgecolor=lc[i])


ax = plt.gca()


plt.axis([-2,2,-2,2]) #this needs to be adjusted to whatever scale the ellipses are on
plt.plot()

plt.xlabel('$\sigma_{2}^{2}$')
plt.ylabel("$\mu_{2}$")
#ax.scatter(gauss[1][8000:-1], gauss[0][8000:-1],  c = 'purple', s = 0.2, alpha = 0.1) #this plots the MCMC point from CosmoSIS if you have them to compare to
ax.add_patch(e_1s)
ax.add_patch(e_2s)


ax.legend()

plt.show()

