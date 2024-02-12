import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import special
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.ticker import FormatStrFormatter
import math 


pi=np.arccos(-1)
eps_=10**-20

log_= lambda x: np.log(x)
e_  = lambda x: np.exp(x)
sqt_= lambda x: np.sqrt(abs(x)+eps_)
erf_= lambda x: special.erf(x)
derf_=lambda x: (2/sqt_(pi))*e_(-x*x)
tanh_= lambda x: np.tanh(x)



erf_= lambda x: special.erf(x)
derf_=lambda x: (2/sqt_(pi))*e_(-x*x)

Var1=lambda x,y,z: (x-y)/sqt_(2*z)
dx_Var1=lambda x,y,z: 1/sqt_(2*z)
dy_Var1=lambda x,y,z: -1/sqt_(2*z)
dz_Var1=lambda x,y,z: (-1/2)*(x-y)/(sqt_(2*z)*z)

Var2=lambda x,y,z: (x+y)/sqt_(2*z)
dx_Var2=lambda x,y,z: 1/sqt_(2*z)
dy_Var2=lambda x,y,z: 1/sqt_(2*z)
dz_Var2=lambda x,y,z: (-1/2)*(x+y)/(sqt_(2*z)*z)

H=    lambda x,y,z: (1/2)*erf_(Var1(x,y,z))+(1/2)*erf_(Var2(x,y,z)) if np.sign(abs(Var1(x,y,z))-5.5)+np.sign(abs(Var2(x,y,z))-5.5)<1.5  else  (      np.sign(Var1(x,y,z))/2+np.sign(Var2(x,y,z))/2 -e_(-Var1(x,y,z)**2)/(2*sqt_(pi)*(Var1(x,y,z)))-e_(-Var2(x,y,z)**2)/(2*sqt_(pi)*(Var2(x,y,z)))     )
G=    lambda x,y,z: (1/(sqt_(2*pi)))*( Var1(x,y,z)*e_(-Var1(x,y,z)**2) + Var2(x,y,z)*e_(-Var2(x,y,z)**2) )
###### Computing κ_SAT ######
#############################
def κ_SAT_finding(α):
    def α_critical(κ_):
        κ=abs(κ_)
        integral=integrate.quad(lambda z: (np.exp(-z*z/2)/np.sqrt(2*pi)), -κ, κ)[0]
        out= α+np.log(2)/np.log(integral)
        
        return  abs(out)
    
    κo=(np.sqrt(2*pi)/2)*np.exp(-np.log(2)/α)
    κ_SAT=abs(optimize.fmin(α_critical,κo))
    κ_SAT=abs(optimize.fsolve(α_critical,κ_SAT))
    return κ_SAT



####### Test of int ##########
##############################
def test_ex(q,m,κ_old,κ_new,α,cut):
    q_=q+(10**-3)*(1-q)
    dq=(10**-3)*(1-q)
    
    var1= lambda B: (κ_old+B)/np.sqrt(2*(q-m*m))
    var2= lambda B: (κ_old-B)/np.sqrt(2*(q-m*m))
    var1_= lambda B: (κ_old+B)/np.sqrt(2*(q_-m*m))
    var2_= lambda B: (κ_old-B)/np.sqrt(2*(q_-m*m))
    
    var11= lambda B: (κ_new+B)/np.sqrt(2*(1-q))
    var22= lambda B: (κ_new-B)/np.sqrt(2*(1-q))
    var11_= lambda B: (κ_new+B)/np.sqrt(2*(1-q_))
    var22_= lambda B: (κ_new-B)/np.sqrt(2*(1-q_))
    
    func1= lambda B: (np.exp(-B*B/2)/np.sqrt(2*pi))   *   (1/2)*( special.erf(var1(B))  +  special.erf(var2(B)) )   *\
                                                   np.log((1/2)*( special.erf(var11(B)) +  special.erf(var22(B)) ))
    

    func2= lambda B: (np.exp(-B*B/2)/np.sqrt(2*pi))   *   (1/2)*( special.erf(var1_(B)) +  special.erf(var2_(B)) )   *\
                                                   np.log((1/2)*( special.erf(var11(B)) +  special.erf(var22(B)) ))


    func3= lambda B: (np.exp(-B*B/2)/np.sqrt(2*pi))   *   (1/2)*( special.erf(var1(B))   +  special.erf(var2(B)) )   *\
                                                   np.log((1/2)*( special.erf(var11_(B)) +  special.erf(var22_(B)) ))
                                                   
    func_test= lambda B: (func2(B)-func1(B))/dq
    func_test_=lambda B: (func3(B)-func1(B))/dq

    print(integrate.quad(func_test,-cut,cut)[0],integrate.quad(func_test_,-cut,cut)[0],integrate.quad(func_test,-cut,cut)[0]+integrate.quad(func_test_,-cut,cut)[0])


########### SE ##############
#############################

def State_evolution(x):
    Nt=int(x[0])
    q_in=x[1]
    m_in=x[2]
    κ_old=x[3]
    κ_new=x[4]
    α=x[5]
    
    P = lambda w: np.exp(-w*w/2)
    N_w=integrate.quad(P, -κ_old, κ_old)[0]
    
    m=np.zeros(Nt+1)
    q=np.zeros(Nt+1)
    m_hat=np.zeros(Nt+1)
    q_hat=np.zeros(Nt+1)
    
    xo=1
    η=0.01
    conv=0
    
    cut=8
    dcut=0.001
    
    ####### Derivatives channel out #######
    #######################################
    def dq_Ф_out(Q,q,m,κ,α):
        eps=10**(-40)
        Gab=(q-m**2)
        
        def Z_out(t,w,q,m):
            Z=(1/2)*special.erf(  (κ-t*np.sqrt(abs((q-m**2))-m*w))/np.sqrt(2*abs(Q-q))  ) + (1/2)*special.erf(  (κ+t*np.sqrt(abs((q-m**2))+m*w))/np.sqrt(2*abs(Q-q))) 
            if Z>eps:
                return Z
            else: 
                return 0
        
        def potential(t,w):
            var1=(κ-t*np.sqrt(Gab)-w*m)/np.sqrt(2*(Q-q+eps))
            var11=-t/np.sqrt(2*(Q-q+eps)*Gab)
            var2=(κ+t*np.sqrt(Gab)+w*m)/np.sqrt(2*(Q-q+eps))
            var22=t/np.sqrt(2*(Q-q+eps)*Gab)
        
            Z=(special.erf(var1)+special.erf(var2)+eps)
            out1=(1/((Q-q+eps)*np.sqrt(pi)))  *   (var1*np.exp(-var1**2)+var2*np.exp(-var2**2)) / Z
            out1+=(1/(np.sqrt(pi)))  *   (var11*np.exp(-var1**2)+var22*np.exp(-var2**2)) / Z
            out2=out1*(np.exp(-t*t/2)/np.sqrt(2*pi))*(np.exp(-w*w/2)/N_w)

            if abs(Z)>eps:
                return out2
            else:
                return 0
            
        def potential2(t,w):
            dq=0.001*q
            Z=Z_out(t,w,q,m)
            if abs(Z)>eps:
                return ((Z_out(t,w,q+dq,m)-Z)/(Z*dq))  *(np.exp(-t*t/2)/np.sqrt(2*pi))*(np.exp(-w*w/2)/N_w)
            else:
                return 0
            
        return α*integrate.nquad(potential, [[-20, 20],[-κ_old, κ_old]])[0]    
    
    def dq_Ф_out_simple(Q,q,m,κ,α,cut):
        eps=10**(-40)
        
        def dZ_out(x):
            var1=(κ-x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            var2=(κ+x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            Z=0.5*(special.erf(var1)+special.erf(var2)+eps)
            
            out=(-np.exp(-var1**2)+np.exp(-var2**2))/(np.sqrt(2*pi*abs(Q-q))*Z)

            if Z>eps:
                return out
            else: 
                return 0
            
        def f(x):
            var1=(κ-x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            var2=(κ+x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            Z=0.5*(special.erf(var1)+special.erf(var2)+eps)
            
            out=κ*(np.exp(-var1**2)+np.exp(-var2**2))/(2*abs(Q-q)*np.sqrt(2*pi*abs(Q-q))*Z)

            if Z>eps:
                return out
            else: 
                return 0
             
        def potential_simple(x):
            var1=(+m*x+np.sqrt(q)*κ_old)/np.sqrt(2*(q-m**2))
            var2=(-m*x+np.sqrt(q)*κ_old)/np.sqrt(2*(q-m**2))
            Z=0.5*(special.erf(var1)+special.erf(var2)+eps)
            
            out1=m/(2*np.sqrt(q*(q-m**2)))*dZ_out(x)*(-np.exp(-var1**2)+np.exp(-var2**2))/np.sqrt(2*pi) 
            out2= (     (  x/(2*np.sqrt(q)) +np.sqrt(q)*x/(2*(Q-q)) )*dZ_out(x) + f(x)       ) * Z
            
            return (out1+out2)*(np.exp(-x*x/2)/N_w)
                

            
        return α*integrate.quad(potential_simple,-cut,cut)[0]
      
    def dm_Ф_out(Q,q,m,κ,α):
        eps=10**(-40)
        Gab=(q-m**2)
        dGab=-2*m
           
        def Z_out(t,w,q,m):
            Z=(1/2)*special.erf(  (κ-t*np.sqrt(abs((q-m**2))-m*w))/np.sqrt(2*abs(Q-q))  ) + (1/2)*special.erf(  (κ+t*np.sqrt(abs((q-m**2))+m*w))/np.sqrt(2*abs(Q-q))) 
            if Z>eps:
                return Z
            else: 
                return Z
        
        def potential(t,w):
            var1=(κ-t*np.sqrt(Gab)-w*m)/np.sqrt(2*(Q-q+eps))
            var11=-t*dGab/np.sqrt(2*(Q-q+eps)*Gab)
            var111=-w/np.sqrt(2*(Q-q+eps))
            var2=(κ+t*np.sqrt(Gab)+w*m)/np.sqrt(2*(Q-q+eps))
            var22=t*dGab/np.sqrt(2*(Q-q+eps)*Gab)
            var222=w/np.sqrt(2*(Q-q+eps))
         
            Z=(special.erf(var1)+special.erf(var2)+eps)
            out1= (1/(np.sqrt(pi)))  *   (var11*np.exp(-var1**2)+var22*np.exp(-var2**2))/Z  
            out1+=(1/(np.sqrt(pi)))  *   (var111*np.exp(-var1**2)+var222*np.exp(-var2**2))/(Z/2) 
            out2= out1*(np.exp(-t*t/2)/np.sqrt(2*pi))*(np.exp(-w*w/2)/N_w)
            
            if abs(Z)>eps:
                return out2        
            else: 
                return 0
        
        def potential2(t,w):
            dm=0.001*m
            Z=Z_out(t,w,q,m)
            if abs(Z)>eps:
                return ((Z_out(t,w,q,m+dm)-Z)/(Z*dm)) *(np.exp(-t*t/2)/np.sqrt(2*pi))*(np.exp(-w*w/2)/N_w)
            else:
                return 0
        
        return α*integrate.nquad(potential, [[-20, 20],[-κ_old, κ_old]])[0]  

    def dm_Ф_out_simple(Q,q,m,κ,α,cut):
        eps=10**(-40)
           
        def dZ_out(x):
            var1=(κ-x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            var2=(κ+x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            Z=0.5*(special.erf(var1)+special.erf(var2)+eps)
            
            out=(-np.exp(-var1**2)+np.exp(-var2**2))/(np.sqrt(2*pi*abs(Q-q))*Z)

            if Z>eps:
                return out
            else: 
                return 0
  
        def potential_simple(x):
            var1=(+m*x+np.sqrt(q)*κ_old)/np.sqrt(2*(q-m**2))
            var2=(-m*x+np.sqrt(q)*κ_old)/np.sqrt(2*(q-m**2))
            
            out=-np.sqrt(q/(q-m**2))*dZ_out(x)*(-np.exp(-var1**2)+np.exp(-var2**2))/np.sqrt(2*pi) 
            
            return (out)*(np.exp(-x*x/2)/N_w)
            
    
        
        return α*integrate.quad(potential_simple, -cut, cut)[0]
      

    def potential_cut(x,Q,q,m,κ,α):
        eps=10**(-40)
        
        def dZ_out(x):
            var1=(κ-x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            var2=(κ+x*np.sqrt(q))/np.sqrt(2*(Q-q+eps))
            Z=0.5*(special.erf(var1)+special.erf(var2)+eps)
            
            out=(-np.exp(-var1**2)+np.exp(-var2**2))/(np.sqrt(2*pi*abs(Q-q))*Z)

            if Z>eps:
                return out
            else: 
                return 0
            
            
        var1=(+m*x+np.sqrt(q)*κ_old)/np.sqrt(2*(q-m**2))
        var2=(-m*x+np.sqrt(q)*κ_old)/np.sqrt(2*(q-m**2))
            
        out=-np.sqrt(q/(q-m**2))*dZ_out(x)*(-np.exp(-var1**2)+np.exp(-var2**2))/np.sqrt(2*pi) 
            
        return abs(out*(np.exp(-x*x/2)/N_w))
    
    ######## Derivatives channel in #######
    #######################################    
    def dqhat_Ф_in(q_hat,m_hat):
        eps=10**(-10)
        potential_1= lambda t: (1/(2*np.sqrt(q_hat+eps)))*t*np.tanh(m_hat*xo+np.sqrt(q_hat+eps)*t)
        potential_2= lambda t: potential_1(t)*(np.exp(-t*t/2)/np.sqrt(2*pi))
        
        return integrate.quad(potential_2, -np.Infinity, np.Infinity)[0]
    
    def dmhat_Ф_in(q_hat,m_hat):
        eps=10**(-10)
        potential_1= lambda t: xo*np.tanh(m_hat*xo+np.sqrt(q_hat+eps)*t)
        potential_2= lambda t: potential_1(t)*(np.exp(-t*t/2)/np.sqrt(2*pi))
        
        return integrate.quad(potential_2, -np.Infinity, np.Infinity)[0]
    
    
    
    
    
    ####### Running the SE #######
    ##############################     
    q[0]=q_in
    m[0]=m_in
    
    for t in range(Nt):
        
        ##### defining the cut in the integrals #####
        test1=0
        test2=0
        threshold=10**(-10)
        
        for j in range(4000):
            
            
            if abs(potential_cut(cut,1,q[t],m[t],κ_new,α))<threshold:
                test2=test1
                test1=+1
                cut+=-dcut 
                 
            if abs(potential_cut(cut,1,q[t],m[t],κ_new,α))>threshold:
                test2=test1
                test1=-1
                cut+=+dcut 
    
            if test1*test2<0:
                break
                
            if cut<0.01:
                cut=0.01
                break
                
            
            
        ##### hat variables #####
        q_hat[t]=-2*dq_Ф_out_simple(1,q[t],m[t],κ_new,α,cut) 
        m_hat[t]=dm_Ф_out_simple(1,q[t],m[t],κ_new,α,cut)
        

        ##### usuelle variables #####        
        if q_hat[t]==0.0 or m_hat[t]==0.0:
            print("!!! pb in integrals")
            q_new=max(0.9999*q[t],0)
            m_new=max(0.9999*m[t],0)
        else:
            q_new=(1-2*dqhat_Ф_in(q_hat[t],m_hat[t]))
            m_new=(dmhat_Ф_in(q_hat[t],m_hat[t]))
        q[t+1]=η*q_new+(1-η)*q[t]
        m[t+1]=η*m_new+(1-η)*m[t]
        
        
        
        
        print("t,α,κ_old,κ_new,η,cut:",t,α,round(κ_old,5),round(κ_new,5),η,round(cut,5))
        print("q,m,dq:",1,round(q[t],9),round(m[t],9),(abs(q[t]-q[t+1])/(1-q[t])))
        print("q_hat,m_hat,conv:",round(q_hat[t],9),round(m_hat[t],9),conv)
        print("test !!")
        if conv>30:
            test_ex(q[t],m[t],κ_old,κ_new,α,cut)
        print("")
        
        
        ### Criteria for convergence of SE ####
        if abs(q[t]-q[t+1])/(1-q[t])<1*10**(-4) and t>30:
            η=0.01
            conv+=1
            t_mem=t
            
        if abs(q[t]-q[t+1])/(1-q[t])<10**(-4) and conv>40:
            t_mem=t
            print("Converging normal")
            break
        
        if t==Nt-1:
            t_mem=t
            
        #### Criteria if SE explodes ####
        if (q_hat[t])>500 and t>10:
            t_mem=t
            print("Converging q_hat")
            break

        #### Criteria if SE has a problem ####
        if math.isnan(q[t+1])==True or math.isnan(m[t+1])==True:
            q[t+1]=max(0.9999*q[0],m[0]**2)
            m[t+1]=max(0.9999*m[0],0)
            
    x[1]=q[t_mem]
    x[2]=m[t_mem]
    return[q[t_mem],m[t_mem],q_hat[t_mem],m_hat[t_mem],cut]    


####### Free Energy #########
#############################



# Planted
def Free_energy(q,m,q_hat,m_hat,κ_old,κ_new,α):
    
    P = lambda w: np.exp(-w*w/2)
    N_w=integrate.quad(P, -κ_old, κ_old)[0]
    xo=1
    
    def Ф_out(Q,q,m,κ,α):
        eps=10**(-25)
        Gab=(q-m**2)
        Q=1
        
        Z= lambda t,w: (1/2)*special.erf(  (κ-t*np.sqrt(Gab)-m*w)/np.sqrt(2*(Q-q+eps))  ) + (1/2)*special.erf(  (κ+t*np.sqrt(Gab)+m*w)/np.sqrt(2*(Q-q+eps))+eps)  
        potential_1= lambda t,w: np.log((1/2)*special.erf(  (κ-t*np.sqrt(Gab)-m*w)/np.sqrt(2*(Q-q+eps))  ) + (1/2)*special.erf(  (κ+t*np.sqrt(Gab)+m*w)/np.sqrt(2*(Q-q+eps))  ) +eps)
        potential_2= lambda t,w: potential_1(t,w)*(np.exp(-t*t/2)/np.sqrt(2*pi))*(np.exp(-w*w/2)/N_w) if abs(Z(t,w))>eps else 0
        
        return integrate.nquad(potential_2, [[-20, 20],[-κ_old, κ_old]])[0]
    G_out=Ф_out(1,q,m,κ_new,α)
    
    def Ф_in(q_hat,m_hat):
        eps=10**(-25)
        potential_1= lambda t: np.log(2*np.cosh(m_hat*xo+np.sqrt(q_hat+eps)*t) )
        potential_2= lambda t: potential_1(t)*(np.exp(-t*t/2)/np.sqrt(2*pi))
        
        return integrate.quad(potential_2,-20,20)[0]
    G_in=Ф_in(q_hat,m_hat)
    
    
    return m*m_hat+0.5*(1-q)*q_hat-α*G_out-G_in

def Free_energy_extr(x):
    q=x[0]
    m=x[1]
    q_hat=x[2]
    m_hat=x[3]
    
    
    κ_old=x[4]
    κ_new=x[5]
    α=x[6]
    xo=+1
    
    P = lambda w: np.exp(-w*w/2)
    N_w=integrate.quad(P, -κ_old, κ_old)[0]
    
    
    def dq_Ф_out(Q,q,m,κ,α):
        eps=10**(-25)
        Gab=(q-m**2)
        
        def Z_out(t,w,q,m):
            eps=10**(-25)
            Z=(1/2)*special.erf(  (κ-t*np.sqrt(abs((q-m**2))-m*w))/np.sqrt(2*abs(Q-q))  ) + (1/2)*special.erf(  (κ+t*np.sqrt(abs((q-m**2))+m*w))/np.sqrt(2*abs(Q-q))) 
            if Z>eps:
                return Z
            else: 
                return 0
        
        def potential(t,w):
            var1=(κ-t*np.sqrt(Gab)-w*m)/np.sqrt(2*(Q-q+eps))
            var11=-t/np.sqrt(2*(Q-q+eps)*Gab)
            var2=(κ+t*np.sqrt(Gab)+w*m)/np.sqrt(2*(Q-q+eps))
            var22=t/np.sqrt(2*(Q-q+eps)*Gab)
        
            Z=(special.erf(var1)+special.erf(var2)+eps)
            out1=(1/((Q-q+eps)*np.sqrt(pi)))  *   (var1*np.exp(-var1**2)+var2*np.exp(-var2**2)) / Z
            out1+=(1/(np.sqrt(pi)))  *   (var11*np.exp(-var1**2)+var22*np.exp(-var2**2)) / Z
            out2=out1*(np.exp(-t*t/2)/np.sqrt(2*pi))*(np.exp(-w*w/2)/N_w)

            if abs(Z)>eps:
                return out2
            else:
                return 0
            
        def potential2(t,w):
            dq=0.001*q
            Z=Z_out(t,w,q,m)
            if abs(Z)>eps:
                return ((Z_out(t,w,q+dq,m)-Z)/(Z*dq))**(np.exp(-t*t/2)/np.sqrt(2*pi))*(np.exp(-w*w/2)/N_w)
            else:
                return 0
            
        return α*integrate.nquad(potential, [[-20, 20],[-κ_old, κ_old]])[0]    
      
    def dm_Ф_out(Q,q,m,κ,α):
        eps=10**(-15)
        Gab=(q-m**2)
        dGab=-2*m
        
        
        def Z_out(t,w,q,m):
            Z=(1/2)*special.erf(  (κ-t*np.sqrt(abs((q-m**2))-m*w))/np.sqrt(2*abs(Q-q))  ) + (1/2)*special.erf(  (κ+t*np.sqrt(abs((q-m**2))+m*w))/np.sqrt(2*abs(Q-q))) 
            if Z>eps:
                return Z
            else: 
                return Z
        
        def potential(t,w):
            var1=(κ-t*np.sqrt(Gab)-w*m)/np.sqrt(2*(Q-q+eps))
            var11=-t*dGab/np.sqrt(2*(Q-q+eps)*Gab)
            var111=-w/np.sqrt(2*(Q-q+eps))
            var2=(κ+t*np.sqrt(Gab)+w*m)/np.sqrt(2*(Q-q+eps))
            var22=t*dGab/np.sqrt(2*(Q-q+eps)*Gab)
            var222=w/np.sqrt(2*(Q-q+eps))
         
            Z=(special.erf(var1)+special.erf(var2)+eps)
            out1= (1/(np.sqrt(pi)))  *   (var11*np.exp(-var1**2)+var22*np.exp(-var2**2))/Z  
            out1+=(1/(np.sqrt(pi)))  *   (var111*np.exp(-var1**2)+var222*np.exp(-var2**2))/(Z/2) 
            out2= out1*(np.exp(-t*t/2)/np.sqrt(2*pi))*(np.exp(-w*w/2)/N_w)
            
            if abs(Z)>eps:
                return out2        
            else: 
                return 0
        
        def potential2(t,w):
            dm=0.001*m
            Z=Z_out(t,w,q,m)
            if abs(Z)>eps:
                return ((Z_out(t,w,q,m+dm)-Z)/(Z*dm))*(np.exp(-t*t/2)/np.sqrt(2*pi))*(np.exp(-w*w/2)/N_w)
            else:
                return 0
        
        return α*integrate.nquad(potential, [[-20, 20],[-κ_old, κ_old]])[0]  
    
    def dqhat_Ф_in(q_hat,m_hat):
        eps=10**(-25)
        potential_1= lambda t: (1/(2*np.sqrt(q_hat+eps)))*t*np.tanh(m_hat*xo+np.sqrt(q_hat+eps)*t)
        potential_2= lambda t: potential_1(t)*(np.exp(-t*t/2)/np.sqrt(2*pi))
        
        return integrate.quad(potential_2, -np.Infinity, np.Infinity)[0]
    
    def dmhat_Ф_in(q_hat,m_hat):
        potential_1= lambda t: xo*np.tanh(m_hat*xo+np.sqrt(q_hat)*t)
        potential_2= lambda t: potential_1(t)*(np.exp(-t*t/2)/np.sqrt(2*pi))
        
        return integrate.quad(potential_2, -np.Infinity, np.Infinity)[0]
    
    return [q-(1-2*dqhat_Ф_in(q_hat,m_hat)),m-dmhat_Ф_in(q_hat,m_hat),q_hat+2*dq_Ф_out(1,q,m,κ_new,α),m_hat-dm_Ф_out(1,q,m,κ_new,α)]
  
def Free_energy_extr_routine(x):
    q=x[0]
    m=x[1]
    q_hat=x[2]
    m_hat=x[3]
    out=Free_energy_extr([q,m,q_hat,m_hat,κ_old,κ_new,α])
    print("input",x)
    print("output",out)
    return out


def Stability(q,m,q_hat,m_hat,κ_old,κ_new,α,cut):
    f= lambda t: (e_(-t*t/2)/sqt_(2*pi))  *  (1-tanh_(sqt_(q_hat)*t+m_hat)**2)**2
    g= lambda B: (e_(-B*B/2)/sqt_(2*pi))  *  H(sqt_(q)*κ_old,m*B,q-m*m)  *   (  G(κ_new,sqt_(q)*B,1-q)**2  )/(  H(κ_new,sqt_(q)*B,1-q)**2  )
    
    a=integrate.quad(f,-20,20)[0]
    b=integrate.quad(g,-cut,cut)[0]
    
    return     (α/((1-q)**2))*a*b



# Planted simplified
def Free_energy_planted_simplified_kold_zero(m,κ_new,α):
    return -((1-m)/2)*np.log(((1-m)/2))-((1+m)/2)*np.log(((1+m)/2))+α*np.log(special.erf(κ_new/np.sqrt(2*(1-m*m))))

def Free_energy_planted_simplified_kold_zero_rescaled(dm,κ_new_rescaled,α):
    return α*(dm/4)+α*np.log(special.erf(κ_new_rescaled/np.sqrt(2*dm)))

def Free_energy_planted_simplified(m_,κ_old,κ_new,α):
    
    P = lambda w: np.exp(-w*w/2)
    N_w=integrate.quad(P, -κ_old, κ_old)[0]
    term1=-((1-m_)/2)*np.log(((1-m_)/2))-((1+m_)/2)*np.log(((1+m_)/2))
    
    func_= lambda B: (np.exp(-B*B/2)/N_w)*np.log(   0.5*special.erf((κ_new+B)/np.sqrt(2*(1-m_*m_)))    +   0.5*special.erf((κ_new-B)/np.sqrt(2*(1-m_*m_)))       )
    term2=α*integrate.quad(func_,-κ_old,κ_old)[0]
    
    return term1+term2

def Free_energy_planted_simplified_rescaled(dm,κ_old_rescaled,κ_new_rescaled,α):
    
    P = lambda w: np.exp(-w*w/2)
    N_w=integrate.quad(P, -κ_old_rescaled, κ_old_rescaled)[0]
    term1=α*dm/4
    
    func_= lambda B: (np.exp(-B*B/2)/N_w)*np.log(   0.5*special.erf((κ_new_rescaled+B)/np.sqrt(2*dm))    +   0.5*special.erf((κ_new_rescaled-B)/np.sqrt(2*dm))       )
    term2=α*integrate.quad(func_,-κ_old_rescaled,κ_old_rescaled)[0]
    
    
    return term1+term2







# 1-RSB
def Free_eneregy_1RSB_simplified(q1,q1_hat,κ,α,x,branch):
    term1=-x*q1_hat/2+x*(1-x)*q1*q1_hat/2
    term2=np.log(2)+x*x*q1_hat/2+x*np.exp(-2*(x-1)*q1_hat)
    if branch==1:
        zo=(κ-np.sqrt(2*(1-q1)*np.log(x)))/(np.sqrt(q1))
        term3=α*np.log(erf_(abs(zo)/np.sqrt(2)))
        
        
    if branch==2:
        Δ=np.sqrt(2/pi)*q1*κ*np.exp(-κ*κ/(2*(1-q1)))/(erf_(κ/np.sqrt(2*(1-q1)))*(1-q1)**(3/2))
        term3=α*x*np.log(erf_(κ/np.sqrt(2*(1-q1))))-(α/2)*np.log(1+x*Δ)
        
    return term1+term2+term3

def Deriv_Free_eneregy_1RSB_simplified(q1,q1_hat,κ,α,x,branch):
    term1=x*(1-x)*q1_hat/2
    term2=0
    if branch==1:
        zo=(κ-np.sqrt(2*(1-q1)*np.log(x)))/(np.sqrt(q1))
        dzo=sqt_(log_(x)/(2*q1*(1-q1)))+sqt_((1-q1)*log_(x)/2)/((q1)**(3/2))
        
        term3=α*(dzo/sqt_(2))*derf_(zo/sqt_(2))/erf_(zo/sqt_(2))
        
    if branch==2:
        
        Δ=  sqt_(2/pi)*q1*κ*e_(-κ*κ/(2*(1-q1)))                                                   /(erf_(κ/sqt_(2*(1-q1)))*(1-q1)**(3/2))
        dΔ= sqt_(2/pi)   *κ*e_(-κ*κ/(2*(1-q1)))                                                   /(erf_(κ/sqt_(2*(1-q1)))*(1-q1)**(3/2))\
           +sqt_(2/pi)*q1*κ*e_(-κ*κ/(2*(1-q1)))*(-κ*κ/(2*(1-q1)**(2)))                            /(erf_(κ/sqt_(2*(1-q1)))*(1-q1)**(3/2))\
           +sqt_(2/pi)*q1*κ*e_(-κ*κ/(2*(1-q1)))*(-1)*(κ/(2*(1-q1))**(3/2))*derf_(κ/sqt_(2*(1-q1)))/(erf_(κ/sqt_(2*(1-q1)))*(1-q1)**(3/2))\
           +sqt_(2/pi)*q1*κ*e_(-κ*κ/(2*(1-q1)))*(3/2)                                             /(erf_(κ/sqt_(2*(1-q1)))*(1-q1)**(5/2))
           
           
        term3=α*x*(κ/(2*(1-q1))**(3/2))*derf_(κ/sqt_(2*(1-q1)))/erf_(κ/sqt_(2*(1-q1)))\
             -(α/2)*(x*dΔ/(1+x*Δ))
        
    return term1+term2+term3

def Entropy_1RSB_simplified(q1,q1_hat,κ,α,x,branch):
    term1=-q1_hat/2+(1-2*x)*q1*q1_hat/2
    term2=x*q1_hat        +np.exp(-2*(x-1)*q1_hat)-2*x*q1_hat*np.exp(-2*(x-1)*q1_hat)
    if branch==1:
        zo=(κ-np.sqrt(2*(1-q1)*np.log(x)))/(np.sqrt(q1))
        dzo= -np.sqrt(2*(1-q1))*(1/x)     /(2*np.sqrt(np.log(x))*np.sqrt(q1))
        
        term3=α*(dzo/np.sqrt(2))*derf_(abs(zo)/np.sqrt(2))/erf_(abs(zo)/np.sqrt(2))
    
    if branch==2:
        Δ=np.sqrt(2/pi)*q1*κ*np.exp(-κ*κ/(2*(1-q1)))/(erf_(κ/np.sqrt(2*(1-q1)))*(1-q1)**(3/2))
        term3=α*np.log(special.erf(κ/np.sqrt(2*(1-q1))))-(α/2)*Δ/(1+x*Δ)
        
    return term1+term2+term3

def Saddle_1RSB(κ,α,x,guess,branch,print_):
    

    def Find_zero(dq1_0):
        q1_0=1-dq1_0*(-α/np.log(α))
        dq1_sol1=dq1_0
        dq1_sol2=dq1_0
        dq1_inter_sol1=dq1_0
        dq1_inter_sol2=dq1_0
        
        N_test=1000
        
        d_dq_sol1=-dq1_0/N_test
        d_dq_sol2=-dq1_0/N_test
        
        
        q1_hat_=-(1/(2*(x-1)))*np.log((1-q1_0)/4)
        out_sol1=Deriv_Free_eneregy_1RSB_simplified(q1_0,q1_hat_,κ,α,x,branch)
        #dq=min(10**-5,(1-q1_0)*10**(-3))
        #(Free_eneregy_1RSB_simplified(q1_0+dq,q1_hat_,κ,α,x,branch)-Free_eneregy_1RSB_simplified(q1_0,q1_hat_,κ,α,x,branch))/dq
        
        out_sol2=out_sol1
        
        Conv_sol1=0
        Conv_sol2=0
        
        for k in range(N_test+4):
            ### Sol number 1
            dq1_inter_sol1=dq1_inter_sol1+d_dq_sol1
            q1_inter_sol1=1-dq1_inter_sol1*(-α/np.log(α))
            q1_hat_inter=-(1/(2*(x-1)))*np.log((1-q1_inter_sol1)/4)

            out_inter=Deriv_Free_eneregy_1RSB_simplified(q1_inter_sol1,q1_hat_inter,κ,α,x,branch)
            #(Free_eneregy_1RSB_simplified(q1_inter_sol1+dq,q1_hat_inter,κ,α,x,branch)-Free_eneregy_1RSB_simplified(q1_inter_sol1,q1_hat_inter,κ,α,x,branch))/dq
            #dq=min(10**-5,(1-q1_inter_sol1)*10**(-3))
            
            

            if out_inter*out_sol1<0:
                d_dq_sol1=-d_dq_sol1/2
                dq1_sol1=dq1_inter_sol1
                out_sol1=out_inter
                
                Conv_sol1+=1
                
                
            ### Sol number 2
            dq1_inter_sol2=dq1_inter_sol2+d_dq_sol2
            q1_inter_sol2=1-dq1_inter_sol2*(-α/np.log(α))
            q1_hat_inter=-(1/(2*(x-1)))*np.log((1-q1_inter_sol2)/4)

            out_inter=Deriv_Free_eneregy_1RSB_simplified(q1_inter_sol2,q1_hat_inter,κ,α,x,branch)
            #(Free_eneregy_1RSB_simplified(q1_inter_sol2+dq,q1_hat_inter,κ,α,x,branch)-Free_eneregy_1RSB_simplified(q1_inter_sol2,q1_hat_inter,κ,α,x,branch))/dq
            #dq=min(10**-5,(1-q1_inter_sol2)*10**(-3))                  
                
            
            
            if Conv_sol1<5:
                dq1_sol2=dq1_inter_sol2
                out_sol2=out_inter
            else:
                if out_inter*out_sol2<0:
                    d_dq_sol2=-d_dq_sol2/2
                    dq1_sol2=dq1_inter_sol2
                    out_sol2=out_inter
                
                    Conv_sol2+=1
                
                
                
            if Conv_sol1>13 and Conv_sol2>13:
                break
                
            
            
        return dq1_sol1,dq1_sol2
            
    
    dq1_0,dq1_1=Find_zero(guess[0])

  
    if print_==1:
        Nq=2500
        x_list=np.zeros(Nq)
        y_list=np.zeros(Nq)
        z_list=np.zeros(Nq)
        
        dqmin=0
        dqmax=2*guess[0]
        
        for k in range(Nq):
            x_list[k]=dqmin+(dqmax-dqmin)*(k/Nq)
            q1_=1-x_list[k]*(-α/np.log(α))
            dq1=(10**-2)*(1-q1_)
            q1_hat_=-(1/(2*(x-1)))*np.log((1-q1_)/4)
        
            z_list[k]=Deriv_Free_eneregy_1RSB_simplified(q1_,q1_hat_,κ,α,x,branch)*0.1/x
            y_list[k]=(Free_eneregy_1RSB_simplified(q1_+dq1,q1_hat_,κ,α,x,branch)-Free_eneregy_1RSB_simplified(q1_,q1_hat_,κ,α,x,branch))*0.1/(x*dq1)
         
            
        plt.plot(x_list,z_list)
        plt.plot(x_list,y_list)
        
        plt.plot([dq1_0,dq1_0],[-2,2])
        plt.plot([dq1_1,dq1_1],[-2,2])
        
        plt.plot([guess[0],guess[0]],[-2,2],c='black',linestyle='--')

        plt.plot([dqmin,dqmax],[0,0],c='black',linestyle='--')
        plt.ylim([-3,3])
      #  plt.xlim([0,0.01])
        plt.xlabel('x='+str(x)+"  "+str(branch))
        plt.show()
        
            
    return dq1_0,dq1_1
    
def Saddle_1RSB_test(q1,κ,α,x,branch):    
        q1_hat=-(1/(2*(x-1)))*np.log((1-q1)/4)
        dq=min(10**-5,(1-q1)*10**(-3))
        out=(Free_eneregy_1RSB_simplified(q1+dq,q1_hat,κ,α,x,branch)-Free_eneregy_1RSB_simplified(q1,q1_hat,κ,α,x,branch))/dq
        return out





# 1-RSB rescaled
def Free_eneregy_1RSB_simplified_rescaled(q1,q1_hat,κ,α,x,branch):

    if branch==1:
        term1=-x*q1_hat/2+x*(1-x)*q1*q1_hat/2
        term2=np.log(2)+x*x*q1_hat/2+x*np.exp(-2*(x-1)*q1_hat)
        
        zo=(κ-np.sqrt(2*(1-q1)*np.log(x)))/(np.sqrt(q1))
        term3=α*np.log(erf_(abs(zo)/np.sqrt(2)))
        
        
    if branch==2:
        term1=x*(1-q1)/(-4/log_(α))     -x*q1_hat/2+x*q1*q1_hat/2
        term2=np.log(2)
        
        Δ=np.sqrt(2/pi)*q1*κ*np.exp(-κ*κ/(2*(1-q1)))/(erf_(κ/np.sqrt(2*(1-q1)))*(1-q1)**(3/2))
        term3=α*x*np.log(erf_(κ/np.sqrt(2*(1-q1))))-(α/2)*np.log(1+x*Δ)
        
    return term1+term2+term3


def Deriv_Free_eneregy_1RSB_simplified_rescaled(q1,q1_hat,κ,α,x,branch):

    if branch==1:
        term1=x*(1-x)*q1_hat/2
        term2=0
        zo=(κ-np.sqrt(2*(1-q1)*np.log(x)))/(np.sqrt(q1))
        dzo=sqt_(log_(x)/(2*q1*(1-q1)))+sqt_((1-q1)*log_(x)/2)/((q1)**(3/2))
        
        term3=α*(dzo/sqt_(2))*derf_(zo/sqt_(2))/erf_(zo/sqt_(2))
        
    if branch==2:
        
        term1=x*log_(α)/4#x*(1-x)*q1_hat/2#
        term2=0
        
        Δ=  sqt_(2/pi)*q1*κ*e_(-κ*κ/(2*(1-q1)))                                                   /(erf_(κ/sqt_(2*(1-q1)))*(1-q1)**(3/2))
        dΔ= sqt_(2/pi)   *κ*e_(-κ*κ/(2*(1-q1)))                                                   /(erf_(κ/sqt_(2*(1-q1)))*(1-q1)**(3/2))\
           +sqt_(2/pi)*q1*κ*e_(-κ*κ/(2*(1-q1)))*(-κ*κ/(2*(1-q1)**(2)))                            /(erf_(κ/sqt_(2*(1-q1)))*(1-q1)**(3/2))\
           +sqt_(2/pi)*q1*κ*e_(-κ*κ/(2*(1-q1)))*(-1)*(κ/(2*(1-q1))**(3/2))*derf_(κ/sqt_(2*(1-q1)))/(erf_(κ/sqt_(2*(1-q1)))*(1-q1)**(3/2))\
           +sqt_(2/pi)*q1*κ*e_(-κ*κ/(2*(1-q1)))*(3/2)                                             /(erf_(κ/sqt_(2*(1-q1)))*(1-q1)**(5/2))
           
           
        term3= α*x*(κ/(2*(1-q1))**(3/2))*derf_(κ/sqt_(2*(1-q1)))/erf_(κ/sqt_(2*(1-q1)))\
              -(α/2)*(x*dΔ/(1+x*Δ))
        
    return term1+term2+term3

def Entropy_1RSB_simplified_rescaled(q1,q1_hat,κ,α,x,branch):
 
    if branch==1:
        term1=-q1_hat/2+(1-2*x)*q1*q1_hat/2
        term2=x*q1_hat        +np.exp(-2*(x-1)*q1_hat)-2*x*q1_hat*np.exp(-2*(x-1)*q1_hat)
        
        zo=(κ-np.sqrt(2*(1-q1)*np.log(x)))/(np.sqrt(q1))
        dzo= -np.sqrt(2*(1-q1))*(1/x)     /(2*np.sqrt(np.log(x))*np.sqrt(q1))
        
        term3=α*(dzo/np.sqrt(2))*derf_(abs(zo)/np.sqrt(2))/erf_(abs(zo)/np.sqrt(2))
        
    if branch==2:
        term1=(1-q1)/(-4/log_(α))
        term2=0
        
        Δ=np.sqrt(2/pi)*q1*κ*np.exp(-κ*κ/(2*(1-q1)))/(erf_(κ/np.sqrt(2*(1-q1)))*(1-q1)**(3/2))
        
        term3=α*np.log(special.erf(κ/np.sqrt(2*(1-q1))))-(α/2)*Δ/(1+x*Δ)
        
    return term1+term2+term3

def Saddle_1RSB_simplified_rescaled(κ,α,x,guess,branch,print_):
    

    def Find_zero(dq1_0):
        q1_0=1-dq1_0*(-α/np.log(α))
        dq1_sol1=dq1_0
        dq1_sol2=dq1_0
        dq1_inter_sol1=dq1_0
        dq1_inter_sol2=dq1_0
        
        N_test=1000
        if branch==1:
            N_testo=N_test
        else:
            N_testo=int(9*N_test/10)
        d_dq_sol1=-dq1_0/N_test
        d_dq_sol2=-dq1_0/N_test
        
        
        q1_hat_=-(1/(2*(x-1)))*np.log((1-q1_0)/4)
        out_sol1=Deriv_Free_eneregy_1RSB_simplified_rescaled(q1_0,q1_hat_,κ,α,x,branch)

        out_sol2=out_sol1
        
        Conv_sol1=0
        Conv_sol2=0
        
        for k in range(N_testo):
            ### Sol number 1
            dq1_inter_sol1=dq1_inter_sol1+d_dq_sol1
            q1_inter_sol1=1-dq1_inter_sol1*(-α/np.log(α))
            q1_hat_inter=-(1/(2*(x-1)))*np.log((1-q1_inter_sol1)/4)

            out_inter=Deriv_Free_eneregy_1RSB_simplified_rescaled(q1_inter_sol1,q1_hat_inter,κ,α,x,branch)
 
            if out_inter*out_sol1<0:
                d_dq_sol1=-d_dq_sol1/2
                dq1_sol1=dq1_inter_sol1
                out_sol1=out_inter
                
                Conv_sol1+=1
                
                
            ### Sol number 2
            dq1_inter_sol2=dq1_inter_sol2+d_dq_sol2
            q1_inter_sol2=1-dq1_inter_sol2*(-α/np.log(α))
            q1_hat_inter=-(1/(2*(x-1)))*np.log((1-q1_inter_sol2)/4)

            out_inter=Deriv_Free_eneregy_1RSB_simplified_rescaled(q1_inter_sol2,q1_hat_inter,κ,α,x,branch)
            
            if Conv_sol1<10:
                dq1_sol2=dq1_inter_sol2
                out_sol2=out_inter
            else:
                if out_inter*out_sol2<0:
                    d_dq_sol2=-d_dq_sol2/2
                    dq1_sol2=dq1_inter_sol2
                    out_sol2=out_inter
                
                    Conv_sol2+=1
                else:
                    dq1_sol2=dq1_inter_sol2
                    out_sol2=out_inter
                
                
                
            if Conv_sol1>13 and Conv_sol2>13:
                break
                

        return dq1_sol1,dq1_sol2
            
    
    dq1_0,dq1_1=Find_zero(guess[0])

      
    if print_==1:
        Nq=250
        x_list=np.zeros(Nq)
        y_list=np.zeros(Nq)
        z_list=np.zeros(Nq)
        
        dqmin=0
        dqmax=1.01*guess[0]
        
        for k in range(Nq):
            x_list[k]=dqmin+(dqmax-dqmin)*(k/Nq)
            q1_=1-x_list[k]*(-α/np.log(α))
            dq1=(10**-2)*(1-q1_)
            q1_hat_=-(1/(2*(x-1)))*np.log((1-q1_)/4)
        
            z_list[k]=Deriv_Free_eneregy_1RSB_simplified_rescaled(q1_,q1_hat_,κ,α,x,branch)*0.1/x
            y_list[k]=0#(Free_eneregy_1RSB_simplified(q1_+dq1,q1_hat_,κ,α,x,branch)-Free_eneregy_1RSB_simplified(q1_,q1_hat_,κ,α,x,branch))*0.1/(x*dq1)
            if math.isnan(z_list[k]):
                z_list[k]=z_list[k-1]
                
         
            
        plt.plot(x_list,z_list)
       # plt.plot(x_list,y_list)
        
        plt.plot([dq1_0,dq1_0],[min(z_list),max(z_list)])
        plt.plot([dq1_1,dq1_1],[min(z_list),max(z_list)])
        
        plt.plot([guess[0],guess[0]],[min(z_list),max(z_list)],c='black',linestyle='--')

        plt.plot([dqmin,dqmax],[0,0],c='black',linestyle='--')
       
      
        plt.xlabel('x='+str(x)+"  "+str(branch)+" "+str(max(z_list)))
        plt.show()
        
            
    return dq1_0,dq1_1
    
def Stability_SE_curve(κ,α,x,branch):
    Nq=500
    x_list=np.zeros(Nq)
    y_list=np.zeros(Nq)
    z_list=np.zeros(Nq)
    

    
    if branch==2:
        dq_max=0.225
        dq_min=0.015
        fmt_x = lambda x, pos: '1'+'{:.0f}'.format((x-1)*10**13, pos)+r'$\times10^{-13}$' if x!=1 else '1'
        fmt_y = lambda x, pos: '1'+'{:.0f}'.format((x-1)*10**13, pos)+r'$\times10^{-13}$' if x!=1 else '1'

    if branch==1:
        dq_max=0.1
        dq_min=0.0015
        fmt_x = lambda x, pos: '1'+'{:.0f}'.format((x-1)*10**13, pos)+r'$\times10^{-13}$' if x!=1 else '1'
        fmt_y = lambda x, pos: '1'+'{:.0f}'.format((x-1)*10**13, pos)+r'$\times10^{-13}$' if x!=1 else '1'


    ymax=1+0.5*10**-13
    ymin=1-2*dq_max*(-α/log_(α))
    for i in range(Nq):
        dq=dq_min+(dq_max-dq_min)*i/Nq
        q1=1-dq*(-α/log_(α))

        
        q1_hat=(2/(x*(x-1)))*Deriv_Free_eneregy_1RSB_simplified(q1,0,κ,α,x,branch)
        x_list[i]=1-dq*(-α/log_(α))
        y_list[i]=1-4*e_(-2*(x-1)*q1_hat)
        
        z_list[i]=1-dq*(-α/log_(α))



    fig, axs = mpl.pylab.subplots(1, 1)

    axs.plot(x_list,z_list,label=r'$f(q_1)=q_1$')
    axs.plot(x_list,y_list,label=r'$f(q_1)=1-4e^{-2(x-1)\hat{q}_1}$')
    


    plt.ylabel(r'$f(q_1)$')
    plt.ylim([ymin,ymax])
    axs.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt_y))
    
    plt.xlabel(r'$q_1$'+r"$\quad  (\tilde{\kappa}=1,\, \alpha=10^{-10},\,x=$"+str(round(x,1))+")")
    plt.xlim([x_list[Nq-1],x_list[0]])
    axs.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt_x))
    axs.xaxis.set_major_locator(plt.MaxNLocator(6))

    plt.legend()
    plt.show()








####### Parameters ##########
#############################
N=5000 #Not relevant for state evolution
Nt=2000
α=0.5
κ_SAT=κ_SAT_finding(α)[0]
print(κ_SAT)


####### Planted State evolution ########
N_κ_new=1
N_κ_old=30

q=np.zeros((N_κ_new,N_κ_old))
m=np.zeros((N_κ_new,N_κ_old))
q_hat=np.zeros((N_κ_new,N_κ_old))
m_hat=np.zeros((N_κ_new,N_κ_old))
Ф=np.zeros((N_κ_new,N_κ_old))
Σ=np.zeros((N_κ_new,N_κ_old))
κ_old_list=np.zeros((N_κ_new,N_κ_old))#[κ_SAT,κ_SAT+0.05,κ_SAT+0.1,κ_SAT+0.15]
κ_new_list=np.zeros((N_κ_new,N_κ_old))
cut_list=np.zeros((N_κ_new,N_κ_old))
stab=np.zeros((N_κ_new,N_κ_old))

SE=1

if SE==1:
    
    file=open('SE (test)(κ_SAT='+str(κ_SAT)+', α='+str(α)+').txt','w')
    file.write("κ_old	κ_new	q	m	q_hat	m_hat	Ф")
    file.write("\n")

    
    for d_κ in range(N_κ_new):
        
        κ_new_min=1.2*κ_SAT
        κ_new_max=0.8
            
        breaking=0
        κ_energ=0

        if  SE==1:
            for i in range(N_κ_old):
                list_=[0.47]
                κ_new=list_[d_κ]#κ_new_min+(κ_new_max-κ_new_min)*(d_κ+1)/N_κ_new
                κ_new_list[d_κ,i]=κ_new
                
                κ_old_min=κ_SAT
                κ_old_max=0.97*κ_new
                
                κ_old=κ_old_max-(κ_old_max-κ_old_min)*(i+1)/N_κ_old#κ_old_list[i]
                
                if d_κ==0:
                    q[d_κ,i],m[d_κ,i],q_hat[d_κ,i],m_hat[d_κ,i],cut_list[d_κ,i]=State_evolution([Nt,0.99999372, 0.999993553,κ_old,κ_new,α]) 
                    Ф[d_κ,i]=-Free_energy(q[d_κ,i],m[d_κ,i],q_hat[d_κ,i],m_hat[d_κ,i],κ_old,κ_new,α)
                    Σ[d_κ,i]=-Free_energy(0,0,0,0,κ_old,κ_old,α)
                    stab[d_κ,i]=Stability(q[d_κ,i],m[d_κ,i],q_hat[d_κ,i],m_hat[d_κ,i],κ_old,κ_new,α,cut_list[d_κ,i])
                else:
                    if breaking==0:
                        q[d_κ,i],m[d_κ,i],q_hat[d_κ,i],m_hat[d_κ,i],cut_list[d_κ,i]=State_evolution([Nt,q[d_κ-1,i],m[d_κ-1,i],κ_old,κ_new,α]) 
                        Ф[d_κ,i]=-Free_energy(q[d_κ,i],m[d_κ,i],q_hat[d_κ,i],m_hat[d_κ,i],κ_old,κ_new,α)
                        Σ[d_κ,i]=-Free_energy(0,0,0,0,κ_old,κ_old,α)
                        stab[d_κ,i]=Stability(q[d_κ,i],m[d_κ,i],q_hat[d_κ,i],m_hat[d_κ,i],κ_old,κ_new,α,cut_list[d_κ,i])
                    else:
                        q[d_κ,i]=q[d_κ-1,i]
                        m[d_κ,i]=m[d_κ-1,i]
                        q_hat[d_κ,i]=q_hat[d_κ-1,i]
                        m_hat[d_κ,i]=m_hat[d_κ-1,i]
                        Ф[d_κ,i]=Ф[d_κ-1,i]
                        Σ[d_κ,i]=Σ[d_κ-1,i]
                
                
                if q[d_κ,i]<0.9:
                    if breaking==0:
                        κ_energ=κ_new
                        
                    print(κ_new_list[d_κ-1,i],m[d_κ-1,i],q[d_κ-1,i])
                    breaking=1
                    
                file.write(str(κ_old))
                file.write("	")
                file.write(str(κ_new))
                file.write("	")
                file.write(str(q[d_κ,i]))
                file.write("	")
                file.write(str(m[d_κ,i]))
                file.write("	")
                file.write(str(q_hat[d_κ,i]))
                file.write("	")
                file.write(str(m_hat[d_κ,i]))
                file.write("	")
                file.write(str(-Ф[d_κ,i]))
                file.write("\n")
            
        if SE==0:
            κ_new=51
            q,m,q_hat,m_hat= optimize.fsolve(Free_energy_extr_routine,[0.9999,0.9998,20,20])

        
    file.close()                


if SE==0:
        
    file=open('SE (κ_SAT='+str(κ_SAT)+', α='+str(α)+').txt','r')
    file.readline()
    for i in range(N_κ_old):
        for d_κ in range(N_κ_new):
            
            print(i,N_κ_old)
            
            line=file.readline()
            line=line.split()
            κ_old=float(line[0])
            
            κ_new_list[d_κ,i]=float(line[1])
            q[d_κ,i]=float(line[2])
            m[d_κ,i]=float(line[3])
            q_hat[d_κ,i]=float(line[4])
            m_hat[d_κ,i]=float(line[5])
            Ф[d_κ,i]=-float(line[6])
            Σ[d_κ,i]=-Free_energy(0,0,0,0,κ_old,κ_old,α)
    file.close()         

for k in range(N_κ_new):
    plt.plot(Ф[k,:],Σ[k,:])
plt.show()

print("!!!!!! Stab !!!!!")
print(stab)



####### Planted simplified saddle point ########
α=10**-8
N_κ_new=1
N_κ_old=1
rescaling=1

κ_new_list=[1.1,1.2,1.3,1.4]

κ_old_rescaled=np.zeros(N_κ_old)
κ_new_rescaled=np.zeros(N_κ_new)
dm_rescaled=np.zeros((N_κ_new,N_κ_old))
m_simple=np.zeros((N_κ_new,N_κ_old))
Ф_simple=np.zeros((N_κ_new,N_κ_old))
Σ_simple=np.zeros((N_κ_new,N_κ_old))

file=open('Rescaled_planting.txt','w')
file.write('κ_old	κ_new	1-m*m	entropy	complexity	(all rescaled)')
file.write("\n")

for i in range(N_κ_new):

    κ_new=κ_new_list[i]*np.sqrt(-α/np.log(α))
    κ_new_r=κ_new_list[i]   
    κ_old_min=(0.2/15)*κ_new
    κ_old_max=(14/15)*κ_new
            
    breaking=0
    k_break=0
    
    for k in range(N_κ_old):

        if breaking==0:
            κ_old=κ_old_max-(κ_old_max-κ_old_min)*k/N_κ_old
            κ_old_r=κ_old/(np.sqrt(-α/np.log(α)))
            κ_old_rescaled[k]=κ_old_r
        else:
            κ_old=κ_old_max-(κ_old_max-κ_old_min)*k_break/N_κ_old
            κ_old_r=κ_old/(np.sqrt(-α/np.log(α)))
            κ_old_rescaled[k]=κ_old_r
            
        
        

        def Find_saddle(dm_):
            m_=1-np.exp(-min(abs(dm_[0]),30))
            d_m_=(1-m_*m_)*10**(-4)
            
            eq=(Free_energy_planted_simplified(m_+d_m_,κ_old,κ_new,α)-Free_energy_planted_simplified(m_,κ_old,κ_new,α))/d_m_
            
            return abs(eq)
        
        def Find_saddle_rescaled(dm):
            dm_=dm*(10**-4)
            eq=(Free_energy_planted_simplified_rescaled(dm+dm_,κ_old_r,κ_new_r,α)-Free_energy_planted_simplified_rescaled(dm,κ_old_r,κ_new_r,α))/(dm_*α)
            
            return abs(eq)




        if rescaling==0:
            if k==0:
                dm=[-25]
            
            if breaking==0:
                dm=optimize.fmin(Find_saddle,dm,disp=False)
        
                m_simple[0,k]=1-np.exp(-abs(dm[0]))
                print(k,κ_old,m_simple[0,k])
                print("")
                Ф_simple[0,k]=Free_energy_planted_simplified(m_simple[0,k],κ_old,κ_new,α)/α
                Σ_simple[0,k]=-Free_energy(0,0,0,0,κ_old,κ_old,α)
                
                print_=0
                if print_==1:
                    N_test=500
                    x_list=np.zeros(N_test)
                    y_list=np.zeros(N_test)
                    qmin=0.999995
                    qmax=0.9999999
                    for j in range(N_test):
                        x_list[j]=qmin*np.exp(np.log(qmax/qmin)*(j/N_test))
                        y_list[j]=Free_energy_planted_simplified(x_list[j],κ_old,κ_new,α)
            
                    plt.plot(x_list,y_list)
                    plt.plot([m_simple[0,k],m_simple[0,k]],[min(y_list),max(y_list)])
                    plt.show()
                    
                if Find_saddle(dm)>10**-7:
                    breaking=1
                    k_break=k
                    
            else:
                m_simple[0,k]=1-np.exp(-abs(dm[0]))
                print(k,κ_old,m_simple[0,k])
                print("")
                Ф_simple[0,k]=Free_energy_planted_simplified(m_simple[0,k],κ_old,κ_new,α)/α
                Σ_simple[0,k]=-Free_energy(0,0,0,0,κ_old,κ_old,α)
                    
                print_=0
                if print_==1:
                    N_test=500
                    x_list=np.zeros(N_test)
                    y_list=np.zeros(N_test)
                    qmin=0.999995
                    qmax=0.9999999
                    for j in range(N_test):
                        x_list[j]=qmin*np.exp(np.log(qmax/qmin)*(j/N_test))
                        y_list[j]=Free_energy_planted_simplified(x_list[j],κ_old,κ_new,α)
                
                    plt.plot(x_list,y_list)
                    plt.plot([m_simple[0,k],m_simple[0,k]],[min(y_list),max(y_list)])
                    plt.show()
            
        if rescaling==1:
            
            
            if k==0:
                dm=[0.001]
                
                
            if breaking==0:
                dm=optimize.fmin(Find_saddle_rescaled,dm,disp=False)
                dm=optimize.fsolve(Find_saddle_rescaled,dm)
        
                m_simple[i,k]=np.sqrt(1-dm[0]*(-α/np.log(α)))
                dm_rescaled[i,k]=dm[0]
                
                
                print("loop index: ",i,k)
                print("kappa_new/old (rescaled): ",κ_new_r,κ_old_r)
                print("1-m*m (rescaled), saddle error: ",dm_rescaled[i,k],Find_saddle_rescaled(dm))
                print("")
                
                
                Ф_simple[i,k]=Free_energy_planted_simplified_rescaled(dm[0],κ_old_r,κ_new_r,α)/α
                Σ_simple[i,k]=(-Free_energy(0,0,0,0,κ_old,κ_old,α)-np.log(2)-α*np.log(-α/np.log(α)))/α
                
                file.write(str(κ_old_r))
                file.write('	')
                file.write(str(κ_new_r))
                file.write('	')
                file.write(str(dm_rescaled[i,k]))
                file.write('	')
                file.write(str(Ф_simple[i,k]))
                file.write('	')
                file.write(str(Σ_simple[i,k]))
                file.write("\n")
                
                
                print_=0
                if print_==1:
                    N_test=500
                    x_list=np.zeros(N_test)
                    y_list=np.zeros(N_test)
                    dqmin=0.01
                    dqmax=4
                    for j in range(N_test):
                        x_list[j]=dqmin+(dqmax-dqmin)*(j/N_test)
                        y_list[j]=Free_energy_planted_simplified_rescaled(x_list[j],κ_old_r,κ_new_r,α)/α
            
                    plt.plot(x_list,y_list)
                    plt.plot([dm_rescaled[i,k],dm_rescaled[i,k]],[min(y_list),max(y_list)])
                    plt.show()
                
                if abs(Find_saddle_rescaled(dm))>10**(-4):
                    breaking=1
                    k_break=k
            
   
            else:
        
                m_simple[i,k]=np.sqrt(1-dm[0]*(-α/np.log(α)))
                dm_rescaled[i,k]=dm[0]
                
                
                print("loop index: ",i,k, 'loop broken')
                print("kappa_new/old (rescaled): ",κ_new_r,κ_old_r)
                print("1-m*m (rescaled), saddle error: ",dm_rescaled[i,k],'loop broken')
                print("")
                
                Ф_simple[i,k]=Ф_simple[i,k_break]
                Σ_simple[i,k]=Σ_simple[i,k_break]
   

         
        

if rescaling==1:
    for i in range(N_κ_new):
        plt.plot(Ф_simple[i,:],Σ_simple[i,:],label=r"$\tilde{\kappa}=$"+str(round(κ_new_list[i],4)))
    plt.xlabel(r"$\frac{s[\tilde{\kappa}',\tilde{\kappa}]}{\alpha}$")
    plt.ylabel(r"$\frac{\Sigma[\tilde{\kappa}']-\Sigma_o}{\alpha}$")
    #plt.ylim([3.5,7.3])
    plt.legend()
    plt.show()
    
file.close()



for dκ in range(N_κ_new):
    κ=κ_new_rescaled[dκ]*np.sqrt(-α/np.log(α))
    plt.plot(κ_old_rescaled[:],dm_rescaled[dκ,:],linewidth=2.5)
    
plt.xlabel(r"$dm$")
plt.ylabel(r"$\kappa_o$")
plt.legend()
plt.savefig('plot.png')
plt.show()



####### 1-RSB x→∞ sadlle point ########
α=10**-8
Nx=1
x_max_1=200
x_min_1=25

x_max_2=10**5
x_min_2=50

Ф_1RSB     =np.zeros((N_κ_new,4,Nx))
s_1RSB     =np.zeros((N_κ_new,4,Nx))
Σ_1RSB     =np.zeros((N_κ_new,4,Nx))
q1_1RSB    =np.zeros((N_κ_new,4,Nx))
dq1_1RSB   =np.zeros((N_κ_new,4,Nx))
q1_hat_1RSB=np.zeros((N_κ_new,4,Nx))
x_list     =np.zeros((N_κ_new,4,Nx))



for dκ in range(N_κ_new):
    κ=κ_new_list[dκ]*np.sqrt(-α/np.log(α))
    break_branch1=0
    break_branch2=0
    
    for dx in range(Nx):

        x_1=x_max_1-(x_max_1-x_min_1)*(dx/Nx)
        x_2=x_max_2*e_(log_(x_min_2/x_max_2)*(dx/Nx))
        print(dκ,dx,x_1,x_2)
    
        if dx==0:
            if dκ==0:
                guess1=[0.102]
                guess2=[0.6]
                print_=0
            if dκ==1:
                guess1=[0.13]
                guess2=[0.6]
                print_=0
            if dκ==2:
                guess1=[0.145]
                guess2=[0.6]
                print_=0
            if dκ==3:
                guess1=[0.18]
                guess2=[0.6]
                print_=0
        else:
            guess1=[1.02*dq1_1RSB[dκ,2,dx-1]]
            guess2=[1.02*dq1_1RSB[dκ,0,dx-1]]
            print_=0
   
    
   
    
    
        branch=2
        if break_branch1==0:
            
            if dx==int(20*Nx/21):
                Stability_SE_curve(κ,α,x_2,branch)
                print_=0
            else:
                print_=0
            x_list[dκ,0,dx]=x_2
            x_list[dκ,1,dx]=x_2
            
            dq1_1RSB[dκ,0,dx],dq1_1RSB[dκ,1,dx]=Saddle_1RSB_simplified_rescaled(κ,α,x_2,guess2,branch,print_)
            q1_1RSB[dκ,0,dx]=1-dq1_1RSB[dκ,0,dx]*(-α/log_(α))
            q1_1RSB[dκ,1,dx]=1-dq1_1RSB[dκ,1,dx]*(-α/log_(α))
        
            q1_hat_1RSB[dκ,0,dx]=-(1/(2*(x_2-1)))*np.log((1-q1_1RSB[dκ,0,dx])/4)
            Ф_1RSB[dκ,0,dx]=Free_eneregy_1RSB_simplified_rescaled(q1_1RSB[dκ,0,dx],q1_hat_1RSB[dκ,0,dx],κ,α,x_2,branch)
            s_1RSB[dκ,0,dx]=Entropy_1RSB_simplified_rescaled(q1_1RSB[dκ,0,dx],q1_hat_1RSB[dκ,0,dx],κ,α,x_2,branch)
            Σ_1RSB[dκ,0,dx]=Ф_1RSB[dκ,0,dx]-x_2*s_1RSB[dκ,0,dx]
           

            q1_hat_1RSB[dκ,1,dx]=-(1/(2*(x_2-1)))*np.log((1-q1_1RSB[dκ,1,dx])/4)
            Ф_1RSB[dκ,1,dx]=Free_eneregy_1RSB_simplified_rescaled(q1_1RSB[dκ,1,dx],q1_hat_1RSB[dκ,1,dx],κ,α,x_2,branch)
            s_1RSB[dκ,1,dx]=Entropy_1RSB_simplified_rescaled(q1_1RSB[dκ,1,dx],q1_hat_1RSB[dκ,1,dx],κ,α,x_2,branch)
            Σ_1RSB[dκ,1,dx]=Ф_1RSB[dκ,1,dx]-x_2*s_1RSB[dκ,1,dx]
            

                
            
            if dx>1 and (q1_1RSB[dκ,1,dx]-q1_1RSB[dκ,1,dx-1])>(1-q1_1RSB[dκ,1,dx])*10**(-2):
                break_branch1=1
                
                x_list[dκ,0,dx]=x_list[dκ,0,dx-1]
                x_list[dκ,1,dx]=x_list[dκ,1,dx-1]
                
                q1_1RSB[dκ,0,dx]=q1_1RSB[dκ,0,dx-1]
                q1_hat_1RSB[dκ,0,dx]=q1_hat_1RSB[dκ,0,dx-1]
                Ф_1RSB[dκ,0,dx]=Ф_1RSB[dκ,0,dx-1]
                s_1RSB[dκ,0,dx]=s_1RSB[dκ,0,dx-1]
                Σ_1RSB[dκ,0,dx]=Σ_1RSB[dκ,0,dx-1]
            

                q1_1RSB[dκ,1,dx]=q1_1RSB[dκ,1,dx-1]
                q1_hat_1RSB[dκ,1,dx]=q1_hat_1RSB[dκ,1,dx-1]
                Ф_1RSB[dκ,1,dx]=Ф_1RSB[dκ,1,dx-1]
                s_1RSB[dκ,1,dx]=s_1RSB[dκ,1,dx-1]
                Σ_1RSB[dκ,1,dx]=Σ_1RSB[dκ,1,dx-1]
  
                
        else:
            x_list[dκ,0,dx]=x_list[dκ,0,dx-1]
            x_list[dκ,1,dx]=x_list[dκ,1,dx-1]
            
            q1_1RSB[dκ,0,dx]=q1_1RSB[dκ,0,dx-1]
            q1_hat_1RSB[dκ,0,dx]=q1_hat_1RSB[dκ,0,dx-1]
            Ф_1RSB[dκ,0,dx]=Ф_1RSB[dκ,0,dx-1]
            s_1RSB[dκ,0,dx]=s_1RSB[dκ,0,dx-1]
            Σ_1RSB[dκ,0,dx]=Σ_1RSB[dκ,0,dx-1]
        

            q1_1RSB[dκ,1,dx]=q1_1RSB[dκ,1,dx-1]
            q1_hat_1RSB[dκ,1,dx]=q1_hat_1RSB[dκ,1,dx-1]
            Ф_1RSB[dκ,1,dx]=Ф_1RSB[dκ,1,dx-1]
            s_1RSB[dκ,1,dx]=s_1RSB[dκ,1,dx-1]
            Σ_1RSB[dκ,1,dx]=Σ_1RSB[dκ,1,dx-1]
            
            
            
                
    
    
        dx_=0.01
        branch=1
        if break_branch2==0:
            
            if dx==int(20*Nx/21):
                Stability_SE_curve(κ,α,x_1,branch)
                print_=0
            else:
                print_=0
            
            x_list[dκ,2,dx]=x_1
            x_list[dκ,3,dx]=x_1
            
            dq1_1RSB[dκ,2,dx],dq1_1RSB[dκ,3,dx]=Saddle_1RSB_simplified_rescaled(κ,α,x_1,guess1,branch,print_)
            q1_1RSB[dκ,2,dx]=1-dq1_1RSB[dκ,2,dx]*(-α/log_(α))
            q1_1RSB[dκ,3,dx]=1-dq1_1RSB[dκ,3,dx]*(-α/log_(α))
            
            q1_hat_1RSB[dκ,2,dx]=-(1/(2*(x_1-1)))*np.log((1-q1_1RSB[dκ,2,dx])/4)
            Ф_1RSB[dκ,2,dx]=Free_eneregy_1RSB_simplified_rescaled(q1_1RSB[dκ,2,dx],q1_hat_1RSB[dκ,2,dx],κ,α,x_1,branch)
            s_1RSB[dκ,2,dx]=Entropy_1RSB_simplified_rescaled(q1_1RSB[dκ,2,dx],q1_hat_1RSB[dκ,2,dx],κ,α,x_1,branch)
            Σ_1RSB[dκ,2,dx]=Ф_1RSB[dκ,2,dx]-x_1*s_1RSB[dκ,2,dx]
        
            q1_hat_1RSB[dκ,3,dx]=-(1/(2*(x_1-1)))*np.log((1-q1_1RSB[dκ,3,dx])/4)
            Ф_1RSB[dκ,3,dx]=Free_eneregy_1RSB_simplified_rescaled(q1_1RSB[dκ,3,dx],q1_hat_1RSB[dκ,3,dx],κ,α,x_1,branch)
            s_1RSB[dκ,3,dx]=Entropy_1RSB_simplified_rescaled(q1_1RSB[dκ,3,dx],q1_hat_1RSB[dκ,3,dx],κ,α,x_1,branch)
            Σ_1RSB[dκ,3,dx]=Ф_1RSB[dκ,3,dx]-x_1*s_1RSB[dκ,3,dx]
            
            test=Saddle_1RSB_test(q1_1RSB[dκ,3,dx],κ,α,x_1,branch)
            test_=math.isnan(test)
            
            if test_==True or s_1RSB[dκ,3,dx]<0:
                break_branch2=1
                
                x_list[dκ,2,dx]=x_list[dκ,2,dx-1]
                x_list[dκ,3,dx]=x_list[dκ,3,dx-1]
                
                q1_1RSB[dκ,2,dx]=q1_1RSB[dκ,2,dx-1]
                q1_hat_1RSB[dκ,2,dx]=q1_hat_1RSB[dκ,2,dx-1]
                Ф_1RSB[dκ,2,dx]=Ф_1RSB[dκ,2,dx-1]
                s_1RSB[dκ,2,dx]=s_1RSB[dκ,2,dx-1]
                Σ_1RSB[dκ,2,dx]=Σ_1RSB[dκ,2,dx-1]
            

                q1_1RSB[dκ,3,dx]=q1_1RSB[dκ,3,dx-2]
                q1_hat_1RSB[dκ,3,dx]=q1_hat_1RSB[dκ,3,dx-2]
                Ф_1RSB[dκ,3,dx]=Ф_1RSB[dκ,3,dx-2]
                s_1RSB[dκ,3,dx]=s_1RSB[dκ,3,dx-2]
                Σ_1RSB[dκ,3,dx]=Σ_1RSB[dκ,3,dx-2]
                
                
        else:
            
            x_list[dκ,2,dx]=x_list[dκ,2,dx-1]
            x_list[dκ,3,dx]=x_list[dκ,3,dx-1]
            
            q1_1RSB[dκ,2,dx]=q1_1RSB[dκ,2,dx-1]
            q1_hat_1RSB[dκ,2,dx]=q1_hat_1RSB[dκ,2,dx-1]
            Ф_1RSB[dκ,2,dx]=Ф_1RSB[dκ,2,dx-1]
            s_1RSB[dκ,2,dx]=s_1RSB[dκ,2,dx-1]
            Σ_1RSB[dκ,2,dx]=Σ_1RSB[dκ,2,dx-1]
        

            q1_1RSB[dκ,3,dx]=q1_1RSB[dκ,3,dx-1]
            q1_hat_1RSB[dκ,3,dx]=q1_hat_1RSB[dκ,3,dx-1]
            Ф_1RSB[dκ,3,dx]=Ф_1RSB[dκ,3,dx-1]
            s_1RSB[dκ,3,dx]=s_1RSB[dκ,3,dx-1]
            Σ_1RSB[dκ,3,dx]=Σ_1RSB[dκ,3,dx-1]


couleur__ =['red' ,'navy'        ,'brown'     ,'green']
couleur_=['pink','lightskyblue','sandybrown','chartreuse']

for dκ in range(N_κ_new):
    κ=κ_new_rescaled[dκ]*np.sqrt(-α/np.log(α))
    plt.plot(Ф_simple[dκ,:],Σ_simple[dκ,:],c=couleur_[dκ],linewidth=2.5)
    plt.plot(s_1RSB[dκ,0,:]/α,(Σ_1RSB[dκ,0,:]-np.log(2)-α*np.log(-α/np.log(α)))/α,linestyle='--',c=couleur__[dκ])
   # plt.plot(s_1RSB[dκ,2,:]/α,(Σ_1RSB[dκ,2,:]-np.log(2)-α*np.log(-α/np.log(α)))/α,label='1-RSB (2nd regime)')
    plt.plot(s_1RSB[dκ,3,:]/α,(Σ_1RSB[dκ,3,:]-np.log(2)-α*np.log(-α/np.log(α)))/α,linestyle='dotted',c=couleur__[dκ])
    
plt.xlabel(r"$\frac{s[\tilde{\kappa}',\tilde{\kappa}]}{\alpha}$")
plt.ylabel(r"$\frac{\Sigma[\tilde{\kappa}']-\Sigma_o}{\alpha}$")
plt.legend()
plt.savefig('plot.png')
plt.ylim([7,11])
plt.show()


for dκ in range(N_κ_new):
    κ=κ_new_rescaled[dκ]*np.sqrt(-α/np.log(α))
    plt.plot(Ф_simple[dκ,:],dm_rescaled[dκ,:],c=couleur_[dκ],linewidth=2.5)
    plt.plot(s_1RSB[dκ,0,:]/α,dq1_1RSB[dκ,0,:],linestyle='--',c=couleur__[dκ])
    plt.plot(s_1RSB[dκ,3,:]/α,dq1_1RSB[dκ,3,:],linestyle='dotted',c=couleur__[dκ])
    
plt.xlabel(r"$\frac{s[\tilde{\kappa}',\tilde{\kappa}]}{\alpha}$")
plt.ylabel(r"$\frac{\Sigma[\tilde{\kappa}']-\Sigma_o}{\alpha}$")
plt.legend()
plt.savefig('plot.png')
plt.show()

for dκ in range(N_κ_new):
    κ=κ_new_rescaled[dκ]*np.sqrt(-α/np.log(α))
    plt.plot(1/log_(x_list[dκ,0,:]),dq1_1RSB[dκ,0,:],linestyle='--',c=couleur__[dκ])
    plt.plot(1/log_(x_list[dκ,3,:]),dq1_1RSB[dκ,3,:],linestyle='dotted',c=couleur__[dκ])
    
plt.xlabel(r"$x$")
plt.ylabel(r"$dq$")
plt.legend()
plt.savefig('plot.png')
plt.show()

 

##### Curve ɸ_FP(m) vs m for k_old=0 #####
Nm=1
N_κ_new=1
κ_new_list=[1.15,1.239,1.34,1.429,1.46,1.5]

κ_new=np.zeros(N_κ_new)
m_simple=np.zeros((N_κ_new,Nm))
dm_simple=np.zeros((N_κ_new,Nm))
Ф_simple=np.zeros((N_κ_new,Nm))
Σ_simple=np.zeros((N_κ_new,Nm))

for i in range(N_κ_new):
    κ_new[i]=κ_new_list[i]*np.sqrt(-α/np.log(α))
    
    for j in range(Nm):
        dm=2*(j/Nm)*(-α/np.log(α))
        dm_simple[i,j]=dm/(-α/np.log(α))
        
        m_simple[i,j]=np.sqrt(1-dm)
        Ф_simple[i,j]=Free_energy_planted_simplified_kold_zero_rescaled(dm_simple[i,j],κ_new_list[i],α)/α
    
   # if i==0:
   #     plt.plot(dm_simple[i,:],Ф_simple[i,:],label=r'$\tilde{\kappa}<\tilde{\kappa}_{\rm energ.}$')
   # if i==1:
   #     plt.plot(dm_simple[i,:],Ф_simple[i,:],label=r'$\tilde{\kappa}=\tilde{\kappa}_{\rm energ.}$')
   # if i==2:
   #     plt.plot(dm_simple[i,:],Ф_simple[i,:],label=r'$\tilde{\kappa}_{\rm energ.}<\tilde{\kappa}<\tilde{\kappa}_{\rm entro.}$')
   # if i==3:
   #     plt.plot(dm_simple[i,:],Ф_simple[i,:],label=r'$\tilde{\kappa}=\tilde{\kappa}_{\rm entro.}$')
   # if i==4:
   #     plt.plot(dm_simple[i,:],Ф_simple[i,:],label=r'$\tilde{\kappa}>\tilde{\kappa}_{\rm entro.}$')
   
    plt.plot(dm_simple[i,:],Ф_simple[i,:],label=r'$\tilde{\kappa}=$'+str(κ_new_list[i]))
    plt.plot([0,2],[0,0],c='black',linestyle='--')
    plt.xlabel(r'$\tilde{m}$')
    plt.ylabel(r'$\phi_{\rm planted}^{\kappa_{\rm SAT},\tilde{\kappa}}[\tilde{m}]/\alpha$')
    plt.legend()
    #plt.xlim([0,2])
    
plt.show()

   
    

    
    
    

