import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt
import math 
from multiprocessing.pool import ThreadPool as Pool
pi=np.arccos(-1)



###### Computing κ_SAT ######
#############################
def κ_SAT_finding(α):
    def α_critical(κ):
        out= α+np.log(2)/np.log(special.erf(κ/np.sqrt(2)))
        print("in (Computing κ_SAT):",κ)
        print("out:",out)
        print("")
        return  abs(out)*np.heaviside(κ,1)+10000*(1-np.heaviside(κ,1))
    
    κ_SAT=optimize.fmin(α_critical,0.4)
    return κ_SAT

########## AMP ##############
#############################
def AMP(x,x_vec,G):
    N=int(x[0])
    Nt=int(x[1])
    α=x[2]
    M=int(α*N)
    
    κ_old=x[3]
    κ_new=x[4]

    
    η_in=0.01
    η_out=1
    
    x=np.zeros((N,Nt))
    x_new=np.zeros((N,Nt))
    V=np.zeros(Nt)
    V_new=np.zeros(Nt)
    
    B=np.zeros((N,Nt))
    B_new=np.zeros((N,Nt))
    w=np.zeros((M,Nt))
    w_new=np.zeros((M,Nt))
    g=np.zeros((M,Nt))
    g_new=np.zeros((M,Nt))
    A=np.zeros(Nt)
    A_new=np.zeros(Nt)
    
    xo=np.zeros(N)+1
    Gt=np.transpose(G)
    
    m_list=np.zeros(Nt)
    
    ########## Channels #############
    #################################
    def Ф_out(w,V,κ,α):
        eps=10**(-40)
        
        out2=(special.erf((κ-w)/np.sqrt(2*V+eps)) + special.erf((κ+w)/np.sqrt(2*V+eps)))
        out3=np.log(out2*np.heaviside(abs(out2)-eps,1)  +  eps*(1-np.heaviside(abs(out2)-eps,1)))
        
        return out3
    
    def dw_Ф_out(w,V,κ,α):
        eps=10**(-50)
        out1=(np.sqrt(2/(pi*V)))*( -np.exp(-((κ-w)**2)/(2*V))+np.exp(-((κ+w)**2)/(2*V)) )
        out2=(special.erf((κ-w)/np.sqrt(2*V)) + special.erf((κ+w)/np.sqrt(2*V)))
        out3=out1*np.heaviside(abs(out2)-eps,1)/(out2 +  (1-np.heaviside(abs(out2)-eps,1))  )
        
        return out3
    
    def dw2_Ф_out(w,V,κ,α):
        dw=1*10**(-4)
        
        out1=dw_Ф_out(w+dw,V,κ,α)
        out2=dw_Ф_out(w,V,κ,α)
        out=((out1-out2)/dw)
        
        return out
    
    def dB_Ф_in(B):
        out=np.tanh(B)
        return out



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
                

        #N_test=500
        #x_list=np.zeros(N_test)
        #y_list=np.zeros(N_test)
        #xmax=10
        #xmin=-10
        #for k in range(N_test):
        #    x_list[k]=xmin+(xmax-xmin)*(k/N_test)
        #    y_list[k]=potential_simple(x_list[k])
        #plt.plot(x_list,y_list)
        #plt.xlabel("q_test m="+str(round(m,5))+",  q="+str(round(q,5)))
        #plt.show()
            
        return α*integrate.quad(potential_simple,-cut,cut)[0]
       
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
            
            
        #N_test=500
        #x_list=np.zeros(N_test)
        #y_list=np.zeros(N_test)
        #xmax=10
        #xmin=-10
        #for k in range(N_test):
        #    x_list[k]=xmin+(xmax-xmin)*(k/N_test)
        #    y_list[k]=potential_simple(x_list[k])
        #    y_list[k]=np.log(max(abs(y_list[k]),eps))
        #plt.plot(x_list,y_list)
        #plt.xlabel("m_test m="+str(round(m,5))+",  q="+str(round(q,5)))
        #plt.show()    
        
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

    def dqhat_Ф_in(q_hat,m_hat):
        eps=10**(-10)
        xo=1
        potential_1= lambda t: (1/(2*np.sqrt(q_hat+eps)))*t*np.tanh(m_hat*xo+np.sqrt(q_hat+eps)*t)
        potential_2= lambda t: potential_1(t)*(np.exp(-t*t/2)/np.sqrt(2*pi))
        
        return integrate.quad(potential_2, -np.Infinity, np.Infinity)[0]
    
    def dmhat_Ф_in(q_hat,m_hat):
        eps=10**(-10)
        xo=1
        potential_1= lambda t: xo*np.tanh(m_hat*xo+np.sqrt(q_hat+eps)*t)
        potential_2= lambda t: potential_1(t)*(np.exp(-t*t/2)/np.sqrt(2*pi))
        
        return integrate.quad(potential_2, -np.Infinity, np.Infinity)[0]
    


    ####### Initialization ##########
    #################################
    x[:,0]=x_vec[:]
    V[0]=(1-np.sum(x[:,0]**2)/N)
    w[:,0]=np.dot(G,x[:,0])
    

    for dt in range(1,Nt):
        P = lambda w: np.exp(-w*w/2)
        N_w=integrate.quad(P, -κ_old, κ_old)[0]
        
        ########## Out ##############
        #############################
        w_new[:,dt]=np.dot(G,x[:,dt-1]) - g[:,dt-1]*(V[dt-1])
        w[:,dt]=(1-η_out)*w[:,dt-1]+η_out*w_new[:,dt]  
        
        g_new[:,dt]=dw_Ф_out(w[:,dt],V[dt-1],κ_new,α)
        g[:,dt]=(1-η_out)*g[:,dt-1]+η_out*g_new[:,dt]   
        
        A_new[dt]=-α*np.sum(dw2_Ф_out(w[:,dt],V[dt-1],κ_new,α))/M
        A[dt]=(1-η_out)*A[dt-1]+η_out*A_new[dt]
    
        B_new[:,dt]=x[:,dt-1]*A[dt] + np.dot(Gt,g[:,dt])
        B[:,dt]=(1-η_out)*B[:,dt-1]+η_out*B_new[:,dt]  
        
    
        
        ########## In ##############
        ############################
        x_new[:,dt]=dB_Ф_in(B[:,dt])
        x[:,dt]=(1-η_in)*x[:,dt-1]+η_in*x_new[:,dt] 

        V[dt]=(1-np.sum(x[:,dt]**2)/N)
        
        print("t,α:",dt,α)
        print("t,α,κ_old,κ_new:",round(κ_old,6),round(κ_new,6))
        print("q,m:",round(np.sum(x[:,dt]**2)/N,8),round(np.sum(x[:,dt]*xo[:])/N,8))
        print("conv", round(abs(np.sum(x[:,dt]**2)/N-np.sum(x[:,dt-1]**2)/N),10))
        print("")
        
        m_hat=np.sum(B[:,dt-1]*xo[:])/N 
        q_hat=np.sum((B[:,dt-1]-m_hat*xo[:])**2)/N 
        q=np.sum(x[:,dt-1]**2)/N
        m=np.sum(x[:,dt-1]*xo[:])/N
        
        m_list[dt]=m
        
        #if abs(np.sum(x[:,dt]**2)/N-np.sum(x[:,dt-1]**2)/N)<1*10**(-8):
        #    print("conv!!!!!")
        #    x[:,Nt-1]=x[:,dt]
        #    B[:,Nt-1]=B[:,dt]
        #    break
    m_list[0]
    plt.plot(m_list,label='size='+str(N))
    return x[:,Nt-1],q_hat,m_hat
    
            
 


####### Free Energy #########
#############################
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






####### Parameters ##########
#############################
No=2000 #Dimension for x
Nt=900000 #Number of steps for AMP
α=5

κ_SAT=κ_SAT_finding(α)[0]
print('κ_SAT:',κ_SAT)
      

N_κ_new=1 # Number of κ_new we'll look at
N_κ_old=1 # Number of κ_old we'll look at


q=np.zeros((N_κ_new,N_κ_old))
m=np.zeros((N_κ_new,N_κ_old))
q_hat=np.zeros((N_κ_new,N_κ_old))
m_hat=np.zeros((N_κ_new,N_κ_old))
Ф=np.zeros((N_κ_new,N_κ_old))



κ_old_list=[κ_SAT*0.95]

κ_new_max=κ_SAT*0.95
κ_new_min=κ_SAT*0.95
κ_new_list=np.zeros((N_κ_new,N_κ_old))




write=1


if write==1:
    
    file=open('AMP (test)(κ_SAT='+str(κ_SAT)+', α='+str(α)+').txt','w')
    file.write("κ_old	κ_new	q	m	q_hat	m_hat	Ф")
    file.write("\n")
    
    
    ########## Loop running diffenrent κ_old ##############
    #######################################################   
    for k in range(N_κ_old):
        κ_old=κ_old_list[0]
        N=No*(1+k)
        
        x=np.zeros((N,N_κ_new,N_κ_old))
        M=int(α*N)
        
        ########## Generated G ##############
        #####################################
        xo=np.zeros(N)+1
        G=np.zeros((M,N)) 
        G_o=np.zeros((M,N))   
        G_scalar_prod=np.zeros(M)
        u=np.zeros(M)
        
        for i in range(M):
            print("matrix creation:",i, "out of",M)
            for j in range(N):
                G_o[i,j]=np.random.normal(0, 1, 1)
                G_scalar_prod[i]+=xo[j]*G_o[i,j]
                
        for i in range(M):
            for j in range(N):        
                G[i,j]=G_o[i,j]-G_scalar_prod[i]*xo[j]/N
        
        for i in range(M):
            for test in range(5000):
                u[i]=np.random.normal(0, 1, 1)
                if abs(u[i])<κ_old:
                    break
            G[i,:]+=u[i]*xo[:]/np.sqrt(N)
        
        G=G/np.sqrt(N)
        #for i in range(M):
        #    for j in range(N):
        #        file.write(str(G[i,j]))
        #        file.write("	 ")
        #    file.write("\n")
            
                
           
        ########## Runing AMP for different κ_new ##############
        ########################################################
        for d_κ in range(N_κ_new):
                κ_new=κ_new_min+(κ_new_max-κ_new_min)*(d_κ+0.002)/N_κ_new
                κ_new_list[d_κ,k]=κ_new
                    
                if d_κ==0:
                    
                    mo=0.
                    qo=0.999 
                    x_orth=np.zeros(N)
                    for n in range(N):
                        x_orth[k]=(-1)**n
                        x[n,d_κ,k]=mo*xo[n]+np.sqrt(qo-mo**2)*x_orth[n]
                    
                    x[:,d_κ,k],q_hat[d_κ,k],m_hat[d_κ,k]=AMP([N,Nt,α,κ_old,κ_new],x[:,d_κ,k],G)
                    q[d_κ,k]=np.sum(x[:,d_κ,k]**2)/N
                    m[d_κ,k]=np.sum(x[:,d_κ,k]*xo[:])/N
                    Ф[d_κ,k]=Free_energy(q[d_κ,k],m[d_κ,k],q_hat[d_κ,k],m_hat[d_κ,k],κ_old,κ_new,α)
                    
                else:
                    
                    x[:,d_κ,k],q_hat[d_κ,k],m_hat[d_κ,k]=AMP([N,Nt,α,κ_old,κ_new],x[:,d_κ-1,k],G)
                    q[d_κ,k]=np.sum(x[:,d_κ,k]**2)/N
                    m[d_κ,k]=np.sum(x[:,d_κ,k]*xo[:])/N
                    Ф[d_κ,k]=Free_energy(q[d_κ,k],m[d_κ,k],q_hat[d_κ,k],m_hat[d_κ,k],κ_old,κ_new,α)
                
                file.write(str(κ_old_list[k]))
                file.write("	")
                file.write(str(κ_new_list[d_κ,k]))
                file.write("	")
                file.write(str(q[d_κ,k]))
                file.write("	")
                file.write(str(m[d_κ,k]))
                file.write("	")
                file.write(str(q_hat[d_κ,k]))
                file.write("	")
                file.write(str(m_hat[d_κ,k]))
                file.write("	")
                file.write(str(-Ф[d_κ,k]))
                file.write("\n")
        
            
        
        
if write==0:
        
    file=open('AMP (κ_SAT='+str(κ_SAT)+', α='+str(α)+').txt','r')
    file.readline()
    for i in range(N_κ_old):
        for d_κ in range(N_κ_new):
            line=file.readline()
            line=line.split()
            κ_new_list[d_κ,i]=line[1]
            q[d_κ,i]=line[2]
            m[d_κ,i]=line[3]
            
file.close()         

plt.legend()
plt.ylim([0.9,1])
plt.xlabel('iterations')
plt.ylabel('m')
plt.show()
    
    
    

    
    
    