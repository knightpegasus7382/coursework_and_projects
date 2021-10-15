from numpy import *
from math import pi
import sys                   # [Module containing the function to accept command-line arguments]
flagc=0; flage=0; flaga=0; ckt=[]; c=0; e=0; count=-1; freq=0.0; w=[0,0,0];
def truncate(p):
    return int(round(p*1000000000))/1000000000
if (len(sys.argv)>2):
    print("\nError: Too many command-line arguments/filenames, please enter one filename")
else:    
    try:
        f=open(sys.argv[1])
    except FileNotFoundError:
        print("\nError: Invalid file name")   # [This error is displayed when a file does not exist with the given filename]
    except IndexError:
        print("\nError: File name not entered")   # [This error is displayed when the filename argument is not entered in the command line at all]
    else:
        for line in f: 
            count+=1
            if (line.strip("\n")==".circuit"):
                flagc+=1; c=count
            if (line.strip("\n")==".end"):
                flage+=1; e=count
            if (line.split(' ')[0].strip()==".ac"):
                flaga+=1; freq=((line.split('#')[0]).strip("\n")).split(' ')[-1]
                if len(((line.split('#')[0]).strip()).split(' '))>1:
                    secondlast=((line.split('#')[0]).strip()).split(' ')[-2]
                try:
                    if (freq=='0'):
                        w[2]+=1
                    freq=float(freq)
                    
                except ValueError:
                    w[0]+=1
                try:
                    secondlast=float(secondlast)
                    w[1]+=1
                except ValueError:
                    pass
        f.seek(0)
        if((flagc!=1) | (flage!=1)):
            print("\nError: There should be exactly one '.circuit' and one '.end' for the file to contain a valid circuit")
            f.close()
        if(flaga>1):
            print("\nError: Please enter single '.ac' line for the circuit") 
            f.close()   
        if(w[0]!=0):
            print("\nError: Please enter valid frequency for the circuit")
            f.close()
        if(w[1]!=0):
            print("\nError: Please enter single frequency for the circuit")         
            f.close()
        if(w[2]!=0):
            print("\nError: Please enter non-zero value of frequency for ac circuit, else use dc")
            f.close()
        elif(flagc==1 and flage==1 and flaga<=1 and w==[0,0,0]):
            f=open(sys.argv[1])
            ckt=[line for count,line in enumerate(f) if (count>c and count<e)]
            f.close()
            for j in range(len(ckt)):
                if ("#" in ckt[j]):
                    ckt[j]=ckt[j][:ckt[j].index("#")].strip()+"\n"    # [Removes the comments in the circuit]
                ckt[j]=(ckt[j].replace("\n"," \n")).split(' ')    # [Tokens in each line are split up]
            err=0 
            for m in ckt:
                if (m[0][0]=='V' or m[0][0]=='I'):
                    if ((m[-3]=='dc') and (freq==0.0)) or ((m[-4]=='ac') and (freq!=0.0)):
                        pass
                    else:
                        err+=1
            
###############################################################################
            if err==0:  
                class resistor:
                  def __init__(self,rname,rnodes,rvalue):
                      self.name=rname
                      self.nodes=rnodes
                      self.value=rvalue
                class capacitor:
                  def __init__(self,cname,cnodes,cvalue):
                      self.name=cname
                      self.nodes=cnodes
                      self.value=cvalue
                class inductor:
                  def __init__(self,lname,lnodes,lvalue):
                      self.name=lname
                      self.nodes=lnodes
                      self.value=lvalue
                if freq==0.0:
                  class vsrc:
                      def __init__(self,vname,vnodes,vtype,vvalue):
                          self.name=vname
                          self.nodes=vnodes
                          self.value=vvalue
                          self.typ=vtype
                  class isrc:
                      def __init__(self,iname,inodes,itype,ivalue):
                          self.name=iname
                          self.nodes=inodes
                          self.value=ivalue
                          self.typ=itype
                else:
                  class vsrc:
                      def __init__(self,vname,vnodes,vtype,vvalue,vphase):
                          self.name=vname
                          self.nodes=vnodes
                          self.value=str(complex((float(vvalue)/2)*cos(float(vphase)), (float(vvalue)/2)*sin(float(vphase))))
                          self.typ=vtype
                  class isrc:
                      def __init__(self,iname,inodes,itype,ivalue,iphase):
                          self.name=iname
                          self.nodes=inodes
                          self.value=str(complex((float(ivalue)/2)*cos(float(iphase)), (float(ivalue)/2)*sin(float(iphase))))
                          self.typ=itype
                nds={}
                nds['GND']='0'
                q=1
                
#Creation of Objects for Components:
                if freq==0.0:
                    v=[vsrc(m[0],[m[1],m[2]],m[3],m[4]) for m in ckt if 'V' in m[0][0]]
                    p=len(v)
                    i=[isrc(m[0],[m[1],m[2]],m[3],m[4]) for m in ckt if 'I' in m[0][0]]
                    c=[]
                    l=[inductor('V'+str(len(v)+1),[m[1],m[2]],'0') for m in ckt if 'L' in m[0][0]]
                    v.extend(l)
                else:
                    v=[vsrc(m[0],[m[1],m[2]],m[3],m[4],m[5]) for m in ckt if 'V' in m[0][0]]
                    i=[isrc(m[0],[m[1],m[2]],m[3],m[4],m[5]) for m in ckt if 'I' in m[0][0]]
                    c=[capacitor(m[0],[m[1],m[2]],m[3]) for m in ckt if 'C' in m[0][0]]
                    l=[inductor(m[0],[m[1],m[2]],m[3]) for m in ckt if 'L' in m[0][0]]    
                r=[resistor(m[0],[m[1],m[2]],m[3]) for m in ckt if 'R' in m[0][0]]
#Creation of Distinct Node Table:
                for m in ckt:
                   if freq==0.0:
                    y=3 if ((m[0][0]=='V' and m[0]!='Vx') or m[0][0]=='I') else 2
                   else:
                    y=4 if ((m[0][0]=='V' and m[0]!='Vx') or m[0][0]=='I') else 2
                   for j in range(1, len(m)-y):
                       if m[j] not in nds.keys():
                           nds[m[j]]=str(q)
                           q+=1
#Building of the Matrix:
                M=array([zeros(len(nds)+len(v)) for m in range (len(nds)+len(v))], dtype=complex)
                B=array(zeros(len(nds)+len(v)), dtype=complex)
                M[0][0]=1+0j
                B[0]=0+0j
                for key,val in nds.items():
                    if(key=='GND'):
                        continue
                    else:
                      if freq==0.0:
                        vol=[m for m in v if key in m.nodes]
                        ind=[]
                      else:
                        vol=[m for m in v if key in m.nodes]
                        ind=[m for m in l if key in m.nodes]
                    res=[m for m in r if key in m.nodes]
                    cur=[m for m in i if key in m.nodes]
                    cap=[m for m in c if key in m.nodes]
                    if len(vol)>0:
                        for m in vol:
                            w=int(m.name[1])-1
                            V=complex(m.value)
                            if (m.nodes[0]!=key):
                                M[int(val)][len(nds)+w]=M[int(val)][len(nds)+w]-1
                            else:
                                M[int(val)][len(nds)+w]=M[int(val)][len(nds)+w]+1
                            M[len(nds)+w][int(nds[m.nodes[0]])]=1
                            M[len(nds)+w][int(nds[m.nodes[1]])]=-1
                            B[len(nds)+w]=V
                    if len(res)>0:
                        for m in res:  
                            Z=complex(m.value)
                            if (m.nodes[0]==key):
                                Z=-Z
                            M[int(val)][int(nds[m.nodes[0]])]=M[int(val)][int(nds[m.nodes[0]])]+1/Z
                            M[int(val)][int(nds[m.nodes[1]])]=M[int(val)][int(nds[m.nodes[1]])]-1/Z
                    if len(cur)>0:    
                        for m in cur:
                            I=complex(m.value)
                            if (m.nodes[0]!=key):
                                I=-I
                            B[int(val)]=I
                    if len(cap)>0:
                        for m in cap: 
                            va=float(m.value)
                            Z=1/complex(0,2.0*pi*float(va)*float(freq))
                            if (m.nodes[0]==key):
                                Z=-Z
                            M[int(val)][int(nds[m.nodes[0]])]=M[int(val)][int(nds[m.nodes[0]])]+1/Z
                            M[int(val)][int(nds[m.nodes[1]])]=M[int(val)][int(nds[m.nodes[1]])]-1/Z
                    if len(ind)>0:
                        for m in ind:  
                            va=float(m.value)
                            Z=complex(0,2.0*pi*float(va)*float(freq))
                            if (m.nodes[0]==key):
                                Z=-Z
                            M[int(val)][int(nds[m.nodes[0]])]=M[int(val)][int(nds[m.nodes[0]])]+1/Z
                            M[int(val)][int(nds[m.nodes[1]])]=M[int(val)][int(nds[m.nodes[1]])]-1/Z
                    vol=[]
                    res=[]
                    cur=[]
                    cap=[]
                    ind=[]
                 
#Solving Matrix Equation:          
                try:        
                    x=linalg.solve(M,B)
                    print("# Currents through voltage sources are positive when flowing from negative to positive terminal of voltage source")
                    for key,val in nds.items():
                        print("V at "+ key+" = "+str(complex(truncate(real(x[int(val)])),truncate(imag(x[int(val)])))))
                    if freq==0.0:
                        for m in range(len(nds),len(nds)+len(v)-len(l)):
                            print("I through voltage source V"+str(m+1-len(nds))+" = "+str(complex(truncate(real(x[m])),truncate(imag(x[m])))))
                    else:
                        for m in range(len(nds),len(nds)+len(v)):
                            print("I through voltage source V"+str(m+1-len(nds))+" = "+str(complex(truncate(real(x[m])),truncate(imag(x[m])))))
                except Exception:
                    print("Circuit cannot be solved")
            else:
                print("\nError: The type of circuit and the type of source are mismatched")      
    



            
            
        
            
            
        

    





        
