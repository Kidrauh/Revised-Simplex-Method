import numpy as np
import copy

def initialize_pa():
    c = np.loadtxt("c.csv",delimiter=",")
    A = np.loadtxt("A.csv",delimiter=",")
    b = np.loadtxt("b.csv",delimiter=",")
    return c,A,b

# generate a bfs
def phase1(A,b,m,n):
    Precise = 1e-10
    Iter=0
    basic_x_index=np.zeros(m,dtype=int)
    # 添加人工变量后的基矩阵是新增的x
    for i in range(m):
        basic_x_index[i]=i+n

    # generate I matrix to form new A
    new=np.zeros((m,m))
    for i in range(m):
        new[i][i]=1
    A=np.hstack((A,new))
    # calculate reduced cost
    r=np.zeros(A.shape[1])
    r[n:]=np.ones(m)
    # print("A:{}".format(A))
    # print("r:{}".format(r))
    # print("--------------------------")
    # f
    f=0
    # convert it to standard form with r=0 for basic variables
    for i in range(m):
        r=r-A[i,:]
        f=f-b[i]

    while(1):
        # print("A:{}".format(A))
        # print("r:{}".format(r))
        # print("--------------------------")
        assert Iter<=5000,"Too many cycles"
        # print("r:{}".format(r))
        if min(r)>=0 or abs(min(r)) <= Precise:
            print("We have found the bfs in phase1")
            break;
        Iter=Iter+1

        "select the smallest r"
        enter=np.argmin(r)
        assert max(A[:,enter])>0, "No boundry"

        "select which x leave the basis"
        min_=[]
        for i in range(m):
            if A[i,enter]>0:
                min_.append((b[i]/A[i,enter],i))
        min_.sort(key=lambda x:x[0])
        leave=min_[0][1]

        "update"
        b[leave]=b[leave]/A[leave,enter]
        A[leave,:]=A[leave,:]/A[leave,enter]
        for rows in range(m):
            if rows!= leave:
                b[rows]-=A[rows,enter]*b[leave]
                A[rows,:]-=A[rows,enter]*A[leave,:]
        basic_x_index[leave]=enter
        # print("r is {}".format(r))
        # print("b is {}".format(b))
        # print("r[enter] is {}".format(r[enter]))
        # print("b[leave] is {}".format(b[leave]))
        f=f-r[enter]*b[leave]
        # print("f is {}".format(f))
        r=r-r[enter]*A[leave,:]
    assert abs(f)<Precise, 'No feasible solution'

    removed_row=[]
    for i in range(m):
        # 基变量中有人工变量,驱赶人工变量出基
        if basic_x_index[i]>=n:
            # 当一行全为0
            flag=0
            for j in range(n):
                if A[i,j]==0:
                    flag+=1
            if flag==n:
                # delete the redundant line
                A=np.delete(A,i,0)
                b=np.delete(b,i)
                removed_row.append(i)
            else:
                # 选择任意一个非零元为pivot
                for j in range(n):
                    if A[i,j]!=0:
                        b[i] = b[i] / A[i, j]
                        A[i,:] = A[i,:] / A[i, j]
                        for rows in range(m):
                            if rows != i:
                                b[rows] -= A[rows,j] * b[i]
                                A[rows,:] -= A[rows,j] * A[i,:]
                        basic_x_index[i]=j
                        f=f-r[j]*b[i]
                        r=r-r[j]*A[i,:]
                        break;
    # print("A:{}".format(A))
    # print("r:{}".format(r))
    # print("--------------------------")
    new_basic=[]
    # print("removed:{}".format(removed_row))
    for i in range(len(basic_x_index)):
        if i not in removed_row:
            new_basic.append(basic_x_index[i])
    # print("new_basic2:{}".format(new_basic))
    new_A=A[:,:n]
    B_inverse=np.linalg.inv(A[:,new_basic])
    # l=len(A[:,new_basic])
    # B_inverse=np.identity(l)
    # print(B_inverse)
    return new_A,B_inverse,b,new_basic,Iter

def revised_simplex(A,b,c,B_inverse,new_basic,n):
    # print("new_basic:{}".format(new_basic))
    Precise = 1e-10
    "calculate b_head"
    b_head=np.matmul(B_inverse,b)
    "calculate cn"
    Cn=[]
    for i in range(n):
        if i not in new_basic:
            Cn.append(c[i])
    Cn=np.array(Cn)
    "calculate non_basis"
    non_basis=[]
    for i in range(n):
        if i not in new_basic:
            non_basis.append(i)
    "calculate cb"
    Cb=[]
    for i in range(len(new_basic)):
        Cb.append(c[new_basic[i]])
    Cb=np.array(Cb)
    

    "calculate N"
    N=np.zeros((A.shape[0],n-len(new_basic)))
    for i in range(n-len(new_basic)):
        N[:,i]= copy.deepcopy(A[:,non_basis[i]])

    "calculate lambda"
    lambd=np.matmul(Cb,B_inverse)

    "Begin"
    Iter=0
    while(1):
        r_N=Cn-np.matmul(lambd,N)
        if min(r_N)>=0 or abs(min(r_N)) <= Precise:
            break
        "choose the minimal to go into the basis"
        min_r=r_N[0]
        "the minimal index"
        rq=0
        for i in range(len(r_N)):
            if r_N[i]<min_r:
                min_r=r_N[i]
                rq=i
        rq=non_basis[rq]
        y_q=np.matmul(B_inverse,A[:,rq])
        "no boundry"
        assert max(y_q)>0,'No boundry'

        list=[]
        for i in range(len(b_head)):
            if y_q[i]>0:
                list.append((b_head[i]/y_q[i],i))
        "choose p to go out of the basis"
        list.sort(key=lambda x:x[0])
        p=list[0][1]

        "update iter"
        Iter+=1

        "update anything else"
        non_basis.remove(rq)
        non_basis.append(new_basic[p])
        new_basic[p]=rq
        non_basis.sort()

        "update lambda"
        lambd+=min_r/y_q[p]* B_inverse[p,:]

        "update the inverse of B"
        new_inv=np.zeros((B_inverse.shape[0], B_inverse.shape[1]))
        for i in range(B_inverse.shape[0]):
            new_inv[i][i]=1
        for i in range(B_inverse.shape[0]):
            for j in range(B_inverse.shape[0]):
                if i!=p:
                    new_inv[i,j]=B_inverse[i,j]-y_q[i]/y_q[p]*B_inverse[p,j]
                else:
                    new_inv[i,j]=B_inverse[p,j]/y_q[p]
        B_inverse=new_inv
        

        "update Cb"
        Cb=[]
        for i in range(n):
            if i in new_basic:
                Cb.append(c[i])
        Cb=np.array(Cb)

        # "update lambda"
        # lambd=np.matmul(Cb,B_inverse)
        # print("lambda:{}".format(lambd))

        "update B-head"
        b_head=np.matmul(B_inverse,b)

        "update Cn"
        Cn=[]
        for i in range(len(non_basis)):
            Cn.append(c[non_basis[i]])
        Cn=np.array(Cn)

        "update N"
        N=np.zeros((A.shape[0],n-len(new_basic)))
        for i in range(n-len(new_basic)):
            N[:,i]= copy.deepcopy(A[:,non_basis[i]])
        
        "check iter times"
        assert Iter<=5000,"Too many iterations"
        
    
    final_x=np.zeros(n)
    
    for i in range(len(new_basic)):
        final_x[new_basic[i]]=b_head[i]
    
    final_result=np.matmul(final_x,c)
    return final_x,final_result,Iter


if __name__ == "__main__":
    c,A,b=initialize_pa()
    n=len(c) # number of variables
    m=len(b) # number of constraints

    # convert it to standard form (make b>0)
    for i in range(m):
        if b[i] < 0:
            b[i] = np.negative(b[i])
            A[i,:] = np.negative(A[i,:])
    
    # start phase1
    A_new, B_inverse, b_new, basic,Iter_ = phase1(A,b,m,n)

    "start revised simplex"
    final_x,final_result,Iter=revised_simplex(A_new,b_new,c,B_inverse,basic,n)
    Iter+=Iter_
    print("fun:{}".format(final_result))
    print("Optimization terminated successfully")
    print("nit:{}".format(Iter))
    print("success: True")
    print("x: array{}".format(final_x))
    




