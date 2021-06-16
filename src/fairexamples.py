from hdmm import workload, fairtemplates, error, fairmechanism, matrix, mechanism, templates
from census_workloads import SF1_Persons
import numpy as np


def example1():
    """ Messing around area """
    print('Example 1')
    W1 = workload.AllRange(4)
    W2 = workload.Prefix(4)

    print(W1.matrix)
    print(W2.matrix)


    W = [W1,W2]
    #takes two parameters p and dimesnions
    pid = fairtemplates.PIdentity(2, 4)
    res = pid.optimize(W)

    #err = error.rootmse(W[0], pid.strategy())
    #err2 = error.rootmse(W[0], workload.Identity(8))
    #print(err, err2)
    #err = error.rootmse(W[1], pid.strategy())
    #err2 = error.rootmse(W[1], workload.Identity(8))
    #print(err, err2)


def example2():

    print('Example 2')
    W1 = workload.AllRange(256)
    W2 = workload.IdentityTotal(256)
    W = [W1,W2]
   
    M = fairmechanism.HDMM(W, np.zeros(256), 1.0)
    M.optimize(restarts=5)
    xest = M.run()

    print(np.sum((xest - 0)**2))


def experiment1():
    print('Experiment1')
    n = 256
    """ We want an example to show that naively running HDMM on the
    entire workload ignoring identities of individual analysts does not
    satisfy sharing incentive. Intuitively, this should happen when one
    analyst has a much smaller/easier workload than the other analysts,
    such that their errors dominate the optimization """
    W1 = workload.AllRange(n)
    W2 = workload.Total(n)
    #W2 = workload.IdentityTotal(n)
    W = workload.VStack([W1,W2])
   
   #workload 1 with half the budget
    pid = templates.PIdentity(max(1, n//16), n)
    pid.optimize(W1)
    print("Workload 1 with half the budget" )
    err1 = error.expected_error(W1, pid.strategy(),eps= 0.5)
    print(err1)

    pid = templates.PIdentity(max(1, n//16), n)
    pid.optimize(W2)
    print("Workload 2 with half the budget" )
    err2 = error.expected_error(W2, pid.strategy(),eps= 0.5)
    print(err2)


    #both workloads together
    pid = templates.PIdentity(max(1, n//16), n)
    pid.optimize(W)
    print("Both workloads with all the budgets" )
    err1all = error.expected_error(W1, pid.strategy(),eps= 1)
    err2all = error.expected_error(W2, pid.strategy(),eps= 1)
    print("W1")
    print(err1all)
    print("W2")
    print(err2all)
    print("Are either of the analysts violating sharing incentive")
    print((err1all >= err1) or (err2all >= err2) )

def experiment2():
    print("experiment2")
    n = 256
    """ We want to show that fixing this problem by partitioning the 
    privacy budget and running each workload independently can make 
    all of the agents worse off in terms of error (should be easy to
     see when the analysts have similar workloads)."""
    W1 = workload.Total(n)
    W1 = np.multiply(W1,1.1)
    W2 = workload.Total(n)
    W = workload.VStack([W1,W2])
   
   #workload 1 with half the budget
    pid = templates.PIdentity(max(1, n//16), n)
    pid.optimize(W1)
    print("Workload 1 with half the budget" )
    err1 = error.expected_error(W1, pid.strategy(),eps= 0.5)
    print(err1)

    pid = templates.PIdentity(max(1, n//16), n)
    pid.optimize(W2)
    print("Workload 2 with half the budget" )
    err2 = error.expected_error(W2, pid.strategy(),eps= 0.5)
    print(err2)


    #both workloads together
    pid = templates.PIdentity(max(1, n//16), n)
    pid.optimize(W)
    print("Both workloads with all the budgets" )
    err1all = error.expected_error(W1, pid.strategy(),eps= 1)
    err2all = error.expected_error(W2, pid.strategy(),eps= 1)
    print("W1")
    print(err1all)
    print("W2")
    print(err2all)
    print("Are both agents worse off by seperating their strategy")
    print((err1all < err1) and (err1all < err2))



def experiment3():
    print('experiment3')
    n =256
    """Want to show that we can satisfy sharing incentive by running
    the entire workload together, but weighting the workload of each 
    analyst in inverse proportion to the sensitivity of their workload """



    #using experiment 1 as an example
    W1 = workload.AllRange(n)
    #W2 = workload.Total(n)
    W2 = workload.Identity(n)
    W = workload.VStack([matrix.EkteloMatrix(np.multiply(W1.matrix,(1/W1.sensitivity()))),matrix.EkteloMatrix(np.multiply(W2.matrix,(1/W2.sensitivity())))])
   
   #workload 1 with half the budget
    pid = templates.PIdentity(max(1, n//16), n)
    pid.optimize(W1)
    print("Workload 1 with half the budget" )
    err1 = error.expected_error(W1, pid.strategy(),eps= 0.5)
    print(err1)

    pid = templates.PIdentity(max(1, n//16), n)
    pid.optimize(W2)
    print("Workload 2 with half the budget" )
    err2 = error.expected_error(W2, pid.strategy(),eps= 0.5)
    print(err2)


    #both workloads together
    pid = templates.PIdentity(max(1, n//16), n)
    pid.optimize(W)
    print("Both workloads with all the budgets" )
    err1all = error.expected_error(W1, pid.strategy(),eps= 1)
    err2all = error.expected_error(W2, pid.strategy(),eps= 1)
    print("W1")
    print(err1all)
    print("W2")
    print(err2all)
    print("Are either of the analysts violating sharing incentive")
    print((err1all >= err1) or (err2all >= err2) )

    """ Issue when you independently scale each matrix then merge you some of the
    quereis from the matrix with high sensitivity may be so low impact they become
    ignored. See all range after scaling for example"""

def experiment4():
    print('experiment4')
    n =256
    """Want to show that we can satisfy sharing incentive by running
    the entire workload together, but weighting the workload of each 
    analyst in inverse proportion to the sensitivity of their workload """

    """ This time we try scaling the big workload matrix"""


    #using experiment 1 as an example
    W1 = workload.AllRange(n)
    #W2 = workload.Total(n)
    W2 = workload.Identity(n)
    W = workload.VStack([W1,W2])
    W = matrix.EkteloMatrix((np.multiply(W.matrix,(1/W.sensitivity()))))
   
   #workload 1 with half the budget
    pid = templates.PIdentity(max(1, n//16), n)
    pid.optimize(W1)
    print("Workload 1 with half the budget" )
    err1 = error.expected_error(W1, pid.strategy(),eps= 0.5)
    print(err1)

    pid = templates.PIdentity(max(1, n//16), n)
    pid.optimize(W2)
    print("Workload 2 with half the budget" )
    err2 = error.expected_error(W2, pid.strategy(),eps= 0.5)
    print(err2)


    #both workloads together
    pid = templates.PIdentity(max(1, n//16), n)
    pid.optimize(W)
    print("Both workloads with all the budgets" )
    err1all = error.expected_error(W1, pid.strategy(),eps= 1)
    err2all = error.expected_error(W2, pid.strategy(),eps= 1)
    print("W1")
    print(err1all)
    print("W2")
    print(err2all)
    print("Are either of the analysts violating sharing incentive")
    print((err1all >= err1) or (err2all >= err2) )   

    """Doesn't work either. May work when the number of analysts scales too high.
    Much closer than the previous one though """

def experiment5():
    print('experiment5')
    n =256
    """Want to show that we can satisfy sharing incentive by running
    the entire workload together, but weighting the workload of each 
    analyst in inverse proportion to the sensitivity of their workload """

    """ This time we try scaling the big workload matrix"""


    #using experiment 1 as an example
    W1 = matrix.EkteloMatrix(np.random.rand(32,n))


    W2 = matrix.EkteloMatrix(np.random.rand(32,n))
    #W1 = workload.Prefix(n)

    #W2 = workload.Total(n)
    #W2 = workload.Identity(n)
    W = [W1,W2]
   
   #workload 1 with half the budget
    pid = templates.PIdentity(max(1, n//16), n)
    pid.optimize(W1)
    print("Workload 1 with half the budget" )
    err1 = error.expected_error(W1, pid.strategy(),eps= 0.5)
    print(err1)

    pid = templates.PIdentity(max(1, n//16), n)
    pid.optimize(W2)
    print("Workload 2 with half the budget" )
    err2 = error.expected_error(W2, pid.strategy(),eps= 0.5)
    print(err2)


    #both workloads together
    pid = fairtemplates.PIdentity(max(1, n//16), n)
    pid.optimize(W)
    print("Both workloads with all the budgets" )
    err1all = error.expected_error(W1, pid.strategy(),eps= 1)
    err2all = error.expected_error(W2, pid.strategy(),eps= 1)
    print("W1")
    print(err1all)
    print("W2")
    print(err2all)
    print("Are either of the analysts violating sharing incentive")
    print((err1all >= err1) or (err2all >= err2) )   

    """Optimizing on egalitarian doesn't work either. it seems to be spending most of 
    its time optimizing the more difficult queries. In fact it seems to be getting almost
    the exact same error should it have just optimized on the first one """

    """ note results change if it's just total vs identity for the second workload.
    I assume this means that the number of queries or significant difference in queries
    matters. I can test this more a bit in the future """

if __name__ == '__main__':
    #example1()
    #example2()
    #experiment1()
    #experiment2()
    #experiment3()
    #experiment4()
    experiment5()