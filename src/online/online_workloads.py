import src.hdmm.workload as workload
import src.census_workloads as census

def identity():
    return workload.Identity(64).dense_matrix()

def total():
    return workload.Total(64).dense_matrix()

def prefix_sum():
    print("prefix_sum")
    
def H2():
    return workload.H2(64).dense_matrix()
    
def custom():
    print("custom")
    
def race1():
    return census.__race1().dense_matrix()
    
def race2():
    return census.__race2().dense_matrix()
    
def race3():
    return census.__white().dense_matrix()