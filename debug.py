import pymp

dict = {}
with pymp.Parallel(num_threads=2) as p:
    for i in p.range(10):
        
        dict[i] = i**2
        
print(dict.values())
