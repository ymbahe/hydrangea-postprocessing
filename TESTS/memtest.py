import numpy as np
import os
import subprocess
import gc

from pdb import set_trace

def print_memory_usage():
    pid = os.getpid()

    ret = subprocess.check_output(['pmap', str(pid)])
    rets = str(ret)
    ind = rets.find("Total:")
    part = rets[ind:]
    colInd = part.find(":")
    kInd = part.find("K")
    mem = int(part[colInd+1 : kInd])

    print("PID={:d} -- using {:d} KB (={:.1f} MB, {:.1f} GB)." .format(pid, mem, mem/1024, mem/1024/1024))

class Simulation:
    def __init__(self):
        self.status = 'top'
        
    def set_holder(self, holder):
        self.holder = holder

class Snapshot:
    def __init__(self, sim, isnap):
        self.status = 'blong'
        self.dummy = []

        self.isnap = isnap
        self.sim = sim
        
class Particles:
    def __init__(self, snap):
        self.status = 'bling'
        self.snap = snap
        snap.particles = self
        

    def allocate(self, size):
        self.arr = np.random.random(size=int(size))

    def reduce(self):
        self.arr = self.arr[:int(2e8)]

    def do_really_important_stuff(self):
        pass

    
ii = -1

for isim in range(30):
    sim = Simulation()

    for isnap in range(1):

        gc.collect()
        print("Sim {:d}, snap {:d}" .format(isim, isnap))

        snap = Snapshot(sim, isnap)

        part = Particles(snap)
        part.allocate(5e8)

        print_memory_usage()
        part.reduce()

        print_memory_usage()

        part.do_really_important_stuff()
        
        #del part.arr
        

    del sim
