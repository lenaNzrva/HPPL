
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from numpy import exp
from numpy import linspace
from numpy import fft

# Initialization for parallelization
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Initialization for Spectrogram
n = 1000

t = linspace(-20*2*pi, 20*2*pi, n)
y = np.sin(t)*exp(-t**2/2/20**2)
y = y+np.sin(3*t)*exp(-(t-5*2*pi)**2/2/20**2)
y = y+np.sin(5.5*t)*exp(-(t-10*2*pi)**2/2/5**2)
y = y+np.sin(4*t)*exp(-(t-7*2*pi)**2/2/5**2)
y = np.array(y)

Res = None


distr = n//(size)
list_of_distr = list(range(0, n+1, distr))
if list_of_distr[-1] != n:
    list_of_distr[-1] = n
    
list_to_send = []
for i in range(len(list_of_distr)-1):
    list_to_send.append([list_of_distr[i],list_of_distr[i+1]])
    
list_to_send = list_to_send[rank]

windows = linspace(-20*2*pi,20*2*pi,n)

# specgram = np.zeros((n,list_to_send[1]-list_to_send[0]))
specgram = []
for i in range(list_to_send[0], list_to_send[1]):
    window_function = exp(-(t-windows[i])**2/2/(2.0*2*pi)**2)
    y_window = y * window_function
    y_windowed = np.array(y_window)
    specgram.append(abs(fft.fft(y_windowed)))

specgram = np.array(specgram)
comm.Barrier()
# comm.Gather(specgram, Res, root=0)
Res = comm.gather(specgram, root=0)

if rank == 0:
    Res = np.array(np.concatenate(Res).ravel())
    Res = Res.reshape((n, n), order='F')
#     print(Res.shape)
    
#     t = np.linspace(-20*2*pi, 20*2*pi, n)
#     w = fft.fftfreq(len(y), d=(t[1]-t[0])/2/pi)

#     plt.figure(figsize=(10,7))
#     plt.imshow(Res, aspect='auto', extent=[-20, 20, y[0], 2 * w[int(len(t)/2)-1]], cmap='plasma')
#     plt.ylim(0,10)
#     plt.title("Spectogram")
#     plt.xlabel('T,cycles')
#     plt.ylabel('Frequency')
#     plt.colorbar()
#     plt.show()
