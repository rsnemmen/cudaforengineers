import pylab, numpy

# reads data file with results 
temp1d = numpy.loadtxt("output.dat",unpack=True)

# reshapes array from 1D to 2D
temp2d=temp1d.reshape((640,640))

# plots
pylab.clf()
pylab.imshow(temp2d)
pylab.colorbar()
pylab.show()