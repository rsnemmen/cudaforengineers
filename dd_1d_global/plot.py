import pylab, numpy

# reads data file with results 
x,u,dd,summed = numpy.loadtxt("results.csv",unpack=True,usecols=(0,1,2,3),delimiter=",")

# plots
pylab.clf()
pylab.plot(x,u,label="$u(x)$")
pylab.plot(x,dd,label="$d^2 u/dx^2$")
pylab.plot(x,summed,label="$u(x)+d^2 u/dx^2$")
pylab.legend()
pylab.xlabel("$x$")
pylab.show()