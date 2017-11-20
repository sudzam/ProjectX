import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.axes as axes

def scatter_plot(x, xname, y, yname,plotname=None):
    #matplotlib.rcParams['axes.unicode_minus'] = False
    #fig,ax = plt.subplots()
    fig = plt.figure()
    ax  = fig.add_subplot(111)

    plot_name = plotname if(plotname != None) else '<noname>'
    plt.title(plot_name+' Plot')

    # for xy in zip(x, y):                                       # <--
    #     ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--

    #plt.ylim(0,100)
    #plt.xlim(0,max(x))
    #plt.grid(color='r', linestyle='-', linewidth=1)

    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.plot(x,y)
    plt.show()
