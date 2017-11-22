import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.axes as axes
f = 0
axarr = 0
init_hist=0

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
    plt.pause(0.5)

def hist_plot(data_arr, plot_label,hold=None,epoch=0):
    global init_hist
    global f, axarr

    if init_hist==0:
        plt.close()
        f, axarr = plt.subplots(len(data_arr), sharex=False)

    init_hist=1

    #plt.tight_layout(pad=0.2)#, w_pad=0.5, h_pad=0.5)

    for layer,data in enumerate(data_arr):
        axarr[layer].hist(data,bins=100, label='L:'+str(layer), range=(0,1))
        axarr[layer].set_title('layer:'+str(layer))

    #plt.ion()
    #
    if (hold): plt.pause(0.001)


def clear_plot():
    t=1#plt.clf()
