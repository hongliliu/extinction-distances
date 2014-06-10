#!/usr/bin/env python
# encoding: utf-8
"""
determine_distance.py

"""

import sys
import os
import numpy as np
import atpy
import copy
import pylab
from scipy.interpolate import interp1d
from collections import defaultdict

cloud_av = 14. #This is 72*0.2 -- i.e. a BGPS flux of 0.2 Jy/beam
cloud_ak = cloud_av*0.114 #Conversion to A_K, check this


def do_besancon_estimate(model_data,kupperlim,klowerlim,colorcut,cloud,upperdensity,lowerdensity,centraldensity,survey):
    """We assume that all model files are of the same size -- 0.04 sq degree"""
    blue_star_density_model = []
    diffuse_model = model_data
    cloud_distances = np.arange(0,8000,50) #Possible cloud distances in pc


    Mags_per_kpc = 0.7

    for cloud_distance in cloud_distances:
        temp_model = copy.deepcopy(diffuse_model)
        #temp_model = read_besancon(model_file)

        foreground = temp_model.where((temp_model['Dist'] <= cloud_distance/1000.))

        try:
            foreground.add_column('corrj',foreground['J-K']+foreground['K'] + Mags_per_kpc*0.276*foreground['Dist'])
            foreground.add_column('corrk',foreground['K'] + Mags_per_kpc*0.114*foreground['Dist'])
        except ValueError:
            #Some model files don't have K, but do have V and V-K
            foreground.add_column('corrj',foreground['J-K']+(foreground['V']-foreground['V-K']) + Mags_per_kpc*0.276*foreground['Dist'])
            foreground.add_column('corrk',(foreground['V']-foreground['V-K']) + Mags_per_kpc*0.114*foreground['Dist'])

        J_min_K = foreground[(foreground['corrk'] < kupperlim) & (foreground['corrk'] > klowerlim) & (foreground['corrj']-foreground['corrk'] < colorcut)]
        #The 25/3600. takes us to per square arcmin
        blue_star_density_model.append(len(J_min_K)*(25/3600.)) #Old X.sum() notation DOES NOT WORK!!

    blah = smooth(np.array(blue_star_density_model),window_len=9,window='hanning')
    #pylab.figure(figsize=(4,4))
    pylab.plot(cloud_distances,blah,label="Besancon",color='k',ls='--')
    pylab.xlabel("Distance [pc]")
    pylab.ylabel("Number of Blue Stars/(sq. arcmin)")
#       print(centraldensity)
    #pylab.show()


    s = interp1d(cloud_distances,blue_star_density_model,kind=5)
    xx = np.linspace(0,7900,num=7900)
    yy = s(xx)

    lower = np.where(yy < lowerdensity)
    upper = np.where(yy > upperdensity)
    try:
        upperlim = upper[0][0]
    except IndexError:
        upperlim = 10.
    try:
        lowerlim = lower[0][-1]
    except IndexError:
        lowerlim = 0.

    center1 = np.where(yy <= centraldensity)
    center2 = np.where(yy > centraldensity)

    central = center1[0][-1]

    pylab.axhline(y=centraldensity,linewidth=2,color='k')
    pylab.axhline(y=upperdensity,color='k',linestyle=':')
    pylab.axhline(y=lowerdensity,color='k',linestyle=':')

    pylab.axvline(x=lowerlim,color='k',linestyle=':')
    pylab.axvline(x=upperlim,color='k',linestyle=':')
    pylab.axvline(x=central,color='k',linewidth=2)
    pylab.figtext(0.15,0.8,cloud.name,ha="left",fontsize='large',backgroundcolor="white")
    upper = str(upperlim-central)
    lower = str(lowerlim-central)
    
    pylab.figtext(0.15,0.75,str(central)+r"$_{"+lower+r"}$"+r"$^{+"+upper+r"}$"+" pc",fontsize='large',backgroundcolor="white")
    pylab.figtext(0.15,0.70,r'Area = '+str(round(cloud.total_poly_area,2))+' arcmin$^2$',ha="left",fontsize='large',backgroundcolor="white")
    pylab.figtext(0.15,0.65,"Survey: "+survey,ha="left",fontsize='large',backgroundcolor="white")

    fig = pylab.gcf()
    fig.set_size_inches(6,6)
    Size = fig.get_size_inches()
    print("Size in Inches: "+str(Size))
    pylab.savefig(os.path.join(cloud.name+"_data",cloud.name+"_Distance_"+survey+'.pdf'))
    pylab.clf()

    print("Distance = "+str(central)+"+"+str(upperlim-central)+str(lowerlim-central))
    perr = upperlim-central
    merr = central-lowerlim
    #return(central,(perr+merr)/2)
    return(central,perr,merr)



def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]


if __name__ == '__main__':
    main()
