import atpy #Needs >0.9.5 due to bug in comment handling
from extinction_distance.completeness import sextractor
import numpy as np
import extinction_distance.support.coord as coord #This is an old slow version, but has no compat problems
import os
import math

def calibrate(source,filtername):

    flag_limit = 2
    ukidss_filter = filtername+"AperMag3"
    ukidss_err = filtername+"AperMag3Err"

    sex = sextractor.SExtractor()
    my_catalog = sex.catalog()
    #print(my_catalog)
    alpha = []
    delta = []
    for star in my_catalog:
        alpha.append(star['ALPHA_J2000'])
        delta.append(star['DELTA_J2000'])

    t = atpy.Table(os.path.join(source+"_data",source+"_UKIDSS_cat.fits"),type='fits')

    blah = coord.match(np.array(alpha),np.array(delta),t['RA'],
                        t['Dec'],1.,seps=False)

    #print(source)
    #print(filtername)
    #print(my_catalog)
    my_mag = []
    ukidss_mag = []
    errs = []
    for i,j in enumerate(blah):
        if j != -1:
            #print(i)
            #print(j)
            #print(t[int(j)][ukidss_filter])
            if ((my_catalog[i]['FLAGS'] < flag_limit) and (t[int(j)][ukidss_filter] < 20) and (t[int(j)][ukidss_filter] > 0)):
                my_mag.append(my_catalog[i]['MAG_APER'])
                ukidss_mag.append(t[int(j)][ukidss_filter])
                errs.append(np.sqrt(t[int(j)][ukidss_err]**2+my_catalog[i]['MAGERR_APER']**2))

    a = np.average(np.array(my_mag)-np.array(ukidss_mag),weights = np.array(errs))
    b = np.median(np.array(my_mag)-np.array(ukidss_mag))
    poss_zp = np.array([a,b])
    ii = np.argmin(np.absolute(poss_zp))
    zp = poss_zp[ii]
    #zp = np.min(a,b)
    print(source)
    print(filtername)
    print("ZP: "+str(round(zp,3)))
    return(zp)
