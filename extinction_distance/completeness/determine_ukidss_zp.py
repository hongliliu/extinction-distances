import atpy #Needs >0.9.5 due to bug in comment handling
import sextractor
import numpy as np
import coord #This is an old slow version, but has no compat problems
import os
import math

def calibrate(source,filtername):

    flag_limit = 2
    ukidss_filter = filtername+"APERMAG3"
    ukidss_err = filtername+"APERMAG3ERR"

    sex = sextractor.SExtractor()
    my_catalog = sex.catalog()
    alpha = []
    delta = []
    for star in my_catalog:
        alpha.append(star['ALPHA_J2000'])
        delta.append(star['DELTA_J2000'])

    t = atpy.Table(os.path.join(source+"_data",source+"_UKIDSS_cat_trim.fits"),type='fits')

    blah = coord.match(np.array(alpha),np.array(delta),t['RA']*180./math.pi,
                        t['DEC']*180./math.pi,1.,seps=False)

    my_mag = []
    ukidss_mag = []
    errs = []
    for i,j in enumerate(blah):
        if j != -1:
            if ((my_catalog[i]['FLAGS'] < flag_limit) and (t[int(j)][ukidss_filter] < 20) and (t[int(j)][ukidss_filter] > 0)):
                my_mag.append(my_catalog[i]['MAG_APER'])
                ukidss_mag.append(t[int(j)][ukidss_filter])
                errs.append(np.sqrt(t[int(j)][ukidss_err]**2+my_catalog[i]['MAGERR_APER']**2))


    a = np.average(np.array(my_mag)-np.array(ukidss_mag),weights = np.array(errs))
    print(source)
    print(filtername)
    print(a)
    return(a)
