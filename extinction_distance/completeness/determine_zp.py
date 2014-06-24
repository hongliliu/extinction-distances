from extinction_distance.completeness import sextractor
import numpy as np
import extinction_distance.support.pyspherematch as pyspherematch #Better version

import os
import math
from astropy.table import Table


def calibrate(source,filtername,survey="UKIDSS"):

    flag_limit = 2

    sex = sextractor.SExtractor()
    my_catalog = sex.catalog()
    g = Table(my_catalog)
    #print(g)
    
    my_catalog = g[(g['FLAGS'] < flag_limit)]
    #print(my_catalog)
    alpha = []
    delta = []
    for star in my_catalog:
        alpha.append(star['ALPHA_J2000'])
        delta.append(star['DELTA_J2000'])

    if survey == "2MASS":
        t = Table.read(os.path.join(source+"_data",source+"_2MASS_cat.vot"),format='votable')
        
        #print(t)
        if filtername == "K_1":
            twomass_magname = "k_m"
            twomass_errname = "k_msigcom"

        if filtername == "H":
            twomass_magname = "h_m"
            twomass_errname = "h_msigcom"

        if filtername == "J":
            twomass_magname = "j_m"
            twomass_errname = "j_msigcom"
        t = t[(t[twomass_magname] < 20) & (t[twomass_magname] > 0)]
        idxs1, idxs2, ds = pyspherematch.spherematch(np.array(alpha),np.array(delta),np.array(t['ra']),
                        np.array(t['dec']),tol=2/3600.)
    else:
        #This might be broken now
        t = Table.read(os.path.join(source+"_data",source+"_"+survey+"_cat.fits"),type='fits')
        ukidss_filter = filtername+"AperMag3"
        ukidss_err = filtername+"AperMag3Err"
        idxs1, idxs2, ds = pyspherematch.spherematch(np.array(alpha),np.array(delta),t['RA'],
                        t['Dec'],tol=2/3600.)    
    
    my_mag = my_catalog[idxs1]['MAG_APER']
    ukidss_mag = t[idxs2][twomass_magname]
    errs = np.sqrt(t[idxs2][twomass_errname]**2+my_catalog[idxs1]['MAGERR_APER']**2)
    
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
