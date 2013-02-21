#!/usr/bin/env python
# encoding: utf-8
"""
determine_completeness_make_catalog.py

This script is designed to test the completeness of UKIDSS images.
I inject fake stars (perfect 2D gaussians) into the image and
try to recover them with the same parameters I use in Sextractor
to create a photometric catalog (yet to do this). I just use integer
magnitudes. The assumption is that the stellar density is uniform
enough over the image that a single correction factor can be applied.
For this it is important to place the test stars within the polygon
I consider to be the cloud.

In the output catalog I can interpolate these points and boost the
contribution of each star accordingly. Thus, my detected blue stars
would look like:
a = [0,0,1,1,1,0...]
I would correct each star based on magnitude
a = [0,0,1.2,1,2.5,0,...]
where fainter stars are given more weight. No stars are down-weighted
of course. Does not matter if I have zeros in there.

Number of stars = sum(a)

Does it matter if I use J or K? We are looking at blue stars, therefore
J - K is small, and J should normally be more sensitive. Thus it makes
sense to trim in K. I need to accurately parameterize how J & K fail
for bright stars. They do, at a fairly bright limit. It is easiest just
to ignore bright stars. Now that I can use more faint ones this won't
influence my numbers hardly at all.
Generally we cannot trust K < 11


Basic outline:
do_completeness measures and saves (as pickle) completeness information.
The number of trials here can/should be adjusted
Finally do_phot runs photometry to produce an actual catalog

"""

import sys
import os
import sextractor
import numpy as np
import pyfits
import math
import matplotlib.nxutils as nx
import astLib.astWCS
import astLib.astCoords
import coord #This is an old slow version, but has no compat problems
import atpy
import montage
import pickle
import pylab
import determine_ukidss_zp

flag_limit = 4

def main():
    #Setup
    survey = "UKIDSS"
    for source in sources:
        sex = do_setup(source,survey=survey)
        k_corr = do_phot(sex,source,survey=survey)
        do_completeness(sex,source,survey=survey,k_corr=k_corr,numtrials = 500)

def do_setup(source,survey="UKIDSS"):
    sex = sextractor.SExtractor()
    if survey == "UKIDSS":
        sex.config['GAIN'] = 4.5
        sex.config['SATUR_LEVEL'] = 40000.0
        sex.config['MAG_ZEROPOINT'] = 25 #sex.Kzptalt
        sex.config['PHOT_APERTURES'] = 5 #sex.Kzptalt

    sex.config['PIXEL_SCALE'] = 0
    sex.config['CHECKIMAGE_TYPE'] = 'NONE'

    sex.config['PARAMETERS_LIST'].append('MAG_APER(1)')
    sex.config['PARAMETERS_LIST'].append('MAGERR_APER(1)')
    sex.config['PARAMETERS_LIST'].append('ALPHA_J2000')
    sex.config['PARAMETERS_LIST'].append('DELTA_J2000')

    return(sex)


def do_phot(sex,source,survey="UKIDSS"):

    print("Doing H photometry...")

    sex.run(os.path.join(source+"_data",source+"_"+survey+"_trim_H.fits"))
    #print(os.path.join(source+"_data",source+"_"+survey+"_trim_H.fits"))
    #Hcatalog = sex.catalog("py-sextractor.cat")
    Hcatalog = sex.catalog()


    print("Doing K photometry...")

    sex.run(os.path.join(source+"_data",source+"_"+survey+"_trim_K.fits"))
    Kcatalog = sex.catalog()
    try:
        k_correct = determine_ukidss_zp.calibrate(source,"K_1")
    except IndexError:
        print("Failed to calibrate, assuming no correction")
        k_correct = 0

    if ((k_correct > 0.5) or (k_correct < -0.5)):
        print("*** Large k-correction ("+k_correct+")***")
        print("Completeness for "+source+" likely wrong")

    print("Doing J photometry...")
    sex.run(os.path.join(source+"_data",source+"_"+survey+"_trim_J.fits"))
    Jcatalog = sex.catalog()
    try:
        j_correct = determine_ukidss_zp.calibrate(source,"J")
    except IndexError:
        print("Failed to calibrate, assuming no correction")
        k_correct = 0


    Kalpha = []
    Kdelta = []
    #
    for star in Kcatalog:
        Kalpha.append(star['ALPHA_J2000'])
        Kdelta.append(star['DELTA_J2000'])
    #
    Jalpha = []
    Jdelta = []
    for star in Jcatalog:
        Jalpha.append(star['ALPHA_J2000'])
        Jdelta.append(star['DELTA_J2000'])
    #
    blah = coord.match(np.array(Kalpha),np.array(Kdelta),np.array(Jalpha),
                                            np.array(Jdelta),0.3,seps=False)
    ra = []
    dec = []
    Jmag = []
    Kmag = []
    L = []
    B = []

    for i,j in enumerate(blah):
        if j != -1:
            if ((Kcatalog[i]['FLAGS'] < flag_limit) and (Jcatalog[int(j)]['FLAGS'] < flag_limit)):
                ra.append(Kcatalog[i]['ALPHA_J2000'])
                dec.append(Kcatalog[i]['DELTA_J2000'])
                Kmag.append(Kcatalog[i]['MAG_APER']-k_correct)

                Jmag.append(Jcatalog[int(j)]['MAG_APER']-j_correct)
                lb = astLib.astCoords.convertCoords("J2000","GALACTIC",
                                Kcatalog[i]['ALPHA_J2000'],Kcatalog[i]['DELTA_J2000'],2000.)
                L.append(lb[0])
                B.append(lb[1])

    #print(Kmag)
    t = atpy.Table()
    t.add_column('RA',ra)
    t.add_column('Dec',dec)
    t.add_column('L',L)
    t.add_column('B',B)

    t.add_column('JMag',Jmag)
    t.add_column('KMag',Kmag)
    #t.describe()

    try:
        os.remove(os.path.join(source+"_data",source+"_MyCatalog_"+survey+".vot"))
    except:
        pass
    t.write(os.path.join(source+"_data",source+"_MyCatalog_"+survey+".vot"))

    catlist = [Kcatalog,Hcatalog,Jcatalog]
    filterlist = ["K","H","J"]
    for cat,filtername in zip(catlist,filterlist):
        ra = []
        dec = []
        mag = []
        magerr = []
        flags = []
        L = []
        B = []

        for star in cat:
            ra.append(star['ALPHA_J2000'])
            dec.append(star['DELTA_J2000'])
            mag.append(star['MAG_APER'])
            magerr.append(star['MAGERR_APER'])
            flags.append(star['FLAGS'])
        t = atpy.Table()

        #print(cat['ALPHA_J2000'])
        t.add_column('RA',ra)
        t.add_column('Dec',dec)
        t.add_column('Mag'+filtername,mag)
        t.add_column('MagErr'+filtername,magerr)
        t.add_column('Flags'+filtername,flags)
        #if filtername=="J":
        #       print(np.average(mag))
        try:
            os.remove(os.path.join(source+"_data",source+"_MyCatalog_"+survey+"_"+filtername+".vot"))
        except:
            pass
        t.write(os.path.join(source+"_data",source+"_MyCatalog_"+survey+"_"+filtername+".vot"))


    sex.clean(config=True,catalog=True,check=True)
    return(k_correct)



def do_completeness(sex,source,survey="UKIDSS",k_corr = 0,numtrials=50):
    if survey == "UKIDSS":
        mags = [11,12,13,14,15,16,17,18,19]
        percent = {11:[],12:[],13:[],14:[],15:[],16:[],17:[],18:[],19:[]}
        zp = 25+k_corr
    if survey == "2MASS":
        #2MASS should be fine down to ~5-6
        mags = [7,8,9,10,11,12,13,14,15]
        percent = {7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[]}
        zp = 20+k_corr

    recovery = np.zeros((numtrials,len(mags)))
    for c in range(numtrials):
        d,h = pyfits.getdata(os.path.join(source+"_data",source+"_"+survey+"_trim_K.fits"),header=True)
        WCS = astLib.astWCS.WCS(h,mode="pyfits")
        all_poly = parse_ds9_regions(os.path.join(source+"_data",source+".reg"))
        fake_stars = insert_fake_stars(d,h,mags,all_poly,WCS,sex,survey=survey,zp=zp)
        #Recover returns an array of [1,1,1,0,0,0,1,0,0,1]
        r = recover(fake_stars,sex)
        recovery[c:] = r
    sex.clean(config=True,catalog=True,check=True)

    #print(recovery)
    print(mags)
    #print(recovery.sum(axis=0)/numtrials)
    comp = recovery.sum(axis=0)/numtrials
    print(comp)
    f = open(os.path.join(source+"_data/",source+"_completeness_"+survey+".pkl"),'w')
    pickle.dump(comp,f)
    f.close()


def insert_fake_stars(d,h,mags,all_poly,WCS,sex,survey="UKIDSS",zp=25.):
    fake_stars = []
    xsize = h['NAXIS1']
    ysize = h['NAXIS2']
    #Insert fake star
    if survey== "UKIDSS":
        size = 5
    if survey == "2MASS":
        size = 5

    for mag in mags:
        flag_in = False
        while flag_in == False:
            for poly in all_poly:
                verts = np.array(poly,float)
                x = np.random.random_sample()*(xsize-size-6)+(size)
                y = np.random.random_sample()*(ysize-size-6)+(size)
                radec = np.array(WCS.pix2wcs(x,y))
                te = np.array(astLib.astCoords.convertCoords("J2000","GALACTIC",radec[0],radec[1],2000.))
                yo = nx.pnpoly(te[0],te[1],verts)
                if yo == 1:
                #print(te)
                #print("is in")
                    flag_in = True
        magnitude = mag
        #zp = sex.config['MAG_ZEROPOINT']
        #Now we pass in zp instead
        expfactor = (magnitude - zp)/(-2.5)
        counts = math.pow(10.,expfactor)
        g = gauss_kern(size,counts) #5 is rough guess for FWHM
    #       print d[y-size:y+size+1,x-size:x+size+1].size
    #       print g.size
        d[y-size:y+size+1,x-size:x+size+1] += g #Damn backward numpy arrays
        fake_stars.append((x,y,mag))
    pyfits.writeto("TestOuput.fits",d,h,clobber=True)
    return(fake_stars)

def insert_fake_star(d,h,mag):
    xsize = h['NAXIS1']
    ysize = h['NAXIS2']
    #Insert fake star
    size = 5
    x = np.random.random_sample()*(xsize-size+1)+(size+1)
    y = np.random.random_sample()*(ysize-size+1)+(size+1)
    #print(x,y)
    #x = 700
    #y = 690
    #print(x,y)
    magnitude = mag
    zp = 25.
    expfactor = (magnitude - zp)/(-2.5)
    counts = math.pow(10.,expfactor)
    print(counts)
    g = gauss_kern(size,counts) #5 is rough guess for FWHM
    d[y-size:y+size+1,x-size:x+size+1] += g #Damn backward numpy arrays

    pyfits.writeto("TestOuput.fits",d,h,clobber=True)
    return(x,y,magnitude)

def recover(properties,sex):
    sex.run("TestOuput.fits")
    catalog = sex.catalog()
    ptol = 1
    mtol = 1.5
    found = np.zeros(len(properties))
    for star in catalog:
        #print(star['X_IMAGE'],star['Y_IMAGE'],star['MAG_APER'])
        ## x and y are off by one. 0/1 index problem with Sextractor?
        for i,prop in enumerate(properties):
            x,y,magnitude = prop
            if ((abs(star['X_IMAGE']-(x+1)) < ptol) and (abs(star['Y_IMAGE']-(y+1)) < ptol)
                    and (abs(star['MAG_APER']-magnitude) < (mtol+star['MAGERR_APER'])) and
                    (star['FLAGS'] < flag_limit)):
            #       print(star['X_IMAGE'],star['Y_IMAGE'],star['MAG_APER'],star['MAGERR_APER'])
                found[i] = 1
    #if Found == 1:
    #       print("Found")
    #else:
    #       print("Not Found")
    #print(found)
    return(found)
#       catalog = sex.catalog()

def gauss_kern(size, flux, sizey=None):
    """ Returns a 2D gauss kernel array as a fake star """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    normfactor = g.sum()/flux
    return g / normfactor


def parse_ds9_regions(region_name):
    f = open(region_name,'r')
    current_poly = []
    all_poly = []
    exclude = False
    include = False
    for line in f:
        if line.startswith("polygon") and line.endswith("red\n"): #Red contours mark good/useful ones
            include = True
        #elif line.startswith("polygon") and line.endswith("cyan\n"): #Cyan contours mean exclude
        #       exclude = True
        if include:
            start = line.find('(')
            end = line.rfind(')')
            subline = line[start+1:end]
            coords = subline.split(',')
            glats = coords[::2]
            glons = coords[1::2]
            for glat,glon in zip(glats,glons):
                current_poly.append((glat,glon))
            all_poly.append(current_poly)
            current_poly = []
            include = False
            exclude = False
    #print(len(all_poly))
    f.close()
    return(all_poly)

if __name__ == '__main__':
    main()
