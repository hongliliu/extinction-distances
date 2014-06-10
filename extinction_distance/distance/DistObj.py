#!/usr/bin/env python
# encoding: utf-8
"""
DistanceObject.py

An object to calculate extinction distances ala
Foster et al. (2012) Blue Number Count Method.

We should be able to subclass this object to
change the datasets in use. Specifically
UKIDSS/2MASS/VVV
BGPS/HiGAL/ATLASGAL
Besancon/TriLegal

current_cloud = DistanceObject.DistanceObject(name,coords)

current_cloud.find_bgps
current_cloud.make_contour
current_cloud.generate_besancon
current_cloud.get_ukidss_data
    current_cloud.process_ukidss_data

    Actually these two steps are intimately connected. So 
    doing the catalog also has to do the completeness estimate
    current_cloud.make_photo_catalog()
current_cloud.do_distance_estimate

"""

#These are basic
import sys
import os
import unittest
import subprocess
import pickle
import math
import os.path


#These are necessary imports
import numpy as np
import atpy
import aplpy
from astropy import wcs
from astropy import coordinates
from astropy import units as u
import astropy.wcs as pywcs
from astropy.io import fits
import matplotlib.pyplot as plt #For contouring and display
from scipy.interpolate import interp1d
from collections import defaultdict

import matplotlib._cntr as _cntr


#These are my programs
#from extinction_distance.completeness import determine_ukidss_zp #does what it says
#from extinction_distance.completeness import determine_completeness
#from extinction_distance.completeness import sextractor
#from extinction_distance.support import coord
#from extinction_distance.distance import determine_distance

#These are more complicated additions
#Sextractor and montage are required
import montage_wrapper as montage
#import ds9

#Astropy related stuff
#from astropy import astroquery
#from astroquery import besancon
#import astropy.io.ascii as asciitable
#from astropy.io.ascii import besancon as besancon_reader
#from astroquery import ukidss
from astroquery.ukidss import Ukidss
from astroquery.magpis import Magpis

class DistObj():
    def __init__(self,name,coords):
        self.name = name
        self.glon,self.glat = coords
        self.glat = float(self.glat)
        self.glon = float(self.glon)

        #print(self.glat)
        self.gc = coordinates.GalacticCoordinates(l=self.glon, b=self.glat, unit=(u.degree, u.degree))
        #self.eq = self.gc.fk5

        self.data_dir = self.name+"_data/"
        try:
            os.makedirs(self.data_dir)
        except OSError:
            pass

        self.besancon_area = 0.4*u.deg*u.deg #Area for model in sq. degrees. Large=less sampling error
        self.ukidss_directory = "" # XXX This needs to point to a way to save XXX
        self.ukidss_im_size = 15*u.arcmin #Size of UKIDSS cutout (symmetric) in arcminutes
        self.small_ukidss_im_size = 0.15*u.deg #Size of UKIDSS image for Sextractor
        
        self.jim = self.data_dir+self.name+"_UKIDSS_J.fits"
        self.him = self.data_dir+self.name+"_UKIDSS_H.fits"
        self.kim = self.data_dir+self.name+"_UKIDSS_K.fits"
        self.ukidss_cat = self.data_dir+self.name+"_UKIDSS_cat.fits"
        self.bgps = self.data_dir+self.name+"_BGPS.fits"
        self.rgbcube = self.data_dir+self.name+"_cube.fits"
        self.rgbcube2d = self.data_dir+self.name+"_cube_2d.fits"
        
        self.rgbpng = self.data_dir+self.name+".png"
        self.contour_check = self.data_dir+self.name+"_contours.png"

        self.model = self.data_dir+self.name+"_model.dat"

        
    def get_ukidss_images(self):
        """
        Get UKIDSS data/images for this region.
        This comes from astropy.astroquery

        Raw data is saved into self.data_dir as 
        self.name+"_UKIDSS_J.fits"

        """
        #Get images
        for filtername,filename in zip(["J","H","K"],(self.jim,self.him,self.kim)):
            images = Ukidss.get_images(coordinates.Galactic(l=self.glon, b=self.glat, 
                                        unit=(u.deg, u.deg)),
                                        waveband=filtername,
                                        image_width=3*u.arcmin)
            print(images)
            #This makes a big assumption that the first UKIDSS image is the one we want
            fits.writeto(filename,
                         images[0][1].data,images[0][1].header)
                         
    def get_ukidss_cat(self):
        """
        Get the UKIDSS catalog
        Catalog (necessary for zero-point determination) is saved
        into self.data_dir as
        self.name+"_UKIDSS_cat.fits"
        """
        table = Ukidss.query_region(coordinates.Galactic(l=self.glon,
                    b=self.glat,  unit=(u.deg, u.deg)), radius=self.ukidss_im_size)
        table.write(self.data_dir+"/"+self.name+"_UKIDSS_cat.fits",format="fits")
        #Get catalog. We need this to establish zero-points/colours
        
    def get_bgps(self,clobber=False):
        if (not os.path.isfile(self.bgps)) or clobber:
            image = Magpis.get_images(coordinates.Galactic(self.glon, self.glat,
                    unit=(u.deg,u.deg)), image_size=self.ukidss_im_size, survey='bolocam')
            fits.writeto(self.bgps,
                         image[0].data,image[0].header,clobber=clobber)
    
    def get_model(self,clobber=False):
        from astroquery import besancon
        import astropy.io.ascii as asciitable
        
        if (not os.path.isfile(self.model)) or clobber:
            besancon_model = besancon.request_besancon('jonathan.b.foster@yale.edu',
                                            glon=self.glon,glat=self.glat,
                                            small_field=True,
                                            area = 0.04,
                                            retrieve_file=True)
        B = asciitable.read(besancon_model,Reader=besancon.reader,guess=False)
        print(B)
            
    def get_contours(self, fitsfile, av=10.):
        """
        Given a Bolocam FITS file, return the contours at a given flux level
        """
        
        hdulist = fits.open(fitsfile)

        header = hdulist[0].header
        img = hdulist[0].data
        #hdulist.close()

        # from Foster 2012
        av_to_jy = 6.77e22/9.4e20 # cm^-2 / Jy / (cm^-2 / AV) = AV/Jy
        #if header.get('BGPSVERS').strip()=='1.0':
        av_to_jy /= 1.5

        contour_level = 0.8 #10 av in Jy?
        #av / av_to_jy

        wcs = pywcs.WCS(header)
        #wcsgrid = wcs.wcs_pix2world( np.array(zip(np.arange(wcs.naxis1),np.arange(wcs.naxis2))), 0 ).T
        yy,xx = np.indices(img.shape)

        img[img!=img] = 0
        C = _cntr.Cntr(yy,xx,img)
        paths = [p for p in C.trace(contour_level) if p.ndim==2]

        wcs_paths = [wcs.wcs_pix2world(p,0) for p in paths]

        return wcs_paths[0]
        
    def show_contours_on_threecolor(self, color='c',clobber=False):
        """

        """
        if (not os.path.isfile(self.rgbcube)) or clobber:
            aplpy.make_rgb_cube([self.kim,self.him,self.jim],self.rgbcube)
            aplpy.make_rgb_image(self.rgbcube,self.rgbpng)
            f = aplpy.FITSFigure(self.rgbcube2d)
            f.show_rgb(self.rgbpng)
            f.show_contour(self.bgps,levels=[0.8],convention='calabretta',colors='white')
            f.save(self.contour_check)
            


    def contour_area(self,xy):
        """ 
            Calculates polygon area.
            x = xy[:,0], y = xy[:,1]
        """
        l = len(xy)
        s = 0.0
        # Python arrys are zero-based
        for i in range(l):
            j = (i+1)%l  # keep index in [0,l)
            s += (xy[j,0] - xy[i,0])*(xy[j,1] + xy[i,1])
        return -0.5*s

        
    
    def make_images(self):
        """
        Make the three check-images
        1) 3-color image of region
        2) J-K histogram
        3) Distance estimate plot
        """
        pass
    

if __name__ == '__main__':
    unittest.main()
