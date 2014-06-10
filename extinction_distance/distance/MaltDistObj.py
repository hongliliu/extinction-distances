#!/usr/bin/env python
# encoding: utf-8
"""
DistanceObject.py

An object to calculate extinction distances ala
Foster et al. (2012) Blue Number Count Method.

This is for Malt90 sources
and uses VVV and ATLASGAL
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
from matplotlib.path import Path

import matplotlib._cntr as _cntr

from extinction_distance.distance import determine_distance

#These are more complicated additions
#Sextractor and montage are required
import montage_wrapper as montage

from astroquery.vista import Vista
from astroquery.magpis import Magpis

class MaltDistObj(DistObj):
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

        self.besancon_area = 0.04*u.deg*u.deg #Area for model in sq. degrees. Large=less sampling error
        self.vista_directory = "" # XXX This needs to point to a way to save XXX
        self.vista_im_size = 6*u.arcmin #Size of VISTA cutout (symmetric) in arcminutes
        self.small_vista_im_size = 3*u.arcmin #Size of VISTA image for Sextractor
        
        self.contour_level = 0.1 #This is for BGPS
        
        self.jim = self.data_dir+self.name+"_VVV_J.fits"
        self.him = self.data_dir+self.name+"_VVV_H.fits"
        self.kim = self.data_dir+self.name+"_VVV_K.fits"
        self.vista_cat = self.data_dir+self.name+"_VVV_cat.fits"
        self.continuum = self.data_dir+self.name+"_AGAL.fits"
        self.rgbcube = self.data_dir+self.name+"_cube.fits"
        self.rgbcube2d = self.data_dir+self.name+"_cube_2d.fits"
        
        self.rgbpng = self.data_dir+self.name+".png"
        self.contour_check = self.data_dir+self.name+"_contours.png"

        self.model = self.data_dir+self.name+"_model.fits"
        self.completeness_filename = os.path.join(self.data_dir,self.name+"_completeness_VVV.pkl")
        
    def get_vista_images(self,clobber=False):
        """
        Get VISTA/VVV data/images for this region.
        This comes from astropy.astroquery

        Raw data is saved into self.data_dir as 
        self.name+"_VVV_J.fits"

        """
        #Get images
        if (not os.path.isfile(self.kim)) or clobber:
            print("Fetching VVV images from server...")
            for filtername,filename in zip(["J","H","K"],(self.jim,self.him,self.kim)):
                images = Vista.get_images(coordinates.Galactic(l=self.glon, b=self.glat, 
                                            unit=(u.deg, u.deg)),
                                            waveband=filtername,
                                            image_width=self.vista_im_size)
                #This makes a big assumption that the first VISTA image is the one we want
                fits.writeto(filename,
                             images[0][1].data,images[0][1].header,clobber=clobber)
        else:
            print("VISTA image already downloaded. Use clobber=True to fetch new versions.")
                         
    def get_vista_cat(self,clobber=False):
        """
        Get the VISTA catalog
        Catalog (necessary for zero-point determination) is saved
        into self.data_dir as
        self.name+"_VISTA_cat.fits"
        """
        if (not os.path.isfile(self.vista_cat)) or clobber:
            print("Fetching VISTA catalog from server...")
            
            table = Vista.query_region(coordinates.Galactic(l=self.glon,
                        b=self.glat,  unit=(u.deg, u.deg)), radius=self.vista_im_size)
            table.write(self.vista_cat,format="fits",overwrite=clobber)
            #Get catalog. We need this to establish zero-points/colours
        else:
            print("VISTA catalog already downloaded. Use clobber=True to fetch new versions.")
            
    def get_continuum(self,clobber=False):
        if (not os.path.isfile(self.continuum)) or clobber:
            print("Fetching ATLASGAL cutout from server...")
            image = Magpis.get_images(coordinates.Galactic(self.glon, self.glat,
                    unit=(u.deg,u.deg)), image_size=self.vista_im_size, survey='atlasgal')
            fits.writeto(self.continuum,
                         image[0].data,image[0].header,clobber=clobber)
        else:
            print("ATLASGAL image already downloaded. Use clobber=True to fetch new versions.")
            
    def make_photo_catalog(self,force_completeness=False):
        """
        Reads from photometry parameters
        Uses the trimmed image data
        """
        from extinction_distance.completeness import determine_completeness
        
        sex = determine_completeness.do_setup(self.name,survey="VISTA")
        k_corr = determine_completeness.do_phot(sex,self.name,survey="VISTA")
        if (force_completeness) or (not os.path.isfile(self.completeness_filename)):
            determine_completeness.do_completeness(sex,self.name,self.contours,survey="VISTA",k_corr=k_corr,numtrials = 100)
        self.catalog = atpy.Table(os.path.join(self.data_dir,self.name+"_MyCatalog_VISTA.vot"))
        self.catalog.describe()
    
    def do_distance_estimate(self):
        """
        Calculate the extinction distance
        based on the surface density of blue
        stars inside the contour and the
        besancon model.
        """
        self.load_data()
        blue_cut = 1.5 #Magic number for J-K cut. Put this elsewhere
        kup = 17 #More magic numbers
        klo = 11
        self.allblue = 0
        self.n_blue = 0
        poly = self.all_poly
        self.find_stars_in_contour(poly,"VISTA")
        blue,n_blue = self.count_blue_stars_in_contour(self.completeness,
                                        blue_cut=blue_cut,
                                        kupperlim = kup,
                                        klowerlim = klo,
                                        ph_qual = False,
                                        catalog=self.catalog,
                                        plot=True,
                                        survey="VISTA")
        print("Total area is...")
        print(self.total_poly_area)
        print("arcminutes")
        self.allblue+=blue
        self.n_blue +=n_blue
        self.model_data = self.load_besancon(self.model,write=False) #Read in the Besancon model
        percenterror = 4*math.sqrt(self.n_blue)/self.n_blue
        self.density = self.allblue/self.total_poly_area
        self.density_upperlim = (((self.allblue)/self.total_poly_area)
                                *(1+percenterror))
        self.density_lowerlim = (((self.allblue)/self.total_poly_area)
                                *(1-percenterror))
        print(self.density)
        print(self.density_upperlim)
        print(self.density_lowerlim)
        d,upp,low = determine_distance.do_besancon_estimate(self.model_data,
                kup,klo,blue_cut,self,
                        self.density_upperlim, #Why pass the object and
                        self.density_lowerlim, #the density?
                        self.density,survey="VISTA")
        print(d)
        distance_vista = d
        upper_vista = upp
        lower_vista = low
        print("==== Distance Results ====")
        print(distance_vista)
        print(lower_vista)
        print(upper_vista)
        print("==========================")
    
    
    def find_stars_in_contour(self,contour,survey):
        """
        From the photometry catalog we
        determine which stars are inside
        the contour of the cloud.
        """
        verts = np.array(contour,float)
        path = Path(verts)        
        if survey == "2MASS":
            points = np.column_stack((self.twomass.L,self.twomass.B))
            yo = path.contains_points(points)
            try:
                self.twomass.add_column("CloudMask",yo,description="If on cloud")
            except ValueError:
                self.twomass.CloudMask = self.twomass.CloudMask + yo
            #self.twomass.write("Modified_2MASS.vot")
        if survey == "VISTA":
            points = np.column_stack((self.catalog.L,self.catalog.B))
            yo = path.contains_points(points)
            try:
                self.catalog.add_column("CloudMask",yo,description="If on cloud")
            except ValueError:
                self.catalog.CloudMask = self.catalog.CloudMask + yo
    
    def count_blue_stars_in_contour(self,completeness,blue_cut=1.3,kupperlim = 15.,klowerlim = 12.,ph_qual = False,plot=False,catalog=None,survey=None):
        """
        Determine which of the stars inside
        the contour are blue. And calculate the
        areal density of such stars. 
        """

        print("Reached Count stage")

        print(completeness[...,0])

        f = interp1d(completeness[...,0],completeness[...,1],kind='cubic')

        good = catalog.where((catalog.KMag < kupperlim) & (catalog.KMag > klowerlim))
        in_contour = good.where(good.CloudMask == 1)
        JminK = in_contour.JMag - in_contour.KMag
        blue_in_contour = in_contour.where((JminK < blue_cut))
        blue_full = good.where(((good.JMag - good.KMag) < blue_cut))
        #Restore this
        #if plot:
        #    self.plotcolorhistogram(good.JMag-good.KMag,blue_full.JMag-blue_full.KMag,label="Full_Cloud",survey=survey)
        self.plot_color_histogram(in_contour.JMag-in_contour.KMag,
                                  blue_in_contour.JMag-blue_in_contour.KMag)

        compfactor = f(blue_in_contour.KMag)
        #print(compfactor)
        #print(blue_in_contour)
        #Hack. Really just wants np.ones()/compfactor
        blue_stars = (blue_in_contour.KMag)/(blue_in_contour.KMag)/compfactor
        n_blue = len(blue_in_contour.KMag)
        print("Number of blue stars in contour:")
        print(n_blue)
        print(sum(blue_stars))

        return(sum(blue_stars),n_blue)
        
    
    def plot_color_histogram(self,JminK,JminK_blue):
        """
        Make the three check-images
        1) 3-color image of region
        2) J-K histogram
        3) Distance estimate plot
        """
        plt.figure()
        plt.hist(JminK,color='gray',bins=np.arange(-0.2,3.2,0.1))
        plt.hist(JminK_blue,color='blue',bins=np.arange(-0.2,3.2,0.1))
        plt.xlabel("J-K [mag]")
        plt.ylabel("Number of Stars")
        plt.axvline(x=1.5,ls=":")
        plt.savefig(self.data_dir+self.name+"_hist.png")
    

if __name__ == '__main__':
    unittest.main()
