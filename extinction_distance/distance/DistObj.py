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

See get_distance.py for an end-to-end example
of how to get a distance for a cloud.

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

        self.besancon_area = 0.04*u.deg*u.deg #Area for model in sq. degrees. Large=less sampling error
        self.ukidss_directory = "" # XXX This needs to point to a way to save XXX
        self.ukidss_im_size = 6*u.arcmin #Size of UKIDSS cutout (symmetric) in arcminutes
        self.small_ukidss_im_size = 3*u.arcmin #Size of UKIDSS image for Sextractor
        
        self.contour_level = 0.1 #This is for BGPS
        
        self.jim = self.data_dir+self.name+"_UKIDSS_J.fits"
        self.him = self.data_dir+self.name+"_UKIDSS_H.fits"
        self.kim = self.data_dir+self.name+"_UKIDSS_K.fits"
        self.ukidss_cat = self.data_dir+self.name+"_UKIDSS_cat.fits"
        self.continuum = self.data_dir+self.name+"_BGPS.fits"
        self.rgbcube = self.data_dir+self.name+"_cube.fits"
        self.rgbcube2d = self.data_dir+self.name+"_cube_2d.fits"
        
        self.rgbpng = self.data_dir+self.name+".png"
        self.contour_check = self.data_dir+self.name+"_contours.png"

        self.model = self.data_dir+self.name+"_model.fits"
        self.completeness_filename = os.path.join(self.data_dir,self.name+"_completeness_UKIDSS.pkl")
        
    def get_ukidss_images(self,clobber=False):
        """
        Get UKIDSS data/images for this region.
        This comes from astropy.astroquery

        Raw data is saved into self.data_dir as 
        self.name+"_UKIDSS_J.fits"

        """
        #Get images
        if (not os.path.isfile(self.kim)) or clobber:
            print("Fetching UKIDSS images from server...")
            for filtername,filename in zip(["J","H","K"],(self.jim,self.him,self.kim)):
                images = Ukidss.get_images(coordinates.Galactic(l=self.glon, b=self.glat, 
                                            unit=(u.deg, u.deg)),
                                            waveband=filtername,
                                            image_width=self.ukidss_im_size)
                #This makes a big assumption that the first UKIDSS image is the one we want
                fits.writeto(filename,
                             images[0][1].data,images[0][1].header,clobber=clobber)
        else:
            print("UKIDSS image already downloaded. Use clobber=True to fetch new versions.")
                         
    def get_ukidss_cat(self,clobber=False):
        """
        Get the UKIDSS catalog
        Catalog (necessary for zero-point determination) is saved
        into self.data_dir as
        self.name+"_UKIDSS_cat.fits"
        """
        if (not os.path.isfile(self.ukidss_cat)) or clobber:
            print("Fetching UKIDSS catalog from server...")
            
            table = Ukidss.query_region(coordinates.Galactic(l=self.glon,
                        b=self.glat,  unit=(u.deg, u.deg)), radius=self.ukidss_im_size)
            table.write(self.ukidss_cat,format="fits",overwrite=clobber)
            #Get catalog. We need this to establish zero-points/colours
        else:
            print("UKIDSS catalog already downloaded. Use clobber=True to fetch new versions.")
            
    def get_continuum(self,clobber=False):
        if (not os.path.isfile(self.continuum)) or clobber:
            print("Fetching BGPS cutout from server...")
            image = Magpis.get_images(coordinates.Galactic(self.glon, self.glat,
                    unit=(u.deg,u.deg)), image_size=self.ukidss_im_size, survey='bolocam')
            fits.writeto(self.continuum,
                         image[0].data,image[0].header,clobber=clobber)
        else:
            print("BGPS image already downloaded. Use clobber=True to fetch new versions.")
            
    def get_model(self,clobber=False):
        """
        Get a Besancon model for this region of sky
        """
        
        from astroquery import besancon
        
        if (not os.path.isfile(self.model)) or clobber:
            print("Fetching Besancon model from server...")
            besancon_model = besancon.Besancon.query(email='adrian.gutierrez@yale.edu',
                                            glon=self.glon,glat=self.glat,
                                            smallfield=True,
                                            area = 0.04,
                                            mag_limits = {"K":(5,19)},
                                            extinction = 0.0,
                                            verbose = True,
                                            retrieve_file=True,
                                            rsup=10.)
            besancon_model.write(self.model,format="fits")
        else:
            print("Besancon model already downloaded. Use clobber=True to fetch new versions.")
            
                                            
            
    def get_contours(self, fitsfile, av=10.):
        """
        Given a Bolocam FITS file, return the contours at a given flux level
        """
        
        hdulist = fits.open(fitsfile)

        header = hdulist[0].header
        img = hdulist[0].data
        #hdulist.close()

        # from Foster 2012
        #av_to_jy = 6.77e22/9.4e20 # cm^-2 / Jy / (cm^-2 / AV) = AV/Jy
        #if header.get('BGPSVERS').strip()=='1.0':
        #av_to_jy /= 1.5

        contour_level = self.contour_level #10 av in Jy?
        #av / av_to_jy

        wcs = pywcs.WCS(header)
        #wcsgrid = wcs.wcs_pix2world( np.array(zip(np.arange(wcs.naxis1),np.arange(wcs.naxis2))), 0 ).T
        yy,xx = np.indices(img.shape)

        img[img!=img] = 0
        
        #Set the borders of an image to be zero (blank) so that all contours close
        img[0,:] = 0.0
        img[-1,:] = 0.0
        img[:,0] = 0.0
        img[:,-1] = 0.0
        
        C = _cntr.Cntr(xx,yy,img)
        paths = [p for p in C.trace(contour_level) if p.ndim==2]

        wcs_paths = [wcs.wcs_pix2world(p,0) for p in paths]


        index = 0
        self.good_contour = False
        
        if len(wcs_paths) > 1:
            print("More than one contour")
            for i,wcs_path in enumerate(wcs_paths):
                path = Path(wcs_path)        
                if path.contains_point((self.glon,self.glat)):
                    index = i
                    print("This was the contour containing the center")
                    self.good_contour = True
            self.contours = wcs_paths[index]
        else:
            self.good_contour = True
            self.contours =  wcs_paths[0]
        self.contour_area = self.calc_contour_area(self.contours)
        
        #Need to find a way to ONLY select the contour closest to our cloud position!!!
        
    def show_contours_on_threecolor(self, color='c',clobber=False):
        """
        Make a three-color image

        """
        from extinction_distance.support import zscale
        print("Making color-contour checkimage...")
        if (not os.path.isfile(self.rgbcube)) or clobber:
            aplpy.make_rgb_cube([self.kim,self.him,self.jim],self.rgbcube,north=True,system="GAL")
        k = fits.getdata(self.kim)
        r1,r2 = zscale.zscale(k)
        h = fits.getdata(self.him)
        g1,g2 = zscale.zscale(h)
        j = fits.getdata(self.jim)
        b1,b2 = zscale.zscale(j)
        
        aplpy.make_rgb_image(self.rgbcube,self.rgbpng,
                             vmin_r = r1, vmax_r = r2,
                             vmin_g = g1, vmax_g = g2,
                             vmin_b = b1, vmax_b = b2)
        f = aplpy.FITSFigure(self.rgbcube2d)
        f.show_rgb(self.rgbpng)
        f.show_markers([self.glon],[self.glat])
        f.show_polygons([self.contours],edgecolor='cyan',linewidth=2)
        #f.show_contour(self.continuum,levels=[self.contour_level],convention='calabretta',colors='white')
        f.save(self.contour_check)


    def calc_contour_area(self,xy):
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
        return np.abs(0.5*s)

    def make_photo_catalog(self,force_completeness=False):
        """
        Reads from photometry parameters
        Uses the trimmed image data
        """
        from extinction_distance.completeness import determine_completeness
        
        
        if self.contour_area > 0.0001 and self.good_contour:
            sex = determine_completeness.do_setup(self.name,survey="UKIDSS")
            k_corr = determine_completeness.do_phot(sex,self.name,survey="UKIDSS")
            if (force_completeness) or (not os.path.isfile(self.completeness_filename)):
                determine_completeness.do_completeness(sex,self.name,self.contours,survey="UKIDSS",k_corr=k_corr,numtrials = 100)
            self.catalog = atpy.Table(os.path.join(self.data_dir,self.name+"_MyCatalog_UKIDSS.vot"))
            self.catalog.describe()
        else:
            print("Bad contour (too small, or does not contain center point)")
            raise(ValueError)
            
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
        self.find_stars_in_contour(poly,"UKIDSS")
        blue,n_blue = self.count_blue_stars_in_contour(self.completeness,
                                        blue_cut=blue_cut,
                                        kupperlim = kup,
                                        klowerlim = klo,
                                        ph_qual = False,
                                        catalog=self.catalog,
                                        plot=True,
                                        survey="UKIDSS")
        print("Total area is...")
        print(self.total_poly_area)
        print("arcminutes")
        self.allblue+=blue
        self.n_blue +=n_blue
        self.model_data = self.load_besancon() #Read in the Besancon model
        #print(self.model_data)
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
                        self.density,survey="UKIDSS")
        print(d)
        distance_ukidss = d
        upper_ukidss = upp
        lower_ukidss = low
        print("==== Distance Results ====")
        print(distance_ukidss)
        print(lower_ukidss)
        print(upper_ukidss)
        print("==========================")
    
    def load_data(self):
        """
        Load the previously generated data.
        Is this really necessary? 
        """
        self.all_poly = self.contours
        self.total_poly_area = self.contour_area*(3600.) #Needs to go to sq arcmin
        self.load_completeness()
        #self.load_photo_catalog() #We just keep this around in memory
                                   #from make_photo_catalog()
                                   #Not necessarily the best plan
    
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
        if survey == "UKIDSS":
            points = np.column_stack((self.catalog.L,self.catalog.B))
            yo = path.contains_points(points)
            try:
                self.catalog.add_column("CloudMask",yo,description="If on cloud")
            except ValueError:
                self.catalog.CloudMask = self.catalog.CloudMask + yo
    
    
    def load_completeness(self):
        """
        Load in the saved estimates of completeness.
        """
        mag  = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19])
        g = open(self.completeness_filename,'r')
        comp = pickle.load(g)
        g.close()
        self.completeness = np.column_stack((mag,comp))
    
    def load_besancon(self):
        from astropy.table import Table
        return(Table.read(self.model))
     
    def load_besancon_old(self,filename,write=False):
        """
        Read a besancon model file
        """
        f = open(filename,'r')
        lines = f.readlines()

        d = defaultdict(list)

        head_done = False
        #Enumerate is for debugging code
        for i,line in enumerate(lines):

            if line.startswith("  Dist"): #Header row
                head_done = not head_done #Header row is duplicated at end of data
                headers = line.split()
                n_head = len(headers)
            elif head_done == True:
                line_elements = line.split()
                if len(line_elements) < n_head: #Age/Mass columns are sometimes conjoined
                    #print("***** Problem Line Found ****")
                    #print(line_elements)

                    temp = line_elements[8:]
                    trouble_entry = line_elements[6]
                    age = trouble_entry[0]
                    mass = trouble_entry[1:]
                    line_elements.append('0')
                    line_elements[9:] = temp
                    line_elements[7] = mass
                    line_elements[6] = age
                    #print(line_elements)
                for header,element in zip(headers,line_elements):
                    d[header].append(float(element))

        #Put into atpy Table.
        #This is optional, we could just return d
        t = atpy.Table()
        for header in headers:
        #       print(header)
            t.add_column(header,d[header])
        t.columns['Dist'].unit = 'kpc'
        t.columns['Mv'].unit = 'mag'
        t.columns['Mass'].unit = 'Msun'

        t.add_comment("Bescancon Model")


    #       t.describe()
        if write:
            t.write("Test.fits")
        return(t)
    def count_blue_stars_in_contour(self,completeness,blue_cut=1.3,kupperlim = 15.,klowerlim = 12.,ph_qual = False,plot=False,catalog=None,survey=None):
        """
        Determine which of the stars inside
        the contour are blue. And calculate the
        areal density of such stars. 
        """

        print("Reached Count stage")

        print(completeness[...,0])

        f = interp1d(completeness[...,0],completeness[...,1],kind='linear')

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
        print(compfactor)
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
