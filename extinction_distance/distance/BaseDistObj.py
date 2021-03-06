#!/usr/bin/env python
# encoding: utf-8
"""
BaseDistObj.py

A basic object to calculate extinction distances
via the Foster et al. (2012) Blue Number Count Method.

Should be subclassed in different regions to
use different different survey data.

Roughly l > 0 can do UKIDSS/BGPS
and l < 0 can do VVV/ATLASGAL

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
import aplpy
from astropy import wcs
from astropy import coordinates
from astropy import units as u
from astropy.table import Table
import astropy.wcs as pywcs
from astropy.io import fits
import matplotlib.pyplot as plt #For contouring and display
from scipy.interpolate import interp1d
from collections import defaultdict
from matplotlib.path import Path
from extinction_distance.support import smooth
import pylab
from scipy.special import gamma
import matplotlib._cntr as _cntr

from extinction_distance.distance import determine_distance

#These are more complicated additions
#Sextractor and montage are required
import montage_wrapper as montage

import shapely
import shapely.geometry #Necessary for contour code

import warnings


from astroquery.magpis import Magpis

class BaseDistObj():
    def __init__(self,name,coords,nir_survey=None,cont_survey=None,Ak=1.0):
        """
        A BaseDistObj is defined by name and coordinates
        In order to do anything useful, you must also 
        specify the near-infrared survey (nir_survey)
        and continuum survey (cont_survey) to use.
        
        Possible choices are 
        nir_survey = "UKIDSS" or "VISTA"
        cont_survey = "BGPS" or "ATLASGAL"
        """
        
        self.name = name
        self.glon,self.glat = coords
        self.glat = float(self.glat)
        self.glon = float(self.glon)
        
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        
        
        magpis_lookup = {"BGPS":"bolocam","ATLASGAL":"atlasgal"}

        self.magpis_survey = magpis_lookup[cont_survey]

        self.gc = coordinates.Galactic(l=self.glon, b=self.glat, unit=(u.degree, u.degree))

        self.data_dir = self.name+"_data/"
        try:
            os.makedirs(self.data_dir)
        except OSError:
            pass
        self.nir_survey = nir_survey
        self.cont_survey = cont_survey

        self.besancon_area = 0.04*u.deg*u.deg #Area for model in sq. degrees. Large=less sampling error
        self.nir_directory = "" # XXX This needs to point to a way to save XXX
        self.nir_im_size = 6*u.arcmin #Size of NIR cutout (symmetric) in arcminutes
        self.small_nir_im_size = 3*u.arcmin #Size of NIR image for Sextractor
        self.continuum_im_size = 10*u.arcmin
        
        self.contour_level = self.calculate_continuum_level(cont_survey=cont_survey,Ak=Ak)
        
        self.jim = self.data_dir+self.name+"_"+self.nir_survey+"_J.fits"
        self.him = self.data_dir+self.name+"_"+self.nir_survey+"_H.fits"
        self.kim = self.data_dir+self.name+"_"+self.nir_survey+"_K.fits"
        self.nir_cal_cat = self.data_dir+self.name+"_2MASS_cat.vot"
        self.nir_cat = self.data_dir+self.name+"_"+self.nir_survey+"_cat.fits"
        self.continuum = self.data_dir+self.name+"_"+self.cont_survey+".fits"
        self.rgbcube = self.data_dir+self.name+"_cube.fits"
        self.rgbcube2d = self.data_dir+self.name+"_cube_2d.fits"
        
        self.rgbpng = self.data_dir+self.name+".png"
        self.contour_check = self.data_dir+self.name+"_contours.png"

        self.model = self.data_dir+self.name+"_model.fits"
        self.completeness_filename = os.path.join(self.data_dir,self.name+"_completeness_"+self.nir_survey+".pkl")
        self.zpcorr_filename = os.path.join(self.name+"_data/",self.name+"_zpcorr_"+self.nir_survey+".pkl")
    
        self.photocatalog = os.path.join(self.data_dir,self.name+"_MyCatalog_"+self.nir_survey+".vot")
    
        #try:
        #    self.get_contours(self.continuum)
        #except:
        #    pass
    
    def calculate_continuum_level(self,cont_survey=None,Ak=1.0,T=20.*u.K):
        """
        Calculate the appropriate continuum contour level 
        in the units used for the maps from a given survey.
        
        Both ATLASGAL and BGPS are in Jy/beam, so we return
        those units.
        """
        if cont_survey == "BGPS":
            lam   = 1120. * u.micron
            theta = 31. * u.arcsec
            knu   = 0.0114 * u.cm * u.cm / u.g
        elif cont_survey == "ATLASGAL":
            lam   = 870. * u.micron
            theta = 19.2 * u.arcsec 
            knu   = 0.0185 * u.cm * u.cm / u.g
        Tdust = T
        Av = Ak*(1./0.112) #Using Schlegel et al. (1998) value for UKIRT K
        NH2 = 9.4e20*Av / (u.cm*u.cm) #Bohlin et al. (1978)
        boltzfac = (1.439*(lam/(u.mm))**(-1)*(Tdust/(10 * u.K))**(-1)).decompose()
        Snu = NH2 / ((2.02e20 * (1./u.cm) * (1./u.cm)) * 
                     (np.exp(boltzfac)) *
                     (knu/(0.01 * u.cm * u.cm/u.g))**(-1) *
                     (theta/(10*u.arcsec))**(-2) * 
                     (lam/u.mm)**(3)) / 1000.
        return(Snu.decompose())
        
    def get_nir_images(self,clobber=False):
        """
        Get NIR data/images for this region.
        This comes from astropy.astroquery

        Raw data is saved into self.data_dir as 
        self.name+"_"+self.nir_survey+"_J.fits"

        """
        #Get images
        if (not (os.path.isfile(self.kim) and os.path.isfile(self.him) 
            and os.path.isfile(self.jim)) or clobber):
            print("Fetching NIR images from server...")
            
            if self.nir_survey == "VISTA":
                from astroquery.vista import Vista as NIR
                kwargs = {"frame_type":"tilestack"}
            if self.nir_survey == "UKIDSS":
                from astroquery.ukidss import Ukidss as NIR
                kwargs = {}
            
            for filtername,filename in zip(["J","H","K"],(self.jim,self.him,self.kim)):
                #Need to trim on deprecated and distinguish between tilestack and tilestackconf?
                images = NIR.get_images(coordinates.Galactic(l=self.glon, b=self.glat, 
                                            unit=(u.deg, u.deg)),
                                            waveband=filtername,
                                            image_width=self.nir_im_size,
                                            **kwargs)
                #This makes a big assumption that the last UKIDSS/VISTA image is the one we want
                fits.writeto(filename,
                             images[0][-1].data,images[0][-1].header,clobber=True)
        else:
            print("NIR image already downloaded. Use clobber=True to fetch new versions.")
                         
    def get_nir_cat(self,clobber=False,use_twomass=True):
        """
        Get the NIR catalog
        Catalog (necessary for zero-point determination) is saved
        into self.data_dir as
        self.name+"_"+self.nir_survey+"cat.fits"
        """
        print("Fetching NIR catalog from server...")
        if use_twomass:
            if (not os.path.isfile(self.nir_cal_cat)) or clobber:
                from astroquery.irsa import Irsa
                Irsa.ROW_LIMIT = 2000.
                table = Irsa.query_region(coordinates.Galactic(l=self.glon,
                        b=self.glat,  unit=(u.deg, u.deg)), 
                        catalog="fp_psc", spatial="Box", 
                        width=self.nir_im_size)
                #print(table)
            #IPAC table does not take overwrite? But FITS does? So inconsistent and bad
                table.write(self.nir_cal_cat,format='votable',overwrite=clobber)
            else:
                print("NIR catalog already downloaded. Use clobber=True to fetch new versions.")
            
        else:
            if (not os.path.isfile(self.nir_cat)) or clobber:
                if self.nir_survey == "VISTA":
                    from astroquery.vista import Vista as NIR
                if self.nir_survey == "UKIDSS":
                    from astroquery.ukidss import Ukidss as NIR
                table = NIR.query_region(coordinates.Galactic(l=self.glon,
                        b=self.glat,  unit=(u.deg, u.deg)), radius=self.nir_im_size)
                table.write(self.nir_cat,format="fits",overwrite=clobber)
            else:
                print("NIR catalog already downloaded. Use clobber=True to fetch new versions.")
            
            
            #Get catalog. We need this to establish zero-points/colours
            
    def get_continuum(self,clobber=False):
        if (not os.path.isfile(self.continuum)) or clobber:
            print("Fetching continuum cutout from server...")
            image = Magpis.get_images(coordinates.Galactic(self.glon, self.glat,
                    unit=(u.deg,u.deg)), image_size=self.continuum_im_size, survey=self.magpis_survey)
            fits.writeto(self.continuum,
                         image[0].data,image[0].header,clobber=clobber)
        else:
            print("Continuum image already downloaded. Use clobber=True to fetch new versions.")
            
    def get_model(self,clobber=False):
        """
        Get a Besancon model for this region of sky
        """
        
        from astroquery import besancon
        import copy
        import besancon_parameters as bp
        
        if (not os.path.isfile(self.model)) or clobber:
            print("Fetching Besancon model from server...")
            besancon_model = besancon.Besancon.query(email='jonathan.b.foster@yale.edu',
                                            glon=self.glon,glat=self.glat,
                                            smallfield=True,
                                            area = 0.04,
                                            mag_limits = {"K":(5,17)},
                                            extinction = 0.0,
                                            verbose = True,
                                            retrieve_file=True,
                                            rsup=10.,
                                            colind=["J-H","H-K","J-K","V-K",],)
            besancon_model.write(self.model,format="fits")
        else:
            print("Besancon model already downloaded. Use clobber=True to fetch new versions.")
            
    
    def make_contour_at_level(self,wcs,img,yy,xx,level):
        C = _cntr.Cntr(xx,yy,img)
        paths = [p for p in C.trace(level) if p.ndim==2]

        wcs_paths = [wcs.wcs_pix2world(p,0) for p in paths]
        return(wcs_paths)
            
            
    def get_contours(self, fitsfile, Ak=None):
        """
        Given a continuum FITS file, return the contours enclosing 
        the source position at a given flux level.
        
        There are a couple complications:
        1) We seek a closed contour
        2) The contour should only extend over the region
            covered by the NIR image (K-band, in this case)
        
        This function could usefully be refactored/simplified
        """
        
        hdulist = fits.open(fitsfile)

        header = hdulist[0].header
        img = hdulist[0].data
        if not Ak:
            contour_level = self.contour_level #10 av in Jy?
            print("Using a contour level of: "+str(round(contour_level,3)))
            
        else:
            contour_level = self.calculate_continuum_level(
                                 cont_survey=self.cont_survey, 
                                 Ak=Ak)
            print("Using a contour level of: "+str(round(contour_level,3)))

        wcs = pywcs.WCS(header)
        yy,xx = np.indices(img.shape)

        img[img!=img] = 0
        
        #Set the borders of an image to be zero (blank) so that all contours close
        img[0,:] = 0.0
        img[-1,:] = 0.0
        img[:,0] = 0.0
        img[:,-1] = 0.0
        
        self.all_contours = []
        
        
        #for contour_level in contour_levels:
        wcs_paths = self.make_contour_at_level(wcs,img,yy,xx,contour_level)
        #    self.all_contours.append(np.array(wcs_paths))

        index = 0
        self.good_contour = False
        
        print(self.glon,self.glat)
        
        if len(wcs_paths) > 1:
            print("More than one contour")
            for i,wcs_path in enumerate(wcs_paths):
                path = Path(wcs_path)
                #print(path)
                if path.contains_point((self.glon,self.glat)):
                    index = i
                    self.good_contour = True
                    print("This was the contour containing the center")
            #index = 1
            #self.good_contour = True
            self.contours = wcs_paths[index]
            #self.contours[:,0] -= 0.02
            #self.contours[:,1] += 0.02
            #print(self.contours[:,1])
            
        else:
            self.good_contour = True
            self.contours =  wcs_paths[0]            
        
        #This selects a contour containing the center
        #Now we trim the contour to the boundaries of the UKIDSS image
        if self.good_contour:
            #And check to see which contour (if any) contains the center
            self.good_contour = False
            #Find the boundaries of the UKIDSS (K-band image) in Galactic coordinates
            h = fits.getheader(self.kim)
            xmin = 0
            xmax = h['NAXIS1']
            ymin = 0
            ymax = h['NAXIS2']
            wcs = pywcs.WCS(h)
            corners = wcs.wcs_pix2world([[xmin,ymin],[xmin,ymax],[xmax,ymax],[xmax,ymin]],0)
            gals = []
            for coord in corners:
                c = coordinates.ICRS(ra=coord[0],dec=coord[1],unit=[u.deg,u.deg])
                gal = c.transform_to(coordinates.Galactic)
                gals.append(gal)
            mycoords = []
            for gal in gals:
                l,b = gal.l.degree,gal.b.degree
                mycoords.append((l,b))
            p1 = shapely.geometry.Polygon(self.contours)
            p1 = p1.buffer(0)
            #print(p1.is_valid)
            p2 = shapely.geometry.Polygon(mycoords)
            p2.buffer(0)
            #print(p2.is_valid)
            ya = p1.intersection(p2)
            #print(ya)
            try:
                mycontours = []
                xx,yy = ya.exterior.coords.xy
                for ix,iy in zip(xx,yy):
                    mycontours.append((ix,iy))
                self.contours = np.array(mycontours)
                self.good_contour = True
            except AttributeError: #MultiPolygon
                mycontours = []
                for j,poly in enumerate(ya):
                    try:
                        #print(poly)
                        yoyo = np.asarray(poly.exterior.coords.xy)
                        #print(yoyo.shape)
                        #print(yoyo)
                        yoyo2 = yoyo.T
                        #print(poly.exterior.coords.xy)
                        #yya = poly.exterior.coords.xy[0]
                        #print(yya.shape)
                        #print(yoyo2)
                        path = Path(yoyo2)
                        if path.contains_point((self.glon,self.glat)):
                            self.good_contour = True
                            index = i
                            print("This was the contour containing the center")
                            xx,yy = poly.exterior.coords.xy
                            for ix,iy in zip(xx,yy):
                                mycontours.append((ix,iy))
                            self.contours = np.array(mycontours)
                    except IOError:
                        pass
                        
        self.contour_area = self.calc_contour_area(self.contours)
        
        if not self.good_contour:
            print("######## No good contour found ########")
            self.contours = None
            self.contour_area = 0

        
        
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
                             #G337 hand-coded
                             #vmin_r = 5671., vmax_r = 5952.,
                             #vmin_g = 8117., vmax_g = 8688.,
                             #vmin_b = 1462., vmax_b = 1606.)
                             vmin_r = r1, vmax_r = r2,
                             vmin_g = g1, vmax_g = g2,
                             vmin_b = b1, vmax_b = b2)
        f = aplpy.FITSFigure(self.rgbcube2d)
        f.show_rgb(self.rgbpng)
        f.show_contour(self.continuum, colors='white', levels=[0.1,0.2,0.4,0.8,1.6,3.2],smooth=3,convention='calabretta')
        f.show_markers([self.glon],[self.glat])                
        try:
            f.show_polygons([self.contours],edgecolor='cyan',linewidth=2)
        except:
            pass
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

    def make_photo_catalog(self,force_completeness=False,clobber=False):
        """
        Reads from photometry parameters
        Uses the trimmed image data
        """
        from extinction_distance.completeness import determine_completeness
        
        if self.contour_area > 0.0001 and self.good_contour:
            sex = determine_completeness.do_setup(self.name,survey=self.nir_survey)
            if ((not os.path.isfile(self.photocatalog)) or (not os.path.isfile(self.zpcorr_filename))) or clobber:
                self.kcorr = determine_completeness.do_phot(sex,self.name,survey=self.nir_survey)
            else:
                self.kcorr = self.load_zpcorr()
            if (force_completeness) or (not os.path.isfile(self.completeness_filename)):
                determine_completeness.do_completeness(sex,self.name,self.contours,survey=self.nir_survey,k_corr=self.kcorr,numtrials = 100)
            self.catalog = Table.read(self.photocatalog)
        else:
            print("Bad contour (too small, or does not contain center point)")
            raise(NoContourException("No valid contour"))
            
            
    def determine_magnitude_cuts(self, completeness_cut = 0.40):
        """
        Determine the magnitudes we wish to consider for counting blue stars
        
        If completeness is low for a given magnitude it is best
        just to exlude these stars entirely from the analysis
        rather than use them, which would produce wildy varying
        distance estimates
        """
        
        mags = self.completeness[:,0]
        comps = self.completeness[:,1]
        print("Completeness Information: ")
        print(mags)
        print(comps)
        print("Cutting at completeness = "+str(completeness_cut))
        good_mags = mags[(comps > completeness_cut) & (mags < 18)] #Our model only goes to K = 17
        return(np.min(good_mags),np.max(good_mags))
        
            
    def do_distance_estimate(self,color_cut=1.5,diffuse=0.7):
        """
        Calculate the extinction distance
        based on the surface density of blue
        stars inside the contour and the
        Besancon model. We want to perform this estimate over
        a range of different limiting magnitudes and take the
        spread in the results as the error (or quote a lower
        bound from Poisson)
        """
        self.load_data()
        blue_cut = color_cut #Magic number for J-K cut. Put this elsewhere
        diffuse = diffuse
        kup = 17 #More magic numbers
        klo = 11
        comp_cut = 0.4
        
        #inds = [ 0, 1, 2, 3, 4, 5, 6, 7, 8]
        #mags = [11,12,13,14,15,16,17,18,19]
        
        #This part is now redundant again
        inds = [3,4,5,6,7]
        inds = [6]
        mags = self.completeness[:,0]
        comps = self.completeness[:,1]
        
        distance_estimates = []
        lower_error = []
        higher_error = []
        
        
        print(self.completeness)
        
        for ind in inds:
            #if comps[ind] > 0.25:
            #    kup = mags[ind]
            klo,kup = self.determine_magnitude_cuts(completeness_cut = comp_cut)
        
            print("Using stars between K = "+str(klo)+" and "+str(kup))
    
            self.allblue = 0
            self.n_blue = 0
            poly = self.all_poly
            self.find_stars_in_contour(poly,self.nir_survey)
            self.nmin,self.nmean,self.nmax = self.count_blue_stars_in_contour(
                                            self.completeness,
                                            blue_cut=blue_cut,
                                            kupperlim = kup,
                                            klowerlim = klo,
                                            ph_qual = False,
                                            catalog=self.catalog,
                                            plot=True,
                                            survey=self.nir_survey)
            print("Total area is...")
            print(self.total_poly_area)
            print("arcminutes")
            self.model_data = self.load_besancon() #Read in the Besancon model
            self.density = self.nmean/self.total_poly_area
            self.density_lowerlim = self.nmin/self.total_poly_area
            self.density_upperlim = self.nmax/self.total_poly_area
            print(self.density)
            print(self.density_upperlim)
            print(self.density_lowerlim)
            distance,upper,lower = self.get_distance(kup,klo,blue_cut,diffuse=diffuse)
            print("==== Distance Results ====")
            print(distance)
            print(lower)
            print(upper)
            print("==========================")
            self.distance_est = distance
            self.distance_lolim = lower
            self.distance_hilim = upper
            
            results = {"name":self.name,"glon":self.glon,"glat":self.glat,
                       "area":self.contour_area,"n_obs_blue":self.n_blue,
                       "n_est_blue":self.allblue,"dist":self.distance_est,
                       "dist_lolim":self.distance_lolim,"dist_hilim":self.distance_hilim,
                       "klo":klo, "kup":kup, "kcorr":self.kcorr}
            f = open(os.path.join(self.name+"_data/",self.name+"_results_k"+str(kup)+".pkl"),'w')
            pickle.dump(results,f)
            f.close()
            distance_estimates.append(self.distance_est)
            lower_error.append(lower)
            higher_error.append(upper)
                
        #if (len(distance_estimates) > 0):
        #    drange = np.ptp(distance_estimates)
        #    sigma = drange/4.
        #    self.distance_est = np.median(distance_estimates)
        #    lower_min = np.min(lower_error)
        #    upper_min = np.min(higher_error)
        #    if (sigma < lower_min) or (lower_min == 0.0):
        #        lowerlim = lower_min
        #    else:
        #        lowerlim = sigma
        #    if sigma < upper_min or (upper_min == 0.0):
        #        upperlim = upper_min
        #    else:
        #        upperlim = sigma
        #    
        #    self.distance_lolim = lowerlim
        #    self.distance_hilim = upperlim

        #else:
        #    pass #just use the most recent value. Should rarely(?) or never happen
        self.distance_est = np.median(distance_estimates)
        self.distance_lolim = np.median(lower_error)
        self.distance_hilim = np.median(higher_error)

        
        results = {"name":self.name,"glon":self.glon,"glat":self.glat,
                   "area":self.contour_area,"n_obs_blue":self.n_blue,
                   "n_est_blue":self.allblue,"dist":self.distance_est,
                   "dist_lolim":self.distance_lolim,"dist_hilim":self.distance_hilim,
                   "klo":klo, "kup":kup, "kcorr":self.kcorr}
        print(results)
        f = open(os.path.join(self.name+"_data/",self.name+"_results.pkl"),'w')
        pickle.dump(results,f)
        f.close()
        return(results)
        
    
    def load_data(self):
        """
        Load the previously generated data.
        Is this really necessary? 
        """
        self.all_poly = self.contours
        self.total_poly_area = self.contour_area*(3600.) #Needs to go to sq arcmin
        self.load_completeness()
        self.catalog = Table.read(self.photocatalog)
    
    def find_stars_in_contour(self,contour,survey):
        """
        From the photometry catalog we
        determine which stars are inside
        the contour of the cloud.
        """
        verts = np.array(contour,float)
        path = Path(verts)        
        if (survey == "UKIDSS") or (survey == "VISTA"):
            points = np.column_stack((self.catalog['L'],self.catalog['B']))
            yo = path.contains_points(points)
            try:
                self.catalog["CloudMask"] = yo
            except ValueError:
                self.catalog['CloudMask'] = self.catalog['CloudMask'] + yo

    def load_zpcorr(self):
        """
        Load in the saved estimates of the zero-point
        """
        g = open(self.zpcorr_filename,'r')
        zpcorr = pickle.load(g)
        g.close()
        return(zpcorr)

    
    
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
     
     

    def count_blue_stars_in_contour(self,completeness,blue_cut=1.3,kupperlim = 15.,klowerlim = 12.,ph_qual = False,plot=False,catalog=None,survey=None):
        """
        Determine which of the stars inside
        the contour are blue. 
        
        Estimate a confidence 0.95 confidence interval
        for the number of stars present given the detection
        completeness. Approximate by binning on magnitude
        and using the binomial distribution for each of these
        bins (with a fixed detection completeness).
        
        Numerically convolve resulting pdfs to get the pdf
        for the sum of individual magnitude bins
        """

        print("Reached Count stage")

        f = interp1d(completeness[...,0],completeness[...,1],kind='linear')

        print(kupperlim)
        print(klowerlim)
        print(catalog['KMag'])
        print(catalog) #Apparently this print statement is necessary to make the selection of good rows
                       #in the next few lines work. That is super broken and bad of astropy.table

        good_rows = np.logical_and(catalog['KMag'] < kupperlim,catalog['KMag'] > klowerlim)
        print(good_rows)
        
        good = catalog[good_rows]
        #good = catalog[(catalog['KMag'] < kupperlim) & (catalog['KMag'] > klowerlim)]
        in_contour = good[good['CloudMask'] == 1]
        JminK = in_contour['JMag'] - in_contour['KMag']
        blue_in_contour = in_contour[JminK < blue_cut]
        self.plot_color_histogram(in_contour['JMag']-in_contour['KMag'],
                                  blue_in_contour['JMag']-blue_in_contour['KMag'],kupperlim)

        compfactor = f(blue_in_contour['KMag'])
        
        from scipy.stats import poisson
        ns = np.ones_like(compfactor)
        lam = np.dot(ns,1./compfactor)
        mmean = lam
        mmin,mmax = poisson.interval(0.99, lam)
        
        print("Number of blue stars in contour")
        print(len(ns))
        #print("Observed in each bin:")
        #print(ns)
        #print("Completeness in each bin")
        #print(compfactor)
        print("Estimated nnumber of blue stars in contour")
        print(mmean)
        print(mmin)
        print(mmax)

        return(mmin,mmean,mmax)
        
    
    
    def plot_color_histogram(self,JminK,JminK_blue,kupperlim):
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
        plt.savefig(self.data_dir+self.name+"_hist_k"+str(kupperlim)+".png")
    
    def get_distance(self,kupperlim,klowerlim,color_cut,diffuse=0.7):
        """
        A faster method to get distances
        
        """
        
        max_distance = 10000. #Max distance in pc
        
        blue_star_density_model = []
        cloud_distances = np.arange(0,max_distance,50)[::-1]
        

        colorcut=color_cut
        Mags_per_kpc = diffuse
        
        foreground = self.model_data[self.model_data['Dist'] <= max_distance/1000.]
        print(foreground)
        #Now, all of a sudden, Besancon does not return J-K?
        #Because I was using a custom set of keyword defaults. Set these manually inside the program
        #Just need to compare current defaults to my custom ones (on Kuon) to see what needs to be updated
        foreground['K'] = foreground['V'] - foreground['V-K']
        foreground['corrj'] = foreground['J-K']+foreground['K'] + Mags_per_kpc*0.276*foreground['Dist']
        foreground['corrk'] = foreground['K'] + Mags_per_kpc*0.114*foreground['Dist']

        for cloud_distance in cloud_distances:
            foreground = foreground[foreground['Dist'] <= cloud_distance/1000.]
            J_min_K = foreground[(foreground['corrk'] < kupperlim) & (foreground['corrk'] > klowerlim) & 
                                 (foreground['corrj']-foreground['corrk'] < colorcut)]
            #The 25/3600. takes us to per square arcmin for a field of 0.04 sq degree
            blue_star_density_model.append(len(J_min_K)*(25/3600.))
        blue_star_density_model = blue_star_density_model[::-1]
        cloud_distances = cloud_distances[::-1]
        
        blah = smooth.smooth(np.array(blue_star_density_model),window_len=9,window='hanning')
        pylab.clf()
        pylab.plot(cloud_distances,blah,label="Besancon",color='k',ls='--')
        pylab.xlabel("Distance [pc]")
        pylab.ylabel("Number of Blue Stars/(sq. arcmin)")

        s = interp1d(cloud_distances,blue_star_density_model,kind=5)
        xx = np.linspace(0,max_distance-100.,num=max_distance-100)
        yy = s(xx)

        lower = np.where(yy < self.density_lowerlim)
        upper = np.where(yy > self.density_upperlim)
    
        center1 = np.where(yy <= self.density)
        center2 = np.where(yy > self.density)

        central = center1[0][-1]
        print("Maximum density probed = ")
        max_density = yy[-1]
        print(max_density)
        #central = np.max(central,yy[-1])
        print("Maximum distance probed = ")
        max_dist = np.where(yy < 0.99*max_density)
        max_distance = max_dist[0][-1]
        print(max_distance)
    
        no_upper = False
        no_lower = False
        no_center = False
    
        try:
            upperlim = upper[0][0]
        except IndexError:
            print("Failed to find upper distance limit")
            upperlim = -999
            no_upper = True
        try:
            lowerlim = lower[0][-1]
        except IndexError:
            print("Failed to find lower distance limit")
            lowerlim = -999
            no_lower = True
            
        print(central)
        if central > max_distance:
            central = max_distance
            no_center = True

        pylab.axhline(y=self.density,linewidth=2,color='k')
        pylab.axhline(y=self.density_upperlim,color='k',linestyle=':')
        pylab.axhline(y=self.density_lowerlim,color='k',linestyle=':')

        pylab.axvline(x=lowerlim,color='k',linestyle=':')
        pylab.axvline(x=upperlim,color='k',linestyle=':')
        pylab.axvline(x=central,color='k',linewidth=2)
        pylab.figtext(0.15,0.8,self.name,ha="left",fontsize='large',backgroundcolor="white")

        if no_lower and no_center and no_upper:
            #Lower limit set at max distance probed
            merr = 0.0
            perr = 0.0
            central = max_distance
        elif no_center and no_upper:
            #Lower limit set by (found) lower limit
            central = lowerlim
            merr = 0.0
            perr = 0.0
        elif no_upper:
            #Upper limit undefined
            central = central
            perr = 0.0
            merr = lowerlim-central
        else:
            central = central
            perr = upperlim - central
            merr = lowerlim-central

        pylab.figtext(0.15,0.75,str(central)+r"$_{"+str(merr)+r"}$"+r"$^{+"+str(perr)+r"}$"+" pc",fontsize='large',backgroundcolor="white")
        pylab.figtext(0.15,0.70,r'Area = '+str(round(self.total_poly_area,2))+' arcmin$^2$',ha="left",fontsize='large',backgroundcolor="white")
        pylab.figtext(0.15,0.65,"Survey: "+self.nir_survey,ha="left",fontsize='large',backgroundcolor="white")

        pylab.xlim(0,10000)
        fig = pylab.gcf()
        fig.set_size_inches(6,6)
        Size = fig.get_size_inches()
        #print("Size in Inches: "+str(Size))
        pylab.savefig(os.path.join(self.name+"_data",self.name+"_Distance_"+self.nir_survey+"_k"+str(kupperlim)+'.png'))
        pylab.clf()
        pylab.close('all')
        print("Distance = "+str(central)+"+"+str(perr)+str(merr))
        return(central,perr,merr)
        
class NoContourException(Exception):
    pass
        
if __name__ == '__main__':
    unittest.main()
