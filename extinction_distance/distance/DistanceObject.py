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

#These are necessary imports
import numpy as np
import atpy
import aplpy
from astropy import wcs
from astropy import coordinates
from astropy import units as u
from astropy.io import fits
import matplotlib.pyplot as plt #For contouring and display
from scipy.interpolate import interp1d
from collections import defaultdict

import matplotlib._cntr as _cntr


#These are my programs
from extinction_distance.completeness import determine_ukidss_zp #does what it says
from extinction_distance.completeness import determine_completeness
from extinction_distance.completeness import sextractor
from extinction_distance.support import coord
from extinction_distance.distance import determine_distance

#These are more complicated additions
#Sextractor and montage are required
#import montage
#import ds9

#Astropy related stuff
#from astropy import astroquery
#from astroquery import besancon
#import astropy.io.ascii as asciitable
#from astropy.io.ascii import besancon as besancon_reader
#from astroquery import ukidss
from astroquery.ukidss import Ukidss


class DistanceObject():
    def __init__(self,name,coords):
        self.name = name
        self.glon,self.glat = coords
        self.glat = float(self.glat)
        self.glon = float(self.glon)

        print(self.glat)
        self.gc = coordinates.GalacticCoordinates(l=self.glon, b=self.glat, unit=(u.degree, u.degree))

        self.data_dir = self.name+"_data/"
        try:
            os.makedirs(self.data_dir)
        except OSError:
            pass

        self.besancon_area = 0.04*u.deg*u.deg #Area for model in sq. degrees. Large=less sampling error
        self.ukidss_directory = "" # XXX This needs to point to a way to save XXX
        self.ukidss_im_size = 15*u.arcmin #Size of UKIDSS cutout (symmetric) in arcminutes
        self.small_ukidss_im_size = 0.15*u.deg #Size of UKIDSS image for Sextractor
    
    def generate_besancon(self):
        """
        Get the besancon model for this region.
        This comes from astropy.astroquery
        """
        besancon_model = besancon.request_besancon('your@email.net',
                        self.glon,self.glat,smallfield=True,area=self.besancon_area)
        #I probably want to save this as well? Model files take some time to generate.
        self.besancon_model = asciitable.read(besancon_model,
                        Reader=asciitable.besancon.BesanconFixed,guess=False)
    
    def get_ukidss_images(self):
        """
        Get UKIDSS data/images for this region.
        This comes from astropy.astroquery

        Raw data is saved into self.data_dir as 
        self.name+"_UKIDSS_J.fits"

        """
        #Get images
        for filtername in ["J","H","K"]:
            iamges = Ukidss.get_images(coord.Galactic(l=self.glon, b=self.glat, unit=(u.deg, u.deg)),
                                       waveband=filtername,
                                       radius=self.ukidss_im_size)
            fits.writeto(self.data_dir+"/"+self.name+"_UKIDSS_"+filtername+".fits",
                         Jim[0][1].data,Jim[0][1].header)
                         
    def get_ukidss_cat(self):
        """
        Get the UKIDSS catalog
        Catalog (necessary for zero-point determination) is saved
        into self.data_dir as
        self.name+"_UKIDSS_cat.fits"
        """
        table = Ukidss.query_region(coord.Galactic(l=self.glon,
                    b=self.glat,  unit=(u.deg, u.deg)), radius=self.ukidss_im_size)
        table.write(self.data_dir+"/"+self.name+"_UKIDSS_cat.fits",format="fits")
        #Get catalog. We need this to establish zero-points/colours
        
    
    def find_bgps(self):
        """
        Choose the appropriate continuum file
        from which to make a contour. For now
        this is BGPS, but this could be swapped
        out for ATLASGAL or HiGAL.
        """
        bgps_file = self.selectBolocamImage()
        self.bgps_filename = "/Volumes/Data1/BGPS/"+bgps_file
        
        
    
    def make_contour(self):
        """
        Make a contour from the file identified
        in find_bgps(). 
        This currently makes use of DS9 region files.
        This makes it easy to see/display/edit the contours
        but is a fair amount of overhead for a simple contouring
        process.

        """
        BGPS_con_level = 0.10 #Contour level. Make this flexible?


        #Display Bolocam
        print(self.bgps_filename)
        #bgps,hdr = fits.getdata(self.bgps_filename,header=True)
        #bwcs = wcs.WCS(hdr)
        s = 15 #Size of cut-out array in arcmin
        #clipped = ai.clipImageSectionWCS(bgps,bwcs,self.glon,self.glat,s/60.,returnWCS=True)
        blah = self.gc.icrs
        ra = blah.ra.degrees
        dec = blah.dec.degrees
        print(ra,dec)
        montage.mSubimage(self.bgps_filename,"temp.fits",ra,dec,s/60.)
        #bgps = clipped['data']
        #bwcs = clipped['wcs']
        #ai.saveFITS("temp.fits",clipped['data'],clipped['wcs'])
        f = fits.open("temp.fits")
        #d.set_np2arr(f[0])

        d = ds9.ds9()
        d.set('file "temp.fits"')
        d.set('zscale')
        d.set('wcs sky galactic degrees')
        d.set('contour color red')

        d.set('contour')
        d.set('contour clear')
        d.set('contour nlevels 1')
        d.set('contour limits '+str(BGPS_con_level)+" "+str(BGPS_con_level))
        d.set('contour smooth 5')
        d.set('contour method smooth')
        d.set('contour apply')
        d.set('contour yes')
        #d.set('contour copy')
        self.contour_name = self.name+".reg"
        self.contour_path = self.data_dir+self.contour_name
        d.set('contour convert')
        d.set('regions skyformat degrees')
        d.set('regions save '+self.contour_path+' wcs galactic')
        #return(ds9)
        #I wrote to parsing code to handle DS9 regions, not contour
        #files directly. For now I continue with this, somewhat odd,
        #choice of format

        #Need to kill ds9 process at the end. 

    
    def process_ukidss_data(self):
        """
        Trim the images obtained from
        get_ukidss_data to the smallest
        useful size for completeness.
        """
        survey="UKIDSS"
        bands = ["J","H","K"]
        for band in bands:
            infile = os.path.join(self.data_dir,self.name+"_"+survey+"_"+band+".fits")
            outfile = os.path.join(self.data_dir,self.name+"_"+survey+"_trim_"+band+".fits")
            try:
                data,hdr = fits.getdata(infile,1,header=True)
            except IndexError:
                data,hdr = fits.getdata(infile,header=True)
            data = np.array(data,dtype=np.float64) #Explicit cast to help Sextractor?
            fits.writeto("Temp.fits",data,hdr,clobber=True)
            montage.mSubimage("Temp.fits",outfile,self.glon,self.glat,self.small_ukidss_im_size)
            os.remove("Temp.fits")
        
    
    def make_photo_catalog(self):
        """
        Reads from photometry parameters
        Uses the trimmed image data
        """
        sex = determine_completeness.do_setup(self.name,survey="UKIDSS")
        k_corr = determine_completeness.do_phot(sex,self.name,survey="UKIDSS")
        determine_completeness.do_completeness(sex,self.name,survey="UKIDSS",k_corr=k_corr,numtrials = 10)
        self.catalog = atpy.Table(os.path.join(self.data_dir,self.name+"_MyCatalog_UKIDSS.vot"))
        self.catalog.describe()

    def do_distance_estimate(self):
        """
        Calculate the extinction distance
        based on the surface density of blue
        stars inside the contour and the
        besancon model.
        """
        pass
        self.load_data()
        blue_cut = 1.5 #Magic number for J-K cut. Put this elsewhere
        kup = 17 #More magic numbers
        klo = 11
        self.allblue = 0
        self.n_blue = 0
        for poly in self.all_poly:
            self.find_stars_in_contour(poly,"UKIDSS")
            blue,n_blue = self.count_blue_stars_in_contour(self.completeness,
                                            blue_cut=blue_cut,
                                            kupperlim = kup,
                                            klowerlim = klo,
                                            ph_qual = False,
                                            catalog=self.catalog,
                                            plot=True,
                                            survey="UKIDSS")
            self.allblue+=blue
            self.n_blue +=n_blue
        self.model_data = self.load_besancon(os.path.join(self.data_dir,self.name+".resu"),write=False) #Read in the Besancon model
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

        #make_images()
    
    def load_data(self):
        """
        Load the previously generated data.
        Is this really necessary? 
        """
        all_poly,total_area = self.load_contour(self.name,self.glon,self.glat)
        self.all_poly = all_poly
        self.total_poly_area = total_area
        self.load_completeness()
        #self.load_photo_catalog() #We just keep this around in memory
                                   #from make_photo_catalog()
                                   #Not necessarily the best plan
    
    def load_completeness(self):
        """
        Load in the saved estimates of completeness.
        """
        mag  = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19])
        g = open(os.path.join(self.data_dir,self.name+"_completeness_UKIDSS.pkl"),'r')
        comp = pickle.load(g)
        g.close()
        self.completeness = np.column_stack((mag,comp))


    def load_contour(self,region_name,cen_l,cen_b,size_l=0.4,size_b=0.4,do_area=True):
        """
        Load in the DS9 contour file and calculate the area by throwing
        down random points in the area and seeing which ones are inside
        the contour.
        """
        # because nxutils is mpl>1.2, include it here to reduce chance of crash on import
        import matplotlib.nxutils as nx
        f = open(os.path.join(region_name+"_data",region_name+".reg"),'r')
        current_poly = []
        all_poly = []
        exclude = False
        include = False
        for line in f:
            if line.startswith("polygon") and line.endswith("red\n"): #Red contours mark good/useful ones
                include = True
            #elif line.startswith("polygon") and line.endswith("cyan\n"): #Cyan contours mean exclude
            #       exclude = True
            if include or exclude:
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
        f.close()
        total_area = 0
        if do_area:
        #print(len(all_poly))
            n_samp = 400000.
            testpoints = np.random.rand(n_samp,2)
            testpoints[:,0] = (testpoints[:,0]-0.5)*size_l+cen_l
            testpoints[:,1] = (testpoints[:,1]-0.5)*size_b+cen_b
            plt.plot(testpoints[:,0],testpoints[:,1],'.')

            total_area = 0
            for poly in all_poly:
                verts = np.array(poly,float)
                g = open(os.path.join(region_name+"_data",region_name+".poly"),'a')
                print >>g,verts
                yo = nx.points_inside_poly(testpoints,verts)
                #print(yo)
                plt.plot(testpoints[yo,0],testpoints[yo,1],'.')
                area = yo.sum()/n_samp*(3600*size_l*size_b)
                total_area = total_area+area
        print(total_area)
        #pylab.show()
        #yo = raw_input('Enter any key to quit: ')
        return(all_poly,total_area)


    def load_besancon(self,filename,write=False):
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

    def find_stars_in_contour(self,contour,survey):
        """
        From the photometry catalog we
        determine which stars are inside
        the contour of the cloud.
        """

        import matplotlib.nxutils as nx
        verts = np.array(contour,float)
        if survey == "2MASS":
            points = np.column_stack((self.twomass.L,self.twomass.B))
            yo = nx.points_inside_poly(points,verts)
            try:
                self.twomass.add_column("CloudMask",yo,description="If on cloud")
            except ValueError:
                self.twomass.CloudMask = self.twomass.CloudMask + yo
            #self.twomass.write("Modified_2MASS.vot")
        if survey == "UKIDSS":
            points = np.column_stack((self.catalog.L,self.catalog.B))
            yo = nx.points_inside_poly(points,verts)
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
        #    self.plotcolorhistogram(in_contour.JMag-in_contour.KMag,blue_in_contour.JMag-blue_in_contour.KMag,label="On_Source",survey=survey)

        compfactor = f(blue_in_contour.KMag)
        #print(compfactor)
        #print(blue_in_contour)
        #Hack. Really just wants np.ones()/compfactor
        blue_stars = (blue_in_contour.KMag)/(blue_in_contour.KMag)/compfactor
        n_blue = len(blue_in_contour.KMag)
        print(sum(blue_stars))

        return(sum(blue_stars),n_blue)
        
    
    def make_images(self):
        """
        Make the three check-images
        1) 3-color image of region
        2) J-K histogram
        3) Distance estimate plot
        """
        pass
    
    def selectBolocamImage(self):
        if (self.glon > 0.0 and self.glon < 4.5) or (self.glon > 358.50 and self.glon < 360.0):
            bnum = "super_gc"
        elif self.glon > 4.5 and self.glon < 7.5:
            bnum = "l006"
        elif self.glon > 7.5 and self.glon < 10.5:
            bnum = "l009"
        elif self.glon > 10.5 and self.glon < 13.5:
            bnum = "l012"
        elif self.glon > 13.5 and self.glon < 21.5:
            bnum = "l018"
        elif self.glon > 21.5 and self.glon < 27.5:
            bnum = "l024"
        elif self.glon > 27.5 and self.glon < 29.5:
            bnum = "l029"
        elif self.glon > 29.5 and self.glon < 30.5:
            bnum = "l030"
        elif self.glon > 30.5 and self.glon < 33.5:
            bnum = "l032"
        elif self.glon > 33.5 and self.glon < 37.5:
            bnum = "l035"
        elif self.glon > 37.5 and self.glon < 42.5:
            bnum = "l040"
        elif self.glon > 42.5 and self.glon < 47.5:
            bnum = "l045"
        elif self.glon > 47.5 and self.glon < 52.5:
            bnum = "l050"
        elif self.glon > 52.5 and self.glon < 58.5:
            bnum = "l055"
        elif self.glon > 58.5 and self.glon < 69.5:
            bnum = "l065"
        elif self.glon > 69.5 and self.glon < 75.5:
            bnum = "l072"
        elif self.glon > 75.5 and self.glon < 77.5:
            bnum = "l077"
        elif self.glon > 77.5 and self.glon < 80.5:
            bnum = "l079"
        elif self.glon > 78.5 and self.glon < 84.5:
            bnum = "l082"
        elif self.glon > 84.5 and self.glon < 87.5:
            bnum = "l086"
        elif self.glon > 87.5 and self.glon < 90.5:
            bnum = "l089"
        elif self.glon > 349.5 and self.glon < 352.5:
            bnum = "l351"
        elif self.glon > 352.5 and self.glon < 355.5:
            bnum = "l354"
        elif self.glon > 355.5 and self.glon < 358.5:
            bnum = "l357"
        else:
            return("0")
        return("v1.0.2_"+bnum+"_13pca_map50_crop.fits")
    
    
class untitled:
    def __init__(self):
        pass


class untitledTests(unittest.TestCase):
    def setUp(self):
        pass


if __name__ == '__main__':
    unittest.main()
