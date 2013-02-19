#!/usr/bin/env python
# encoding: utf-8
"""
Re-project a map from equitorial to galactic coordinates.

Optionally trim to make it square.
Optrionally trim to a fixed size.
"""

import montage
import os
import pyfits
import astLib.astCoords as ac
import astLib.astWCS as aw
import subprocess
import montage.status as status

def main():
    coords = {
            "G05.89-0.39":(270.12628955, -24.06794286),
            "G09.62+0.20":(271.55484,-20.527951),
            "G23.01-0.41":(278.66184,-9.0104685),
            "G23.44-0.18":(278.6821,-8.5613784), #Only 100 completness trials
            "G23.66-0.13":(278.72294,-8.2942529),#These coords are different to center box Only 100 completness trials
            "G34.39+0.22":(283.32919,1.44268),
            "G35.20-0.74":(284.55216,1.6536033), #These coords are different #"G35.20-0.74":(284.54894,1.6610633) Only 100 trials
            "G35.20-1.74":(285.41884,1.195691),
            "G49.49-0.39":(290.9276,14.523112),
            "G52.10+1.04":(290.90539,17.48237),#Only 100 completness trials# "G52.10+1.04":(290.93547536,14.50860007),
            "G59.78+0.06":(295.80112,23.73793)#These coords are different to center box Only 100 completness trials
            }
    sizes  = {
        "G05.89-0.39":0.10,
        "G09.62+0.20":0.10,
        "G23.01-0.41":0.08,
        "G23.44-0.18":0.17,
        "G23.66-0.13":0.041,
        "G34.39+0.22":0.14,
        "G35.20-0.74":0.15, #Was 0.125
        "G35.20-1.74":0.14,
        "G49.49-0.39":0.155,
        "G52.10+1.04":0.085,
        "G59.78+0.06":0.06
        }
    maser_positions = {
        "G05.89-0.39":(5.8842,-0.3924),
        "G09.62+0.20":(9.6211,0.1959),
        "G23.01-0.41":(23.00096,-0.4105),
        "G23.44-0.18":(23.4398,-0.1822),
        "G23.66-0.13":(23.6566,-0.1272),
        "G34.39+0.22":(34.3940,0.2215),
        "G35.20-0.74":(35.1970,-0.7431),
        "G35.20-1.74":(35.2002,-1.7364),
        "G49.49-0.39":(49.4884,-0.3879),
        "G52.10+1.04":(52.1005,1.0429),
        "G59.78+0.06":(59.7828,0.0647)
    }


    #source = "G59.78+0.06"
    source = "G23.01-0.41"
    #source = "G34.39+0.22"
    source = "G23.66-0.13"
    source = "G05.89-0.39"
    source = "G59.78+0.06"
    bp     = "/Users/jfoster/Desktop/Current/IRDC_Distances/MasersVisit/Masers/"
    input_sys = "J2000"
    output_sys = "GALACTIC"

    sources = maser_positions.keys()
    #sources = ["G09.62+0.20"]
    for source in sources:
        print(source)
        for filter_name in ["J","H","K"]:
            ffile  = os.path.join(bp,source,source+"_UKIDSS_"+filter_name+".fits")
            #print(ffile)
            #print(sizes[source])
            try:
                do_reprojection(ffile,input_sys,output_sys,size = sizes[source],center=coords[source])
            except:
                print("Failed: "+source+" "+filter_name)
    #get_center_position(h,input_sys,output_sys)
    #montage.mHdr()

def do_reprojection(filename,input_sys,output_sys,size=None,center=None,outfile=None):
    system_lookup= {"GALACTIC":"galactic","J2000":"equatorial"}
    filename_lookup = {"GALACTIC":"GAL","J2000":"EQ"}

    d,h = pyfits.getdata(filename,1,header=True)
    if not size:
        size = get_size(h)
        
    #Center coords always in J2000
    if not center:
        x,y = get_center_position(h,input_sys,"J2000")
    else:
        x,y = ac.convertCoords(input_sys,"J2000",center[0],center[1],2000.)
    pix_size = get_pixel_size(h)*3600. #I do not think I had this factor of 3600. in for 2MASS. Needed for UKIDSS though.
    #print(x,y)
    print(size)
    coord_string = str(x)+" "+str(y)
    print(coord_string)
    print(pix_size)
    #Note that older versions on Montage do not work because of normal split
    #dividing the coordinates. Fix is to use shlex.split instead.
    montage.mHdr(coord_string,size,"Test.hdr",system=system_lookup[output_sys],pix_size=pix_size)
    #print(yo)
    #print(size)
    montage.mSubimage(filename,"Test.fits",x,y,size*2,hdu=1)
    if not outfile: #Try a reasonable guess for output file
        outfile = filename.replace(".fits","_"+filename_lookup[output_sys]+".fits")
    montage.mProject("Test.fits",outfile,"Test.hdr")
    #Yay, this works.

def get_center_position(h,input_sys,output_sys):
    """Get position at center of map, convert to other system."""
    WCS = aw.WCS(h,mode="pyfits")
    xcen = int(h["NAXIS1"]/2.)
    ycen = int(h["NAXIS2"]/2.)

    original_coords = WCS.pix2wcs(xcen,ycen)
    modified_coords = ac.convertCoords(input_sys,output_sys,original_coords[0],original_coords[1],2000.)
    modified_coords = original_coords
    return(modified_coords[0],modified_coords[1])

def get_size(h):
    """Establish a reasonable size for the output image."""
    guess_one = abs(h["NAXIS1"]*h["CDELT1"])
    WCS = aw.WCS(h,mode="pyfits")
    guess_two = WCS.getFullSizeSkyDeg()
    #print(guess_one)
    #print(guess_two)
    return(guess_one)

def get_pixel_size(h):
    #try:
    #    pixel_size = h["CDELT1"]
    #except KeyError:
    #    pixel_size = h["CD1_1"]
    WCS = aw.WCS(h,mode="pyfits")
    pixel_size = WCS.getPixelSizeDeg()
    return(pixel_size)

def make_header():
    pass

if __name__ == '__main__':
    main()
