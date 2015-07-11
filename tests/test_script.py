NEW

from extinction_distance.distance import BaseDistObj

def get_distance2(name,(glon,glat)):
	cloud = BaseDistObj.BaseDistObj(name,(glon,glat),nir_survey="VISTA",cont_survey="ATLASGAL")
	cloud.get_nir_images()
	cloud.get_nir_cat()
	cloud.get_continuum()
	cloud.get_contours(cloud.continuum)
	#This saves the contour and calculates the area
                     #might have to break up this step if contours
                     #need hand editing
	cloud.show_contours_on_threecolor()
	cloud.get_model()
	#use force_completeness=True to re-run the _slow_ completeness step
	cloud.make_photo_catalog()
	cloud.do_distance_estimate()
	        
f = open('ATLASGALsources_coveredby_MALT90.txt','r')
for line in f:
    aid,aname,glon,glat,dum1,dum2,dum3,obs,dum4,dum5,pos,malt90,dum6 = line.split()
    if ((obs != 'notObs') and (float(glon) < 15)):
	name=aid
        print(name,(float(glon),float(glat)))
        try:
            get_distance2(name,(float(glon),float(glat)))
        except:
            pass
#if obs is NOT notObs, then do_something with name=aid and glon and glat
f.close()




cloud = BaseDistObj. BaseDistObj(name,(glon,glat),nir_survey="VISTA",cont_survey="ATLASGAL")


OLD


import extinction_distance.distance.DistanceObject as DistanceObject

current_cloud = DistanceObject.DistanceObject("G23.01-0.41",(23.01,-0.41))
current_cloud.find_bgps()
current_cloud.make_contour()
current_cloud.process_ukidss_data()
current_cloud.make_photo_catalog()
current_cloud.do_distance_estimate()