import DistanceObject

current_cloud = DistanceObject.DistanceObject("G23.01-0.41",(23.01,-0.41))
current_cloud.find_bgps()
current_cloud.make_contour()
current_cloud.process_ukidss_data()
current_cloud.make_photo_catalog()
current_cloud.do_distance_estimate()