
# MISR_AS_sampling_vis_09.py
#
# This is a Python 2.7.9 code to read the MISR AGP, L1B2
# nadir files, and aerosol files and generate a mapped images.
#
# Creation Date: 02/11/2015
# Last Modified: 02/11/2015
#
# by Michael J. Garay
# (Michael.J.Garay@jpl.nasa.gov)

# Import packages

from __future__ import print_function # Makes 2.7 behave like 3.3
import fnmatch
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap
import numpy as np
import os
from pyhdf.HDF import *
from pyhdf.SD import SD, SDC
from pyhdf.V import *
from pyhdf.VS import *
import time
import sys

import img_scale # Extra package (img_scale.py needs to be in execution directory)

version =[]
version_names=[]
versions = ['*F12_0022*.hdf','*22b23-56*.hdf','*22b23-64*.hdf', '*USER*22b24-10d.hdf']#, '*22b24-10d.hdf']#,'*22b24-10-17.6km.hdf']
version_names=['V22_17.6','17.6kmBaseline','4.4kmBaseline','4.4currBaseLine']#,'4.4kmGV_ev']#,'EOFs_17.6km']
plt_colours = ['black', 'blue','green','red','yellow','brown']
plt_shapes = ['o','o','o','*','-','^']
plt_count =0
labels=[]
chisq_abs_new = False
# Set the overall timer
all_start_time = time.time()

# Set the start block, number of blocks and output file names
#start_block = 107 # Start block 1-based
#num_block = 1
start_block = 105 # Start block 1-based
num_block = 4
OUTPUT_PLOT = 'composite.png'

# Set the pixel sizes for 1.1 km data
x_size = 512
y_size = 128
# Set the paths

basepath = '/your/directory/MISR/AGP/'
aeropath = '//your/directory/MISR/AEROSOL_VERSIONS/FULL_ORBITS/'
figpath = '/your/directory/MISR/AEROSOL_CODE/KDW/FIGS/'

# Start the timer

start_time = time.time()

# Change directory to the basepath

os.chdir(basepath)

### Get the MISR AGP file for navigation

file_list = glob.glob('MISR*AGP*.hdf')

# Set the filename

inputName = basepath+file_list[0]

# Tell user location in process

print("Reading: "+inputName)

# Open the file

hdf = SD(inputName, SDC.READ)

# Read the data fields

var01 = hdf.select('GeoLatitude')
var02 = hdf.select('GeoLongitude')

lat_raw = var01.get()
lon_raw = var02.get()

# Close the file

hdf.end()

# Print the time

end_time = time.time()
print("Time to Read AGP data was %g seconds" % (end_time - start_time))

### Get the L1B2 Ellipsoid data for the An camera
start_time = time.time()
os.chdir(aeropath)
file_list = glob.glob('*GRP_ELLIPSOID*AN*.hdf')
print(file_list)

#inputName = basepath+file_list[0]
inputName = aeropath+file_list[0]
print ("inputName "+inputName)
print("Reading: "+inputName)

# Get the scale factors from the EOS Grid data
# Note: The reference numbers were found by parsing the full VDATA description
#       using the code MISR_L1B2_NAV_vis_tool_08.py

f = HDF(inputName)
vs = f.vstart()
v = f.vgstart()

# BlueBand
vg = vs.attach(223) # Scale factor
blue_sf_raw = vg.read()
blue_sf = np.squeeze(blue_sf_raw)
vg.detach()

vg = vs.attach(691) # std_solar_wgted_height
blue_E0_raw = vg.read()
vg.detach()

vg = vs.attach(692) # SunDistanceAU
MISR_AU_raw = vg.read()
vg.detach()

# GreenBand

vg = vs.attach(224) # Scale factor
green_sf_raw = vg.read()
green_sf = np.squeeze(green_sf_raw)
vg.detach()

vg = vs.attach(693) # std_solar_wgted_height
green_E0_raw = vg.read()
vg.detach()

# RedBand

vg = vs.attach(225) # Scale factor
red_sf_raw = vg.read()
red_sf = np.squeeze(red_sf_raw)
vg.detach()

vg = vs.attach(695) # std_solar_wgted_height
red_E0_raw = vg.read()
vg.detach()

# NIRBand

vg = vs.attach(226) # Scale factor
nir_sf_raw = vg.read()
nir_sf = np.squeeze(nir_sf_raw)
vg.detach()

vg = vs.attach(697) # std_solar_wgted_height
nir_E0_raw = vg.read()
vg.detach()

v.end()
vs.end()
f.close()

# Open the file

hdf = SD(inputName, SDC.READ)

# Read the data fields

var01 = hdf.select('Blue Radiance/RDQI')
var02 = hdf.select('Green Radiance/RDQI')
var03 = hdf.select('Red Radiance/RDQI')
var04 = hdf.select('NIR Radiance/RDQI')
var05 = hdf.select('BlueConversionFactor')
var06 = hdf.select('GreenConversionFactor')
var07 = hdf.select('RedConversionFactor')
var08 = hdf.select('NIRConversionFactor')
var09 = hdf.select('SolarZenith')

blue_raw = var01.get()
green_raw = var02.get()
red_raw = var03.get()
nir_raw = var04.get()
blue_cf_raw = var05.get()
green_cf_raw = var06.get()
red_cf_raw = var07.get()
nir_cf_raw = var08.get()
sza_raw = var09.get()

# Close the file

hdf.end()

# Print the time

end_time = time.time()
print("Time to Read MISR data was %g seconds" % (end_time - start_time))

# Start the timer

start_time = time.time()

os.chdir(aeropath)

# ### Get the MISR Aerosol File (V22 = 17.6 km operational)
file_list = glob.glob('*F12_0022*.hdf')
inputName = aeropath+file_list[0]
print("Reading: "+inputName)
# get success flag from V22
hdf = SD(inputName, SDC.READ)
var02 = hdf.select('AerRetrSuccFlag')
succ_flag_01 = var02.get()
hdf.end()
### Analyze the success flag
sf_01 = np.copy(succ_flag_01)
sf_01[succ_flag_01 != 7] = 0
sf_01[succ_flag_01 == 7] = 1
print("V22 Baseline Success Flag (All) - Total Count: %g " %np.sum(sf_01))
sf_01_tot = np.sum(sf_01)

# Set the plot area
fig = plt.figure(figsize=(6,6), dpi=120)
ax1 = fig.add_subplot(111)

## Get the MISR Aerosol file data for each version to compare
for each_version in versions:
	print ("each_version ", each_version)
	file_list = glob.glob(each_version)
	inputName = aeropath+file_list[0]
	print("Reading: "+inputName)
	# Open the file
	hdf = SD(inputName, SDC.READ)
	print ("successfully open %s" %inputName)

	# Read the data fields to calc success rate and 
	# determine Chi-Squared Abs vrs AOD plot 
	try:
		#for old var names
		var02 = hdf.select('AerRetrSuccFlag')
		var03 = hdf.select('RegBestEstimateSpectralOptDepth')
		var04 = hdf.select('ChisqAbs')
		var05 = hdf.select('OptDepthPerMixture')

		succ_flag_02 = var02.get()
		sf_02 = np.copy(succ_flag_02)
		sf_02[succ_flag_02 != 7] = 0
		sf_02[succ_flag_02 == 7] = 1

	except:
		#for new var name mappings
		var02 = hdf.select('Retrieval_Success_Flag') 
		var03 = hdf.select('Aerosol_Optical_Depth_555')
		var04 = hdf.select('Chisq_Per_Mixture')
		var05 = hdf.select('Aerosol_Optical_Depth_Per_Mixture')

		succ_flag_02 = var02.get()
		sf_02 = np.copy(succ_flag_02)
		sf_02[succ_flag_02 != 1] = 0

		chisq_abs_new = True

	rbe_aod_02 = var03.get()
	chisq_abs_02 = var04.get()
	mix_aod_02 = var05.get()

	# Close the file
	hdf.end()

	end_time = time.time()
	print("Time to Read Aerosol data was %g seconds" % (end_time - start_time))

	# sf_02 = np.copy(succ_flag_02)
	# sf_02[succ_flag_02 != 7] = 0
	# sf_02[succ_flag_02 == 7] = 1
	# print("%s Success Flag (All) - Total Count: %g" %(each_version, np.sum(sf_02)))
	sf_02_tot = np.sum(sf_02)
	print("'% 'success rate is: ", (sf_02_tot*1.0)/(sf_01_tot*1.0))

	# Extract the navigation information to get corners for the entire image
	lat = lat_raw[start_block-1,:,:]
	lon = lon_raw[start_block-1,:,:]
	lat_max = np.amax(lat)
	lon_max = np.amax(lon)

	lat = lat_raw[start_block-1+num_block-1,:,:]
	lon = lon_raw[start_block-1+num_block-1,:,:]
	lat_min = np.amin(lat)
	lon_min = np.amin(lon)

	## Loop over blocks

	for i in range(num_block):

	    block = start_block + i - 1 # Block 0-based
	    mix_abs = chisq_abs_02[block,:,:,:]*1.
	    # Extract the data

	    if chisq_abs_new == False:
	    	aod_02 = rbe_aod_02[block,:,:,1]*1. # Green band
	    	
	    	
	    else:
	    	aod_02 = rbe_aod_02[block,:,:]*1. # Green band
	    	#mix_abs = chisq_abs_02[block,:,:,:]*1.
	    	#ChisqAbs =  (Dimensioned by mixture [74], so you have to find the minimum)
	    	#I'm not sure whta this means
	    	

	    mix_aod = mix_aod_02[block,:,:,:]*1.

	    aero_y = mix_aod.shape[0]
	    aero_x = mix_aod.shape[1]
	    
	    new_chisq_02 = np.zeros_like(aod_02)
	    new_aod_02 = np.zeros_like(aod_02)
	    
	    print('TESTING')
	    for y in range(aero_y):
	        for x in range(aero_x):
	            hold = mix_abs[y,x,:]
	            test = np.copy(hold)
	            if(np.amin(test) > 0.0):
	                check = np.argmin(test)
	                new_chisq_02[y,x] = np.amin(test)
	                temp = mix_aod[y,x,:]
	                new_aod_02[y,x] = temp[check]
	    
	    if(i > 0):
	        chisq_all_02 = np.append(chisq_all_02,new_chisq_02)
	        aod_all_02 = np.append(aod_all_02,new_aod_02)
	    else:
	        chisq_all_02 = np.copy(new_chisq_02)
	        aod_all_02 = np.copy(new_aod_02)

	# Plot the data        
	ax1.scatter(aod_all_02,chisq_all_02,marker=plt_shapes[plt_count],
		color=plt_colours[plt_count],s=2, label=version_names[plt_count]) 

	plt_count += 1 
	aod_all_02 =[]
	chisq_all_02 =[]


# Set the plot axes
max_x = 0.20
tick_x = 0.05
max_y = 4.0
tick_y = 0.5

plt.axis([0,max_x,0,max_y])
plt.yticks(np.arange(0,max_y+tick_y,tick_y))
plt.xticks(np.arange(0,max_x+tick_x,tick_x))
plt.xlabel('Green Band AOD')
plt.ylabel('Chi-squared Abs')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          fancybox=True, shadow=True, ncol=len(versions))

# Save the figure
plt.savefig(figpath+OUTPUT_PLOT,dpi=120)

# Show the plot
plt.show()
