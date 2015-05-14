# MISR_AS_V22_DRAGON_01.py
#
# This is a Python 2.7.9 code to read the MISR AGP, L1B2
# nadir files, and aerosol files, generate a mapped images,
# and regressions for the 2013-01-20 PODEX DRAGON case.
# This code is based on the IDL code MISR_PODEX_redux_tool_09.pro
#
# Creation Date: 2015-05-13
# Last Modified: 2015-05-13
#
# by Michael J. Garay
# (Michael.J.Garay@jpl.nasa.gov)

# Import packages

from __future__ import print_function # Makes 2.7 behave like 3.3
import fnmatch
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import os
from pyhdf.HDF import *
from pyhdf.SD import SD, SDC
from pyhdf.V import *
from pyhdf.VS import *
import time

import img_scale # Extra package (img_scale.py needs to be in execution directory)

def main():  # Main code

# Set the overall timer

    all_start_time = time.time()

# Set the pixel sizes for 1.1 km data

    x_size = 512
    y_size = 128

# Set the minimum and maximum AOD for plotting

    aod_plot_min = 0.00
    aod_plot_max = 0.30
    aod_plot_ticks = 7 # Usually 1 more than you think
    aod_plot_step = 0.05

# Set the paths

    agppath = '/Volumes/ChoOyu/DATA/AGP/'
    aeropath = '/Volumes/ChoOyu/DATA/2013_01_20/MISR/'
    figpath = '/Users/mgaray/Desktop/CODING/PYTHON/PY27/MAY15/AEROSOL/FIGS/'

# Set the MISR product

    misr_name = '0022'  # 17.6 km Standard Product

# Set the MISR orbit

    misr_path = 'P042'
    misr_orbit = '69644'

# Set the block range

    start_block = 60 # Start block 1-based
    num_block = 4
    out_base = '_O'+misr_orbit+'_{}'.format(start_block)
    out_base = out_base+'_{}'.format(num_block)+'_'+misr_name+'_DRAGON_01.png'

# Set the AERONET lat, lon, and values
# NOTE: These are taken from the AERONET_colocate_to_MISR.csv file

    aero_lat = [34.137,35.238,35.332,36.819,36.102,36.706,36.785,36.316,
        36.206,36.953,36.597,36.032,35.504,36.634,36.314,36.785]
    
    aero_lon = [-118.126,-118.788,-119.000,-119.716,-119.566,-119.741,-119.773,
        -119.643,-120.105,-120.034,-119.504,-119.055,-119.272,-120.382,-119.393,
        -119.773]
        
    aero_aod = [0.009,0.212,0.231,0.111,0.226,0.130,0.120,0.208,0.159,0.093,
        0.170,0.149,0.201,0.165,0.215,0.114]

### Read the AGP Data for Navigation of Old-Style Data

# Start the timer

    start_time = time.time()

# Change directory to the AGP-path

    os.chdir(agppath)

# Search for the correct MISR path
    
    search_str = 'MISR*'+misr_path+'*.hdf'
    file_list = glob.glob(search_str)

# Set the filename

    inputName = file_list[0]

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

# Extract the navigation information to get corners for the entire image

    lat = lat_raw[start_block-1,:,:]
    lon = lon_raw[start_block-1,:,:]
    lat_max = np.amax(lat)
    lon_max = np.amax(lon)

    lat = lat_raw[start_block-1+num_block-1,:,:]
    lon = lon_raw[start_block-1+num_block-1,:,:]
    lat_min = np.amin(lat)
    lon_min = np.amin(lon)
        
### Read the V22 Data (Old-Style)

# Start the timer

    start_time = time.time()

# Change directory to the basepath

    os.chdir(aeropath)

### Get the first MISR Aerosol File

    search_str = 'MISR_AM1_AS_AEROSOL*'+misr_orbit+'*'+misr_name+'.hdf'

    file_list = glob.glob(search_str)

# Set the filename

    inputName = file_list[0]

# Tell user location in process

    print("Reading: "+inputName)

# Open the file

    hdf = SD(inputName, SDC.READ)

# Read two data fields

    var01 = hdf.select('AlgTypeFlag')
    var02 = hdf.select('AerRetrSuccFlag')
    var03 = hdf.select('RegBestEstimateSpectralOptDepth')
    var04 = hdf.select('RegLowestResidSpectralOptDepth')

    alg_type_v22 = var01.get()
    succ_flag_v22 = var02.get()
    rbe_aod_v22 = var03.get()
    rlr_aod_v22 = var04.get()

# Close the file

    hdf.end()    

# Print the time

    end_time = time.time()
    print("Time to Read Aerosol data was %g seconds" % (end_time - start_time))

# Process the success flag as a mask (set success to 1, otherwise 0)

    sf_v22 = np.copy(succ_flag_v22)
    sf_v22[succ_flag_v22 != 7] = 0
    sf_v22[succ_flag_v22 == 7] = 1
    
## MAP THE BEST ESTIMATE DATA

# Set the plot area

    fig = plt.figure(figsize=(12,6), dpi=120)

# Set the title

    plt.title('V22 Best Estimate AOD')

# Draw basemap

    m = Basemap(llcrnrlon=lon_min,llcrnrlat=lat_min,urcrnrlon=lon_max,urcrnrlat=lat_max,
        projection='cyl',resolution='i')
    m.drawmapboundary(fill_color='0.3')

## Loop over blocks

    for i in range(num_block):

        block = start_block + i - 1 # Block 0-based

# Extract the data

        lat = lat_raw[block,:,:]
        lon = lon_raw[block,:,:]
        aod_01 = rbe_aod_v22[block,:,:,1]*1. # Green band
        succ_01 = sf_v22[block,:,:]*1.
        keep = (succ_01 > 0.0)
        print('V22')
        print('Mean AOD = ',np.mean(aod_01[keep]))
        print(' Max AOD = ',np.amax(aod_01[keep]))

# Mask locations without valid retrievals

        mask = (succ_01 < 1.0)
        aod_01[mask] = 0.0
        
# Resize the navigation from 1.1 km to 4.4 km (to match new product)

        lat_small = np.squeeze(lat.reshape([32,4,128,4]).mean(3).mean(1))
        lon_small = np.squeeze(lon.reshape([32,4,128,4]).mean(3).mean(1))

# Resize the data to match the navigation
# Note: The factor is 4 in both dimensions (17.6 -> 4.4 km)

        aod_full = np.repeat(np.repeat(aod_01,4,axis=0),4,axis=1)

# Plot the MISR data

        img = aod_full

#        im = m.pcolormesh(lon,lat,img,shading='flat',cmap=plt.cm.hot_r,latlon=True,
#            vmin=aod_plot_min,vmax=aod_plot_max)

        im = m.pcolormesh(lon_small,lat_small,img,shading='flat',
            cmap=plt.cm.CMRmap_r,latlon=True,vmin=aod_plot_min,vmax=aod_plot_max)

# Plot the AERONET points

        m.scatter(aero_lon,aero_lat,marker='o',c=aero_aod,
            cmap=plt.cm.CMRmap_r,vmin=aod_plot_min,vmax=aod_plot_max,s=25)

# Draw latitude lines

# m.drawparallels(np.arange(4.,13.,1.),labels=[0,1,0,0])

# Add the coastlines

#coast_color = 'blue'
    coast_color = 'green'
    m.drawcoastlines(color=coast_color,linewidth=0.8)

# Add the colorbar

#    cb = m.colorbar(im,"bottom", size="10%", pad="5%", 
#        ticks=[0,0.05,0.10,0.15,0.20])
    cb = m.colorbar(im,"bottom", size="10%", pad="5%", 
        ticks=np.arange(aod_plot_ticks)*aod_plot_step)
        
    cb.set_label('Green Band AOD')

# Save the figure

    os.chdir(figpath)
    outname = 'AOD_BE_Map'+out_base
    plt.savefig(outname,dpi=120)
    
## MAP THE LOWEST RESIDUAL DATA

# Set the plot area

    fig = plt.figure(figsize=(12,6), dpi=120)

# Set the title

    plt.title('V22 Lowest Residual AOD')

# Draw basemap

    m = Basemap(llcrnrlon=lon_min,llcrnrlat=lat_min,urcrnrlon=lon_max,urcrnrlat=lat_max,
        projection='cyl',resolution='i')
    m.drawmapboundary(fill_color='0.3')

## Loop over blocks

    for i in range(num_block):

        block = start_block + i - 1 # Block 0-based

# Extract the data

        lat = lat_raw[block,:,:]
        lon = lon_raw[block,:,:]
        aod_01 = rlr_aod_v22[block,:,:,1]*1. # Green band
        succ_01 = sf_v22[block,:,:]*1.
        keep = (succ_01 > 0.0)
        print('V22')
        print('Mean AOD = ',np.mean(aod_01[keep]))
        print(' Max AOD = ',np.amax(aod_01[keep]))

# Mask locations without valid retrievals

        mask = (succ_01 < 1.0)
        aod_01[mask] = 0.0
        
# Resize the navigation from 1.1 km to 4.4 km (to match new product)

        lat_small = np.squeeze(lat.reshape([32,4,128,4]).mean(3).mean(1))
        lon_small = np.squeeze(lon.reshape([32,4,128,4]).mean(3).mean(1))

# Resize the data to match the navigation
# Note: The factor is 4 in both dimensions (17.6 -> 4.4 km)

        aod_full = np.repeat(np.repeat(aod_01,4,axis=0),4,axis=1)

# Plot the MISR data

        img = aod_full

#        im = m.pcolormesh(lon,lat,img,shading='flat',cmap=plt.cm.hot_r,latlon=True,
#            vmin=aod_plot_min,vmax=aod_plot_max)

        im = m.pcolormesh(lon_small,lat_small,img,shading='flat',
            cmap=plt.cm.CMRmap_r,latlon=True,vmin=aod_plot_min,vmax=aod_plot_max)

# Plot the AERONET points

        m.scatter(aero_lon,aero_lat,marker='o',c=aero_aod,
            cmap=plt.cm.CMRmap_r,vmin=aod_plot_min,vmax=aod_plot_max,s=25)

# Draw latitude lines

# m.drawparallels(np.arange(4.,13.,1.),labels=[0,1,0,0])

# Add the coastlines

#coast_color = 'blue'
    coast_color = 'green'
    m.drawcoastlines(color=coast_color,linewidth=0.8)

# Add the colorbar

#    cb = m.colorbar(im,"bottom", size="10%", pad="5%", 
#        ticks=[0,0.05,0.10,0.15,0.20])
    cb = m.colorbar(im,"bottom", size="10%", pad="5%", 
        ticks=np.arange(aod_plot_ticks)*aod_plot_step)
        
    cb.set_label('Green Band AOD')

# Save the figure

    os.chdir(figpath)
    outname = 'AOD_LR_Map'+out_base
    plt.savefig(outname,dpi=120)    

## CALCULATE AND PLOT THE REGRESSIONS AGAINST AERONET

# Extract the full (1.1 km) resolution navigation for all the blocks

    lat_all_full = lat_raw[(start_block-1):(start_block-1+num_block),:,:]
    lon_all_full = lon_raw[(start_block-1):(start_block-1+num_block),:,:]

# Rebin the navigation data to 4.4 km resolution

    lat_4 = np.squeeze(lat_all_full.reshape([num_block,1,32,4,
        128,4]).mean(5).mean(3))
    lon_4 = np.squeeze(lon_all_full.reshape([num_block,1,32,4,
        128,4]).mean(5).mean(3))

# Extract the full (17.6 km) resolution aerosol data for all blocks
        
    rbe_all_full = rbe_aod_v22[(start_block-1):(start_block-1+num_block),:,:,1]*1.
    rlr_all_full = rlr_aod_v22[(start_block-1):(start_block-1+num_block),:,:,1]*1.
    succ_all_full = sf_v22[(start_block-1):(start_block-1+num_block),:,:]*1.

# Rebin the aerosol data to 4.4 km resolution

    rbe_raw_4 = np.repeat(np.repeat(rbe_all_full,4,axis=1),4,axis=2)
    rlr_raw_4 = np.repeat(np.repeat(rlr_all_full,4,axis=1),4,axis=2)
    succ_4 = np.repeat(np.repeat(succ_all_full,4,axis=1),4,axis=2)

# Eliminate locations without a valid retrieval

    rbe_4 = rbe_raw_4 * succ_4
    rlr_4 = rlr_raw_4 * succ_4

# Get the number of AERONET sites
    
    num_aero = len(aero_aod)

# Set up arrays to store the matched data

    aero_match = np.zeros(num_aero)
    rbe_v22_match = np.zeros(num_aero)
    rlr_v22_match = np.zeros(num_aero)
    dist_match = np.zeros(num_aero)

# Loop over AERONET retrievals

    for i in range(num_aero):

# Extract AERONET information
    
        tlat = aero_lat[i]
        tlon = aero_lon[i]
        taod = aero_aod[i]

# Calculate the array of distances (in km) from the AERONET point vs. the MISR data
        
        dist = Haversine_Distance(tlat,tlon,lat_4,lon_4)
        
# Find the minimum distance (km)

        min_dist = np.amin(dist)
    
# Test for a match within a single (4.4 km) MISR pixel

        if(min_dist <= 4.4):

# Extract the match

            match = (dist == min_dist)   
            
# Extract the MISR data

            aero_match[i] = taod
            rbe_v22_match[i] = rbe_4[match]
            rlr_v22_match[i] = rlr_4[match]
            dist_match[i] = min_dist

# Plot the data

    max_val = aod_plot_max

# Set the plot area
# NOTE: The base plot size is 6 x 6, so a 2 row, 3 column set would be 18 x 12

    plt.figure(figsize=(12,6), dpi=120)

## Linear plot (Best Estimate)

    good = (rbe_v22_match > 0)
    ref_aod = aero_match[good]
    test_aod = rbe_v22_match[good]

    plt.subplot(1, 2, 1)
    plt.scatter(ref_aod,test_aod,marker='o',color='black',s=25)   
    plt.title("V22 Best Estimate")

# Plot the one-to-one line

    plt.plot([0.0,max_val], [0.0,max_val], color="k", lw=1)

# Plot the envelopes

    dummy_aod = np.logspace(-4,1,num=100)
    up1_aod = 1.20*dummy_aod
    up2_aod = dummy_aod+0.05
    upper_aod = np.maximum(up1_aod,up2_aod)

    lo1_aod = 0.80*dummy_aod
    lo2_aod = dummy_aod-0.05
    lower_aod = np.minimum(lo1_aod,lo2_aod)

    plt.plot(dummy_aod,lower_aod,color="0.75", lw=1)
    plt.plot(dummy_aod,upper_aod,color="0.75", lw=1)

# Set the limits and axis labels

    plt.xlim(0.0,max_val)
    plt.ylim(0.0,max_val)

    plt.xlabel('AERONET AOD')
    plt.ylabel('MISR AOD')

    plt.grid(True)

# Include some text on the Best Estimate Figure

    x_pos = 0.19

    plt.text(x_pos,0.08,'Best Estimate',fontsize=12) # Version

    count = len(test_aod)
    out_text = 'N = '+str(count)
    plt.text(x_pos,0.07,out_text,fontsize=10) # Count

    temp = np.corrcoef(ref_aod,test_aod)
    be_r = temp[0,1]
    out_text = 'r = '+"{0:.4f}".format(be_r)
    plt.text(x_pos,0.06,out_text,fontsize=10) # Correlation coefficient

    rmse = np.sqrt(((test_aod - ref_aod) ** 2).mean())
    out_text = 'RMSE = '+"{0:.4f}".format(rmse)
    plt.text(x_pos,0.05,out_text,fontsize=10) # Root mean squared error

    diff = test_aod - ref_aod
    bias = np.mean(diff)
    out_text = 'Bias = '+"{0:.4f}".format(bias)
    plt.text(x_pos,0.04,out_text,fontsize=10) # Bias

    offset = np.ones_like(ref_aod)*0.05
    inner = np.absolute(diff) < np.maximum(offset,ref_aod*0.2)
    in_frac = (np.sum(inner)/(1.0*count))*100.0
    out_text = 'Percent In = '+"{0:.2f}".format(in_frac)
    plt.text(x_pos,0.03,out_text,fontsize=10) # Percent in envelope

## Linear plot (Lowest Residual)

    good = (rlr_v22_match > 0)
    ref_aod = aero_match[good]
    test_aod = rlr_v22_match[good]

    plt.subplot(1, 2, 2)
    plt.scatter(ref_aod,test_aod,marker='o',color='black',s=25)   
    plt.title("V22 Lowest Residual")

# Plot the one-to-one line

    plt.plot([0.0,max_val], [0.0,max_val], color="k", lw=1)

# Plot the envelopes

    dummy_aod = np.logspace(-4,1,num=100)
    up1_aod = 1.20*dummy_aod
    up2_aod = dummy_aod+0.05
    upper_aod = np.maximum(up1_aod,up2_aod)

    lo1_aod = 0.80*dummy_aod
    lo2_aod = dummy_aod-0.05
    lower_aod = np.minimum(lo1_aod,lo2_aod)

    plt.plot(dummy_aod,lower_aod,color="0.75", lw=1)
    plt.plot(dummy_aod,upper_aod,color="0.75", lw=1)

# Set the limits and axis labels

    plt.xlim(0.0,max_val)
    plt.ylim(0.0,max_val)

    plt.xlabel('AERONET AOD')
    plt.ylabel('MISR AOD')

    plt.grid(True)

# Include some text on the Best Estimate Figure

    x_pos = 0.19

    plt.text(x_pos,0.08,'Lowest Resid',fontsize=12) # Version

    count = len(test_aod)
    out_text = 'N = '+str(count)
    plt.text(x_pos,0.07,out_text,fontsize=10) # Count

    temp = np.corrcoef(ref_aod,test_aod)
    be_r = temp[0,1]
    out_text = 'r = '+"{0:.4f}".format(be_r)
    plt.text(x_pos,0.06,out_text,fontsize=10) # Correlation coefficient

    rmse = np.sqrt(((test_aod - ref_aod) ** 2).mean())
    out_text = 'RMSE = '+"{0:.4f}".format(rmse)
    plt.text(x_pos,0.05,out_text,fontsize=10) # Root mean squared error

    diff = test_aod - ref_aod
    bias = np.mean(diff)
    out_text = 'Bias = '+"{0:.4f}".format(bias)
    plt.text(x_pos,0.04,out_text,fontsize=10) # Bias

    offset = np.ones_like(ref_aod)*0.05
    inner = np.absolute(diff) < np.maximum(offset,ref_aod*0.2)
    in_frac = (np.sum(inner)/(1.0*count))*100.0
    out_text = 'Percent In = '+"{0:.2f}".format(in_frac)
    plt.text(x_pos,0.03,out_text,fontsize=10) # Percent in envelope

# Save the figure

    os.chdir(figpath)
    outname = 'AOD_Regression'+out_base
    plt.savefig(outname,dpi=120)

# Show the plot

    plt.show()
    
# Print the time

    all_end_time = time.time()
    print("Total elapsed time was %g seconds" % (all_end_time - all_start_time))

# Tell user completion was successful

    print("\nSuccessful Completion\n")

### END MAIN FUNCTION


def Haversine_Distance(lat1,lon1,lat_arr,lon_arr):
### Distance Calculation Based on the Haversine Formula
# Creation Date: 2015-05-13
# Last Modified: 2015-05-13
# By Michael J. Garay
# Michael.J.Garay@jpl.nasa.gov
#
# Note: This follows the formula from http://williams.best.vwh.net/avform.htm#Dist
# But see a discussion on the Earth-radius at 
# http://www.cs.nyu.edu/visual/home/proj/tiger/gisfaq.html
#
# Input: lat1 = First latitude, single element(degrees)
#        lon1 = First longitude, single element
#        lat2 = Second latitude, array of values
#        lon2 = Second longitude, array of values
#
# Output: Returns an array of distances (km)

# Convert lat/lon to radians

    rat1 = lat1*np.pi/180.0
    ron1 = lon1*np.pi/180.0
    
    rat2 = lat_arr*np.pi/180.0
    ron2 = lon_arr*np.pi/180.0

# Calculate the distance using the Haversine Formula

    d = 2.0*np.arcsin(np.sqrt((np.sin((rat2-rat1)/2))**2 +
      np.cos(rat2)*np.cos(rat1)*(np.sin((ron2-ron1)/2))**2))

# Convert to kilometers
    
    dist = 6371.0 * d
    
    return dist

### END Haversine_Distance


if __name__ == '__main__':
    main()
