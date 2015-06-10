# MISR_AS_V22b24_DRAGON_04.py
#
# This is a Python 2.7.10 code to read the MISR AGP
# and aerosol files, then generate a mapped images,
# and regressions for DRAGON cases.
#
# Creation Date: 2015-06-04
# Last Modified: 2015-06-04
#
# by Michael J. Garay
# (Michael.J.Garay@jpl.nasa.gov)

# Import packages

from __future__ import print_function # Makes 2.7 behave like 3.3
from astropy.time import Time
from datetime import datetime
import fnmatch
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import numpy as np
import os
from pyhdf.HDF import *
from pyhdf.SD import SD, SDC
from pyhdf.V import *
from pyhdf.VS import *
import time

def main():  # Main code

# Set the file exist (0 to start, 1 after initial run)
# NOTE: The NetCDF filename is hardcoded

#    exist = 0
    exist = 1

# Set the overall timer

    all_start_time = time.time()

# Set the pixel sizes for 4.4 km data
# Note: The MISR base (1.1 km) resolution is 512-x by 128-y

#    x_size = 512
#    y_size = 128
    
    x_44 = 128
    y_44 = 32    

# Set the time window

    time_window = 30  #  Time in minutes     

# Set the minimum and maximum AOD for plotting

    aod_plot_min = 0.00
    
#    aod_plot_max = 0.30
#    aod_plot_max = 0.60
    aod_plot_max = 1.40
    
#    aod_plot_ticks = 7 # Usually 1 more than you think
    aod_plot_ticks = 8 # Usually 1 more than you think
    
#    aod_plot_step = 0.05
#    aod_plot_step = 0.1
    aod_plot_step = 0.2

# Set the paths

    agppath = '/Volumes/ChoOyu/DATA/AGP/'
    aeropath = '/Volumes/ChoOyu/DATA/'
    datapath = '/Volumes/ChoOyu/DATA/AERONET/'
    figpath = '/Users/mgaray/Desktop/CODING/PYTHON/PY27/JUN15/AEROSOL/FIGS/'

# Set the MISR product

#    misr_name = '22b24-10e' # New Product
#    misr_name = '22b24-26' # New Product
#    misr_name = '22b24-26+1' # New Product
#    misr_name = '22b24-26+2' # New Product
#    misr_name = '22b24-27+3' # New Product
#    misr_name = '22b24-29' # New Product
#    misr_name = '22b24-29+1' # New Product
#    misr_name = '22b24-29+3' # New Product
#    misr_name = '22b24-34+0' # New Product
    misr_name = '22b24-34+1' # New Product
#    misr_name = '22b24-34+2' # New Product
#    misr_name = '22b24-34+3' # New Product

# Set the output file

    outfile = 'MISR_AERONET_DRAGON_V22b24-34+1_04.nc'

# Set the MISR orbit, path, and date information to test
# NOTE: Although this is redundant, it provides a check so that the correct
#       data are analyzed.

#    misr_orbit = '60934'
#    misr_orbit = '61633'
#    misr_orbit = '61662'
#    misr_orbit = '65440'
#    misr_orbit = '65731'
#    misr_orbit = '65775'
#    misr_orbit = '65906'
#    misr_orbit = '66139'
#    misr_orbit = '69644'
    misr_orbit = '69877'

#    misr_path = 'P016'
#    misr_path = 'P016'    
#    misr_path = 'P014'
#    misr_path = 'P115'
#    misr_path = 'P111'
#    misr_path = 'P116'
#    misr_path = 'P115'
#    misr_path = 'P115'
#    misr_path = 'P042'
    misr_path = 'P042'
    
#    misr_date = '2011_06_02'
#    misr_date = '2011_07_20'
#    misr_date = '2011_07_22'
#    misr_date = '2012_04_07'
#    misr_date = '2012_04_27'
#    misr_date = '2012_04_30'
#    misr_date = '2012_05_09'
#    misr_date = '2012_05_25'
#    misr_date = '2013_01_20'
    misr_date = '2013_02_05'

    start_block = 60 # MISR start block (1-based)
    end_block = 63 # MISR end block (1-based)

# Update the aerosol path

    aeropath = aeropath+misr_date+'/MISR/'

# Set the base file name

    num_block = end_block - start_block + 1 # Inclusive
    out_base = '_O'+misr_orbit+'_{}'.format(start_block)
    out_base = out_base+'_{}'.format(num_block)+'_'+misr_name+'_DRAGON_04.png'
            
### Read the aerosol data (New-Style)

# Start the timer

    start_time = time.time()

# Change directory to the basepath

    os.chdir(aeropath)

### Get the MISR Aerosol File

    search_str = 'MISR_AM1_AS_AEROSOL_USER_*'+misr_orbit+'*'+misr_name+'.'+'*.hdf'

    file_list = glob.glob(search_str)
    
# Choose the correct filename

    inputName = file_list[0]

# Tell user location in process

    print("Reading: "+inputName)

# Open the file

    hdf = SD(inputName, SDC.READ)

# Read two data fields

    var01 = hdf.select('Latitude')
    var02 = hdf.select('Longitude')
    var03 = hdf.select('TAI_Time')
    var04 = hdf.select('Land_Water_Retrieval_Type_Flag')
    var05 = hdf.select('Retrieval_Success_Flag')
    var06 = hdf.select('Aerosol_Optical_Depth_555')
    var07 = hdf.select('Aerosol_Optical_Depth_Per_Mixture')
    var08 = hdf.select('Chisq_Per_Mixture')

    lat_raw = var01.get()
    lon_raw = var02.get()
    tai_raw = var03.get()
    alg_type = var04.get()
    succ_flag = var05.get()
    rbe_aod = var06.get()
    aod_per_mix = var07.get()
    chisq_per_mix = var08.get()

# Close the file

    hdf.end()    

# Print the time

    end_time = time.time()
    print("Time to Read Aerosol data was %g seconds" % (end_time - start_time))

# Get the dimensions of the aerosol data

    b_aero = rbe_aod.shape[0]
    y_aero = rbe_aod.shape[1]
    x_aero = rbe_aod.shape[2]

# Process the success flag as a mask (if successful set success to 1, otherwise 0)

    success = np.copy(succ_flag)
    success[succ_flag != 1] = 0

# Extract the navigation information to get corners for the entire image
# Note: Because the MISR block numbers are 1-based, but the array indices are 0-based
#       the initial block should be start_block-1.  Because the number of blocks is
#       inclusive, the end block must be start_block-1 + num_block-1, otherwise an
#       additional block with be extracted.

    lat = lat_raw[start_block-1,:,:]
    lon = lon_raw[start_block-1,:,:]
    lat_max = np.amax(lat)
    lon_max = np.amax(lon)

    lat = lat_raw[start_block-1+num_block-1,:,:]
    lon = lon_raw[start_block-1+num_block-1,:,:]
    lat_min = np.amin(lat)
    lon_min = np.amin(lon)    
    
### Get the MISR time information
# Note: Because the MISR block numbers are 1-based, but the array indices are 0-based
#       the initial block should be start_block-1.  Because the number of blocks is
#       inclusive, the end block must be start_block-1 + num_block-1, otherwise an
#       additional block with be extracted.

# Start block

    bct_block = tai_raw[start_block-1]
    bct_raw = bct_block[16,64]  # Choose the central pixel in the block (arbitrary)

# The epoch defined by GPS in the astropy library and the one MISR uses for TAI
# are different, so calculate the offset and add it to the time    
    
    t_gps = Time('1980-01-06 00:00:00')
    t_tai = Time('1993-01-01 00:00:00')
    dt = t_tai - t_gps

# Convert the TAI time into the astropy format
    
    t = Time(bct_raw, format='gps', scale='tai')
    t2 = t + dt

# Output the result
# NOTE: The correct format is utc.iso determined by comparing to the old
#       BlockCenterTime
    
    print(t2.utc.iso)
    
    bct = t2.utc.iso # Similar format to MISR BlockCenterTime
    
    words = bct.split(' ')
    temp = words[0].split('-')
    myear_str = temp[0]
    myear = int(temp[0])
    mmonth = int(temp[1])
    mday = int(temp[2])

    temp = words[1].split(':')
    mhour = int(temp[0])
    mminute = int(temp[1])
    hold = temp[2].split('.')
    msecond = int(hold[0])  #  Note: Not rounding here to nearest second
    
    mhours = float(mhour) + mminute/60. + msecond/3600.
    
    mtime1 = mmonth*10000. + mday*100. + mhours # Eliminate years

# End block
# NOTE: To get agreement with the V22 code, need num_block-2, which suggests that
#       the V22 code is incorrect.

    bct_block = tai_raw[start_block-1 + num_block-2]
    bct_raw = bct_block[16,64]  # Choose the central pixel in the block (arbitrary)

# Convert the TAI time into the astropy format
    
    t = Time(bct_raw, format='gps', scale='tai')
    t2 = t + dt

# Output the result
# NOTE: The correct format is utc.iso determined by comparing to the old
#       BlockCenterTime
    
    print(t2.utc.iso)
    
    bct = t2.utc.iso # Similar format to MISR BlockCenterTime
    
    words = bct.split(' ')
    temp = words[0].split('-')
    myear = int(temp[0])
    mmonth = int(temp[1])
    mday = int(temp[2])

    temp = words[1].split(':')
    mhour = int(temp[0])
    mminute = int(temp[1])
    hold = temp[2].split('.')
    msecond = int(hold[0])  #  Note: Not rounding here to nearest second
    
    mhours = float(mhour) + mminute/60. + msecond/3600.
    
    mtime2 = mmonth*10000. + mday*100. + mhours # Eliminate years

# Calculate the average time
    
    mtime = (mtime1 + mtime2)/2.0 # Calculate average time

# Calculate in time window units    
    
    munits = mtime * (60./time_window) 

### Find and read the correct AERONET NetCDF file

# Set the AERONET filename

    aero_name = "AERONET_"+myear_str+".nc"
    
### READ THE AERONET DATA

    start_time = time.time()

# Change directory to the basepath and get the file list

    os.chdir(datapath)
    file_list = glob.glob(aero_name)

# Choose the first file

    inputName = file_list[0]

# Tell user location in process

    print('Reading: ',inputName)

# Open the NetCDF file

    rootgrp = Dataset(inputName, 'r', format='NETCDF4')

# Assign the variables to an array

    site_name = rootgrp.variables['Site'][:]
    lon = rootgrp.variables['Longitude'][:]
    lat = rootgrp.variables['Latitude'][:]

    year = rootgrp.variables['Year'][:]
    month = rootgrp.variables['Month'][:]
    day = rootgrp.variables['Day'][:]

    hour = rootgrp.variables['Hour'][:]
    minute = rootgrp.variables['Minute'][:]
    second = rootgrp.variables['Second'][:]

    green_poly = rootgrp.variables['MISR_Green_Poly'][:]
    green_line = rootgrp.variables['MISR_Green_Line'][:]

# Close the NetCDF file

    rootgrp.close()

# Print the time

    end_time = time.time()
    print("Time to Read AERONET data was %g seconds" % (end_time - start_time))

### Convert the AERONET times into time units
### Note: This is a fast approach that does not handle day boundaries correctly

    start_time = time.time()

# Convert to hours

    ahours = hour*1. + minute/60. + second/3600.
    atime = month*10000. + day*100. + ahours # Eliminate years

# Convert to time units

    aunits = atime * (60./time_window)

# Print the time

    end_time = time.time()
    print("Time to Convert AERONET data was %g seconds" % (end_time - start_time))

### Match all the data within the image boundary within the time window

    tdiff = abs(aunits - munits)
    
# Extract those within the window (unit value)
    
    found = (tdiff <= 1.0)
    if(np.amax(found) == True):
        aunits_found = aunits[found]
        site_found = site_name[found]
        lon_found = lon[found]
        lat_found = lat[found]
        poly_found = green_poly[found]
        line_found = green_line[found]
        tdiff_found = tdiff[found]
        
# Find the data within the window (lat/lon)

        t1 = (lon_found <= lon_max)
        t2 = (lon_found >= lon_min)
        t3 = (lat_found <= lat_max)
        t4 = (lat_found >= lat_min)
        
        t12 = np.logical_and(t1,t2)
        t34 = np.logical_and(t3,t4)
        check = np.logical_and(t12,t34)
        
        if(np.amax(check) == True):

# Extract the data within the window (block lat/lon)
            
            keep = (check == True)
            
            site_near = site_found[keep]
            lon_near = lon_found[keep]
            lat_near = lat_found[keep]
            poly_near = poly_found[keep]
            line_near = line_found[keep] 
            tdiff_near = tdiff_found[keep]      

# Get the number of unique AERONET locations
# Note: The AERONET stations do not move

            num_unique, indices = np.unique(lon_near, return_index=True)

# Set up arrays to store the matched data

            num_aero = len(indices)

            aero_lat = np.zeros(num_aero)
            aero_lon = np.zeros(num_aero)
            aero_aod = np.zeros(num_aero)
            aero_line = np.zeros(num_aero)
            aero_tdiff = np.zeros(num_aero)
            
            aero_count = 0

# Loop over the unique locations

            for inner in indices:
                
                test_site = site_near[inner]
                test = (site_near == test_site)
                test_lat = lat_near[test]
                test_lon = lon_near[test]
                test_aod = poly_near[test]
                test_line = line_near[test]
                test_tdiff = tdiff_near[test]
                
                min_time = np.amin(test_tdiff)
                match = (test_tdiff == min_time)
                
                aero_lat[aero_count] = test_lat[match]
                aero_lon[aero_count] = test_lon[match]
                aero_aod[aero_count] = test_aod[match]
                aero_line[aero_count] = test_line[match]
                aero_tdiff[aero_count] = min_time
                
                aero_count = aero_count+1
                
# Print some results before plotting

    print()
    print("Number AERONET Sites = ",aero_count)
    out_text = "MAX AERONET AOD = "+"{0:.4f}".format(np.amax(aero_aod))    
    print(out_text)
    print()
        
## MAP THE BEST ESTIMATE DATA

# Set the plot area

    fig = plt.figure(figsize=(12,6), dpi=120)

# Set the title

    plt.title('V'+misr_name+' Best Estimate AOD')
    
# Draw basemap with the appropriate lat/lon boundaries

    m = Basemap(llcrnrlon=lon_min,llcrnrlat=lat_min,urcrnrlon=lon_max,urcrnrlat=lat_max,
        projection='cyl',resolution='i')
    m.drawmapboundary(fill_color='0.3')

## Loop over blocks

    print()
    print("***Best Estimate***")

    for i in range(num_block):

        block = start_block + i - 1 # Block 0-based

# Extract the data for this block

        lat_44 = lat_raw[block,:,:]
        lon_44 = lon_raw[block,:,:]
        aod_01 = rbe_aod[block,:,:]*1.
        succ_01 = success[block,:,:]*1.
        keep = (succ_01 > 0.0)
 
# Test for a successful search        
        
        if(np.amax(keep) == False):
            continue 
        
# Print some information

        print("Block = ",block+1) # MISR-format
        out_text = "   Min AOD = "+"{0:.4f}".format(np.amin(aod_01[keep])) 
        print(out_text)    
        out_text = "Median AOD = "+"{0:.4f}".format(np.median(aod_01[keep]))    
        print(out_text)
        out_text = "  Mean AOD = "+"{0:.4f}".format(np.mean(aod_01[keep]))    
        print(out_text)
        out_text = "   Max AOD = "+"{0:.4f}".format(np.amax(aod_01[keep])) 
        print(out_text)

# Mask locations without valid retrievals

        mask = (succ_01 < 1.0)
        aod_01[mask] = 0.0
            
# Spatial statistics
# Note: First, we resize the arrays from their native resolution to 4 x 4 coarser
#       resolution by summing
#       Then, we calculate the mean AOD value at the coarser resolution using the number
#       of valid retrievals
#       Next, we resize the arrays from the coarser resolution back to their native
#       resolution

        succ_coarse = np.squeeze(succ_01.reshape([y_aero/4,4,x_aero/4,4]).sum(3).sum(1))
        aod_coarse = np.squeeze(aod_01.reshape([y_aero/4,4,x_aero/4,4]).sum(3).sum(1))
        
        denom = succ_coarse
        denom[succ_coarse < 1.0] = 1.0
        mean_aod_coarse = aod_coarse/denom
        
        mean_aod_fine = np.repeat(np.repeat(mean_aod_coarse,4,axis=0),
            4,axis=1)
        
# Calculate the relative sampling of the original resolution data to the coarse
# resolution data
        
        keep = (succ_coarse > 1.0)
        rel_sample = (np.mean(succ_coarse[keep])/16.0) # Denominator is from 4 x 4
        out_text = " Relative Sampling (0.0 to 1.0) = "+"{0:.4f}".format(rel_sample) 
        print(out_text)
        
# Calculate the correlation between the new fine array and the original array

        aod_01_flat = aod_01.flatten() # 1-D array
        mean_aod_fine_flat = mean_aod_fine.flatten() # 1-D array
        
        temp = np.corrcoef(aod_01_flat,mean_aod_fine_flat)
        rel_corr = temp[0,1]
        
        out_text = "Relative Correlation (-1 to +1) = "+"{0:.4f}".format(rel_corr) 
        print(out_text)
                
# Plot the data

        img = aod_01

        im = m.pcolormesh(lon_44,lat_44,img,shading='flat',
            cmap=plt.cm.CMRmap_r,latlon=True,vmin=aod_plot_min,vmax=aod_plot_max)

# Plot the AERONET points

        m.scatter(aero_lon,aero_lat,marker='o',c=aero_aod,
            cmap=plt.cm.CMRmap_r,vmin=aod_plot_min,vmax=aod_plot_max,s=25)

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

    plt.title('V'+misr_name+' Lowest Residual AOD')

# Draw basemap

    m = Basemap(llcrnrlon=lon_min,llcrnrlat=lat_min,urcrnrlon=lon_max,urcrnrlat=lat_max,
        projection='cyl',resolution='i')
    m.drawmapboundary(fill_color='0.3')
    
## Loop over blocks

    print()
    print("***Lowest Residual***")

    for i in range(num_block):

        block = start_block + i - 1 # Block 0-based

# Extract the data

        lat_44 = lat_raw[block,:,:]
        lon_44 = lon_raw[block,:,:]
        aod_01 = rbe_aod[block,:,:]*1.
        succ_01 = success[block,:,:]*1.
        mix_aod = aod_per_mix[block,:,:,:]*1. # Additional mixture dimension
        mix_chisq = chisq_per_mix[block,:,:,:]*1.  # Additional mixture dimension
        
        keep = (succ_01 > 0.0)
        
# Test for a successful search        
        
        if(np.amax(keep) == False):
            continue         
    
        new_chisq_01 = np.zeros_like(aod_01)
        new_aod_01 = np.zeros_like(aod_01)

# Loop through the block and find the location with the lowest (valid) chi-squared
# value and store the chi-squared value and associated AOD
    
        for y in range(y_aero):
            for x in range(x_aero):
                hold = mix_chisq[y,x,:]
                test = np.copy(hold)
                if(np.amax(test) > 0.0):
                    test[hold < 0.0] = 9999.9
                    check = np.argmin(test)
                    new_chisq_01[y,x] = np.amin(test)
                    temp = mix_aod[y,x,:]
                    new_aod_01[y,x] = temp[check]

# Print some information

        print("Block = ",block+1) # MISR-format
        out_text = "   Min AOD = "+"{0:.4f}".format(np.amin(new_aod_01[keep])) 
        print(out_text)  
        out_text = "Median AOD = "+"{0:.4f}".format(np.median(new_aod_01[keep]))    
        print(out_text)
        out_text = "  Mean AOD = "+"{0:.4f}".format(np.mean(new_aod_01[keep]))    
        print(out_text)
        out_text = "   Max AOD = "+"{0:.4f}".format(np.amax(new_aod_01[keep])) 
        print(out_text)

# Mask locations without valid retrievals

        mask = (succ_01 < 1.0)
        new_aod_01[mask] = 0.0

# Spatial statistics
# Note: First, we resize the arrays from their native resolution to 4 x 4 coarser
#       resolution by summing
#       Then, we calculate the mean AOD value at the coarser resolution using the number
#       of valid retrievals
#       Next, we resize the arrays from the coarser resolution back to their native
#       resolution

        succ_coarse = np.squeeze(succ_01.reshape([y_aero/4,4,x_aero/4,4]).sum(3).sum(1))
        aod_coarse = np.squeeze(new_aod_01.reshape([y_aero/4,4,x_aero/4,4]).sum(3).sum(1))
        
        denom = succ_coarse
        denom[succ_coarse < 1.0] = 1.0
        mean_aod_coarse = aod_coarse/denom
        
        mean_aod_fine = np.repeat(np.repeat(mean_aod_coarse,4,axis=0),
            4,axis=1)
        
# Calculate the relative sampling of the original resolution data to the coarse
# resolution data
        
        keep = (succ_coarse > 1.0)
        rel_sample = (np.mean(succ_coarse[keep])/16.0) # Denominator is from 4 x 4
        out_text = " Relative Sampling (0.0 to 1.0) = "+"{0:.4f}".format(rel_sample) 
        print(out_text)
        
# Calculate the correlation between the new fine array and the original array

        aod_01_flat = new_aod_01.flatten() # 1-D array
        mean_aod_fine_flat = mean_aod_fine.flatten() # 1-D array
        
        temp = np.corrcoef(aod_01_flat,mean_aod_fine_flat)
        rel_corr = temp[0,1]
        
        out_text = "Relative Correlation (-1 to +1) = "+"{0:.4f}".format(rel_corr) 
        print(out_text)  
            
# Plot the data

        img = new_aod_01

        im = m.pcolormesh(lon_44,lat_44,img,shading='flat',
            cmap=plt.cm.CMRmap_r,latlon=True,vmin=aod_plot_min,vmax=aod_plot_max)

# Plot the AERONET points

        m.scatter(aero_lon,aero_lat,marker='o',c=aero_aod,
            cmap=plt.cm.CMRmap_r,vmin=aod_plot_min,vmax=aod_plot_max,s=25)

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

# Extract the navigation for all the blocks

    lat_44 = lat_raw[(start_block-1):(start_block-1+num_block),:,:]
    lon_44 = lon_raw[(start_block-1):(start_block-1+num_block),:,:]
    
# Extract the aerosol data for all the blocks

    rbe_raw_44 = rbe_aod[(start_block-1):(start_block-1+num_block),:,:]*1.
    succ_44 = success[(start_block-1):(start_block-1+num_block),:,:]*1.
    mix_aod = aod_per_mix[(start_block-1):(start_block-1+num_block),:,:,:]*1.
    mix_chisq = chisq_per_mix[(start_block-1):(start_block-1+num_block),:,:,:]*1.

# Loop through the blocks and find the location with the lowest (valid) chi-squared
# value and store the chi-squared value and associated AOD
    
    rlr_raw_44 = np.zeros_like(rbe_raw_44)
    rlr_chi2_44 = np.zeros_like(rbe_raw_44)
    rlr_mix_44 = np.zeros_like(rbe_raw_44)
    
    for b in range(num_block):
        for y in range(y_aero):
            for x in range(x_aero):
                hold = mix_chisq[b,y,x,:]
                test = np.copy(hold)
                if(np.amax(test) > 0.0):
                    test[hold < 0.0] = 9999.9
                    check = np.argmin(test)
                    temp = mix_aod[b,y,x,:]
                    rlr_raw_44[b,y,x] = temp[check]
                    rlr_chi2_44[b,y,x] = np.amin(test)
                    rlr_mix_44[b,y,x] = check+1  # Mixture index is 1-based

# Eliminate locations without a valid retrieval
# Note: These arrays were accumulated in the Lowest Residual processing step

    rbe_44 = rbe_raw_44 * succ_44
    rlr_44 = rlr_raw_44 * succ_44
    chi2_44 = rlr_chi2_44 * succ_44
    mix_44 = rlr_mix_44 * succ_44

# Get the number of AERONET sites
    
    num_aero = len(aero_aod)

# Set up arrays to store the matched data

    aero_match = np.zeros(num_aero)
    rbe_v22_match = np.zeros(num_aero)
    rlr_v22_match = np.zeros(num_aero)
    rlr_chi2_match = np.zeros(num_aero)
    rlr_mix_match = np.zeros(num_aero)
    dist_match = np.zeros(num_aero)
    time_match = np.zeros(num_aero)
    lon_match = np.zeros(num_aero)
    lat_match = np.zeros(num_aero)

# Loop over AERONET retrievals

    for i in range(num_aero):

# Extract AERONET information
    
        tlat = aero_lat[i]
        tlon = aero_lon[i]
        taod = aero_aod[i]
        ttim = aero_tdiff[i]

# Calculate the array of distances (in km) from the AERONET point vs. the MISR data
        
        dist = Haversine_Distance(tlat,tlon,lat_44,lon_44)
        
# Find the minimum distance (km)

        min_dist = np.amin(dist)
    
# Test for a match within a single (4.4 km) MISR pixel

        if(min_dist <= 4.4):

# Extract the match

            match = (dist == min_dist)   
            
# Extract the MISR data

            aero_match[i] = taod
            rbe_v22_match[i] = rbe_44[match]
            rlr_v22_match[i] = rlr_44[match]
            rlr_chi2_match[i] = chi2_44[match]
            rlr_mix_match[i] = mix_44[match]
            dist_match[i] = min_dist
            time_match[i] = ttim
            lat_match[i] = tlat
            lon_match[i] = tlon

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
    plt.title('V'+misr_name+' Best Estimate')

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

    x_pos = (aod_plot_max/2.) + (aod_plot_max/10.)
    y_pos1 = (aod_plot_max/10.)
    y_pos2 = y_pos1 + (aod_plot_max/30.)
    y_pos3 = y_pos2 + (aod_plot_max/30.)
    y_pos4 = y_pos3 + (aod_plot_max/30.)
    y_pos5 = y_pos4 + (aod_plot_max/30.)
    y_pos6 = y_pos5 + (aod_plot_max/30.)
    
    plt.text(x_pos,y_pos6,'Best Estimate',fontsize=12) # Version

    count = len(test_aod)
    out_text = 'N = '+str(count)
    plt.text(x_pos,y_pos5,out_text,fontsize=10) # Count

    temp = np.corrcoef(ref_aod,test_aod)
    be_r = temp[0,1]
    out_text = 'r = '+"{0:.4f}".format(be_r)
    plt.text(x_pos,y_pos4,out_text,fontsize=10) # Correlation coefficient

    rmse = np.sqrt(((test_aod - ref_aod) ** 2).mean())
    out_text = 'RMSE = '+"{0:.4f}".format(rmse)
    plt.text(x_pos,y_pos3,out_text,fontsize=10) # Root mean squared error

    diff = test_aod - ref_aod
    bias = np.mean(diff)
    out_text = 'Bias = '+"{0:.4f}".format(bias)
    plt.text(x_pos,y_pos2,out_text,fontsize=10) # Bias

    offset = np.ones_like(ref_aod)*0.05
    inner = np.absolute(diff) < np.maximum(offset,ref_aod*0.2)
    in_frac = (np.sum(inner)/(1.0*count))*100.0
    out_text = 'Percent In = '+"{0:.2f}".format(in_frac)
    plt.text(x_pos,y_pos1,out_text,fontsize=10) # Percent in envelope

## Linear plot (Lowest Residual)

    good = (rlr_v22_match > 0)
    ref_aod = aero_match[good]
    test_aod = rlr_v22_match[good]

    plt.subplot(1, 2, 2)
    plt.scatter(ref_aod,test_aod,marker='o',color='black',s=25)   
    plt.title('V'+misr_name+' Lowest Residual')

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

# Include some text on the Lowest Residual Figure

    plt.text(x_pos,y_pos6,'Lowest Resid',fontsize=12) # Version

    count = len(test_aod)
    out_text = 'N = '+str(count)
    plt.text(x_pos,y_pos5,out_text,fontsize=10) # Count

    temp = np.corrcoef(ref_aod,test_aod)
    be_r = temp[0,1]
    out_text = 'r = '+"{0:.4f}".format(be_r)
    plt.text(x_pos,y_pos4,out_text,fontsize=10) # Correlation coefficient

    rmse = np.sqrt(((test_aod - ref_aod) ** 2).mean())
    out_text = 'RMSE = '+"{0:.4f}".format(rmse)
    plt.text(x_pos,y_pos3,out_text,fontsize=10) # Root mean squared error

    diff = test_aod - ref_aod
    bias = np.mean(diff)
    out_text = 'Bias = '+"{0:.4f}".format(bias)
    plt.text(x_pos,y_pos2,out_text,fontsize=10) # Bias

    offset = np.ones_like(ref_aod)*0.05
    inner = np.absolute(diff) < np.maximum(offset,ref_aod*0.2)
    in_frac = (np.sum(inner)/(1.0*count))*100.0
    out_text = 'Percent In = '+"{0:.2f}".format(in_frac)
    plt.text(x_pos,y_pos1,out_text,fontsize=10) # Percent in envelope

# Save the figure

    os.chdir(figpath)
    outname = 'AOD_Regression'+out_base
    plt.savefig(outname,dpi=120)

# Show the plot

    plt.show()
#    print(error)

### Write out the data to a NetCDF file

# Extract the data

    good = (rlr_v22_match > 0)
    lat_good = lat_match[good]
    lon_good = lon_match[good]
    aero_good = aero_match[good]
    rbe_good = rbe_v22_match[good]
    rlr_good = rlr_v22_match[good]
    chi2_good = rlr_chi2_match[good]
    mix_good = rlr_mix_match[good]
    dist_good = dist_match[good]
    time_good = time_match[good]

    if(exist == 0):  #  Create the initial file

# Open the file
        
        n_id = Dataset(outfile,'w')
        
# Define the dimensions

        xdim = n_id.createDimension('xdim') # Unlimited        

# Define the output variables

        out01 = n_id.createVariable('Latitude','f4',('xdim',),zlib=True)
        out02 = n_id.createVariable('Longitude','f4',('xdim',),zlib=True)
        
        out03 = n_id.createVariable('AERONET_AOD','f4',('xdim',),zlib=True)
        out04 = n_id.createVariable('MISR_BE_AOD','f4',('xdim',),zlib=True)
        out05 = n_id.createVariable('MISR_LR_AOD','f4',('xdim',),zlib=True)
        
        out06 = n_id.createVariable('MISR_LR_CHI2','f4',('xdim',),zlib=True)
        out07 = n_id.createVariable('MISR_LR_MIX','f4',('xdim',),zlib=True)
        
        out08 = n_id.createVariable('Delta_T','f4',('xdim',),zlib=True)
        out09 = n_id.createVariable('Distance','f4',('xdim',),zlib=True)
        
# Put the data into the output

        out01[:] = lat_good 
        out02[:] = lon_good
                 
        out03[:] = aero_good
        out04[:] = rbe_good         
        out05[:] = rlr_good
        
        out06[:] = chi2_good
        out07[:] = mix_good
    
        out08[:] = time_good
        out09[:] = dist_good         
      
# Close the NetCDF file
        
        n_id.close()
            
## Append to existing file
            
    else:    #  File exists    

# Get the number of new entries

        num_good = len(lon_good)
        
# Now open the existing file and append information

        n_id = Dataset(outfile,'a')

# Choose variables
        
        var01 = n_id.variables['Latitude']
        var02 = n_id.variables['Longitude']
        
        var03 = n_id.variables['AERONET_AOD']
        var04 = n_id.variables['MISR_BE_AOD']
        var05 = n_id.variables['MISR_LR_AOD']
        
        var06 = n_id.variables['MISR_LR_CHI2']
        var07 = n_id.variables['MISR_LR_MIX']
        
        var08 = n_id.variables['Delta_T']
        var09 = n_id.variables['Distance']
            
# Get the current size of the variables            
            
        current = len(var01)
            
# Append the information

        var01[current:current+num_good] = lat_good
        var02[current:current+num_good] = lon_good
        
        var03[current:current+num_good] = aero_good
        var04[current:current+num_good] = rbe_good
        var05[current:current+num_good] = rlr_good
        
        var06[current:current+num_good] = chi2_good
        var07[current:current+num_good] = mix_good
        
        var08[current:current+num_good] = time_good
        var09[current:current+num_good] = dist_good
       
# Close the file            
        
        n_id.close()    
    
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
