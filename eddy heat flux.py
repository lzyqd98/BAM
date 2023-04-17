#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 01:23:34 2023

@author: Zhaoyu Liu
"""
############## This code is for eddy heat flux ##################

from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import datetime as dt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd
import glob


###### Filtering Function ######
def bandpass(x,k,w):
    xf = np.fft.fft(x * w)
    xftmp=xf*0.0
    xftmp[k]=xf[k]
    xftmp[-k]=xf[-k]
    xout=np.fft.ifft(xftmp)
    return xout
#################################



#First we get the directory of each nc data

files_u = glob.glob(r"/Volumes/depot/eapsdept/data/ERA_Interim/*u.nc")
files_u.sort()
N=len(files_u)   #The number of nc files


files_v = glob.glob(r"/Volumes/depot/eapsdept/data/ERA_Interim/*v.nc")
files_v.sort()
N=len(files_v)   #The number of nc files

files_t = glob.glob(r"/Volumes/depot/eapsdept/data/ERA_Interim/*t.nc")
files_t.sort()
N=len(files_t)   #The number of nc files


#read any data file to read some basic variables like lon and lat
File0 = netcdf.netcdf_file(files_u[0],'r')

#read lon and lat
lon = File0.variables['longitude'][:]
lat = File0.variables['latitude'][:]
lat_SH = lat[60:]
plev = File0.variables['levelist'][:]

nlon = len(lon)
nlat = len(lat)
nplev = len(plev)

Start_date = dt.datetime(1979, 1, 1)
delta_t = dt.timedelta(days=1)
Datestamp = [Start_date + delta_t*tt for tt in np.arange(13880)]
Date = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Month = Date['date'].dt.month 
Year = Date['date'].dt.year
Day = Date['date'].dt.day


VT_td = np.zeros((13880,nlat,nlon)) #Calculate every 30 year



ti = -1

###--------The core code-----------------


for n in np.arange(152):
    ft= netcdf.netcdf_file(files_t[n],'r')
    fv = netcdf.netcdf_file(files_v[n],'r')
    t_4 = ft.variables['t'].data[:,30,:,:] * ft.variables['t'].scale_factor + ft.variables['t'].add_offset  ## Extract 800hPa ##
    v_4 = fv.variables['v'].data[:,30,:,:] * fv.variables['v'].scale_factor + fv.variables['v'].add_offset

    nt = len(ft.variables['time'].data[:])
    
    
    for t in np.arange(int(nt/4)): 
        ti+=1
        t_d = t_4[t*4:(t+1)*4,:,:].mean(axis=0)
        v_d = v_4[t*4:(t+1)*4,:,:].mean(axis=0)
                            
        t0_zm = t_d[:,:].mean(axis=1)
        v0_zm = v_d[:,:].mean(axis=1)

        t0_a = t_d[:,:] - t0_zm[:,np.newaxis]
        v0_a = v_d[:,:] - v0_zm[:,np.newaxis]
        
        VT_td[ti,:,:] = v0_a * t0_a
            
        print(ti)
        
        
np.save('/Volumes/Backup Plus/VT/ERA_Interim/VT_td.npy', VT_td)


####### Filtered Variance ########
VT_td = np.load('/Volumes/Backup Plus/VT/ERA_Interim/VT_td.npy')
u_td = np.load("/Volumes/Backup Plus/U/ERA_Interim/U_baro.npy")

VT_td_SH = VT_td[:,60:,:]
u_td_SH = u_td[:,60:,:]

Start_date = dt.datetime(1979, 1, 1)
delta_t = dt.timedelta(days=1)
Datestamp = [Start_date + delta_t*tt for tt in np.arange(13880)]
Date = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Month = Date['date'].dt.month 
Year = Date['date'].dt.year
Day = Date['date'].dt.day

##########################
VT_NM = []
for i in np.arange(37):
    n1 = np.squeeze(np.array(np.where(Date['date']==str(1979+i)+'-12-01 00:00:00')))
    n2 = n1+90
    VT_NM.append(VT_td_SH[n1:n2,:,:])
VT_NM2 = np.array(VT_NM)
VT_zm = VT_NM2.mean(axis=3)  
    
U_NM = []
for i in np.arange(37):
    n1 = np.squeeze(np.array(np.where(Date['date']==str(1979+i)+'-12-01 00:00:00')))
    n2 = n1+90
    U_NM.append(u_td_SH[n1:n2,:,:])
U_NM = np.array(U_NM)
U_clit = np.mean(U_NM.mean(axis=0),axis=0)
    
########################
Fs = 1
Ts = 1.0/Fs
t = np.arange(0,90,Ts)

n = 90 #length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T
frq1 = frq[range(int(n/2))] # one side frequency range

n_year=37

coslat = np.cos(np.deg2rad(lat[60:])).clip(0., 1.) #clip limits the arary to a range
wgts = coslat[np.newaxis,:]
wgts = wgts* np.ones((n, 61))



######### First filter the high frequency from 2-10 day ##############

k = np.arange(15, int(n/2)+1)
w = np.hanning(n)    ##Hanning window is required
VT_high = np.zeros((n_year,n,61,240))
VT_high_std = np.zeros((n_year,61,240))
VT_high_zm = np.zeros((n_year,n,61))
VT_high_zm_std = np.zeros((n_year,61))

for i in np.arange(n_year):
    Y = VT_NM[i]
    Y_zm = VT_NM[i].mean(axis=2)
    for la in np.arange(61):
        y0 = Y_zm[:,la] * wgts[:,la]
        y_a = y0 - y0.mean()    ### remove the seasoanl mean
        VT_high_zm[i,:,la] = bandpass(y0, k,w)
        VT_high_zm_std[i,la] = np.std(VT_high_zm[i,:,la])        
        for lo in np.arange(240):
            y0 = Y[:,la,lo] * wgts[:,la]
            y_a = y0 - y0.mean()    ### remove the seasoanl mean
            VT_high[i,:,la,lo] = bandpass(y0, k,w)
            VT_high_std[i,la,lo] = np.std(VT_high[i,:,la,lo])

            
VT_high_m = VT_high.mean(axis=0)
VT_high_m_clim = VT_high_m.mean(axis=0)
VT_high_std_m = VT_high_std.mean(axis=0)
VT_high_zm_std_m = VT_high_zm_std.mean(axis=0)

np.save('/Users/liu3315/Documents/Research_BAM/Fall2022/week1/synoptic_VT_Z500_NH_DJF.npy', VT_high_std_m)

######### Then filter the low frequency from 10-45 day ##############

k = np.arange(2,10)
w = np.hanning(n)    ##Hanning window is required
VT_low = np.zeros((n_year,n,61,240))
VT_low_std = np.zeros((n_year,61,240))
VT_low_zm = np.zeros((n_year,n,61))
VT_low_zm_std = np.zeros((n_year,61))

for i in np.arange(n_year):
    Y = VT_NM[i]
    Y_zm = VT_NM[i].mean(axis=2)
    for la in np.arange(61):
        y0 = Y_zm[:,la] * wgts[:,la]
        y_a = y0 - y0.mean()    ### remove the seasoanl mean
        VT_low_zm[i,:,la] = bandpass(y_a, k,w)
        VT_low_zm_std[i,la] = np.std(VT_low_zm[i,:,la]) 
        
        for lo in np.arange(240):
            y0 = Y[:,la,lo] * wgts[:,la]
            y_a = y0 - y0.mean()
            VT_low[i,:,la,lo] = bandpass(y_a, k,w)
            # VT_low_std[i,la,lo] = np.sqrt( np.sum(VT_low[i,:,la,lo]**2) ) / n
            VT_low_std[i,la,lo] = np.std(VT_low[i,:,la,lo])

            
VT_low_m = VT_low.mean(axis=0) 
VT_low_m_clim = VT_low_m.mean(axis=0)
VT_low_std_m = VT_low_std.mean(axis=0)
VT_low_zm_std_m = VT_low_zm_std.mean(axis=0)

np.save('/Users/liu3315/Documents/Research_BAM/Fall2022/week1/intraseasonal_VT_Z500_NH_DJF.npy', VT_low_std_m)



####### Plot the zonal mean eddy heat flux variance ###########

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

plt.plot(lat_SH, VT_low_zm_std_m, '-r', linewidth=2, label='intraseasonal')
plt.plot(lat_SH, VT_high_zm_std_m, '-k', linewidth=2, label = 'synoptic')
plt.title("zonal mean eddy heat flux variability", pad=5, fontdict={'family':'Times New Roman', 'size':16})
plt.xlabel('latitude',fontsize=12)
plt.ylabel('standard deviation',fontsize=12)
plt.legend()


########### Plot the Southern Hemisphere ##################################

maxlevel = VT_high_std_m[20:48,:].max()
minlevel = VT_high_std_m[20:48,:].min()  
levs = np.linspace(0,10, 10)
    
fig = plt.figure()
ax = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree(central_longitude=180))

h1 = plt.contourf(lon,lat_SH[20:48], VT_high_std_m[20:48,:], levs, transform=ccrs.PlateCarree(), cmap='rainbow' ,extend ='max')  #_r is reverse
h2 = plt.contour(lon,lat_SH[20:48], VT_high_std_m[20:48,:], levs[:], transform=ccrs.PlateCarree(), linewidths=0.5, colors='k') 

plt.ylabel('latitude',fontsize=12)
plt.title("Synoptic Variability (2-7d) of 850hPa Eddy Heat Flux (DJF)", pad=5, fontdict={'family':'Times New Roman', 'size':14})
ax.coastlines()
#ax.gridlines(linestyle="--", alpha=0.7)

ax.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
ax.set_yticks([-70,-60,-50,-40,-30], crs=ccrs.PlateCarree())
# ax.set_extent([0,358.5,-90,-60], crs=ccrs.PlateCarree()) 
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)  



bx = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree(central_longitude=180))

h3 = plt.contourf(lon,lat_SH[20:48], VT_low_std_m[20:48,:], levs, transform=ccrs.PlateCarree(), cmap='rainbow' ,extend ='max')  #_r is reverse
h4 = plt.contour(lon,lat_SH[20:48], VT_low_std_m[20:48,:], levs[:], transform=ccrs.PlateCarree(), linewidths=0.5, colors='k') 

plt.xlabel('longitude',fontsize=12)
plt.ylabel('latitude',fontsize=12)
plt.title("Intraseasonal Variability (10-45d)of 850hPa Eddy Heat Flux (DJF)", pad=5, fontdict={'family':'Times New Roman', 'size':14})
bx.coastlines()
#bx.gridlines(linestyle="--", alpha=0.7)

bx.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
bx.set_yticks([-70,-60,-50,-40,-30], crs=ccrs.PlateCarree())
# ax.set_extent([0,358.5,-90,-60], crs=ccrs.PlateCarree()) 
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
bx.xaxis.set_major_formatter(lon_formatter)
bx.yaxis.set_major_formatter(lat_formatter)  

plt.subplots_adjust(hspace=-0.5, right=0.85)

######## Draw colorbar #######
cbar = fig.add_axes([0.9,0.25,0.015,0.5])
cb = plt.colorbar(h1, cax=cbar, ticks=[0, 2,4,6,8,10])

plt.savefig("/Users/liu3315/Documents/Research_BAM/Spring2023/Manuscript1/VT850.png",dpi=600)


















