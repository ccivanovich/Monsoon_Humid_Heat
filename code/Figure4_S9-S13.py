import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import cartopy.crs as ccrs
import os
import datetime
from datetime import timedelta
from datetime import datetime
from matplotlib import ticker
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.feature as cfeature
import geopandas as gp
import regionmask
import matplotlib.patheffects as pe
import pooch
from matplotlib.lines import Line2D
from scipy import stats
import string

def remove_time_mean(x):
    return x - x.mean(dim='time')

def standardize(x):
    return x/x.std(dim = 'time')

import warnings
warnings.filterwarnings("ignore")

fontsize = 14
plt.rcParams.update({'font.size': fontsize})

# Select daily max wet bulb temperatures during these days
TW_dailymax = xr.open_mfdataset('/dx01/data/ERA5/2mTW_dailymax/TW_daily_max_ERA5_historical_an-sfc_*_0UTC.nc')

TW_anom = TW_dailymax.groupby('time.dayofyear').apply(remove_time_mean)

# Only select data with unique days
_, index = np.unique(TW_dailymax['time'], return_index = True)
TW_unique = TW_dailymax.isel(time=index)

# Same for anomalies
_, index = np.unique(TW_anom['time'], return_index = True)
TW_unique_anom = TW_anom.isel(time=index)

# Select daily max temperatures during these days
temp_dailymax = xr.open_mfdataset('/dx01/data/ERA5/2mtemp_dailymax/*.nc')

# Only select data with unique days
_, index = np.unique(temp_dailymax['time'], return_index = True)
temp_unique = temp_dailymax.isel(time=index)

# Same for anomalies
temp_anom = temp_dailymax.groupby('time.dayofyear').apply(remove_time_mean)

_, index = np.unique(temp_anom['time'], return_index = True)
temp_unique_anom = temp_anom.isel(time=index)

# Surface dewpoint
d2m_data = xr.open_mfdataset('/dx01/data/ERA5/2mdewpoint_dailymean/*.nc', parallel = True, combine = 'by_coords')
d2m_C = d2m_data.d2m - 273.15 #convert from kelvin

# Surface pressure
sp_data = xr.open_mfdataset('/dx01/data/ERA5/surface_pressure_dailymean/*.nc', parallel = True, combine = 'by_coords')
sp_mb = sp_data/100

# Specific humidity
vap_pres = 6.112*np.exp((17.67*d2m_C)/(d2m_C + 243.5))
q = (0.622 * vap_pres)/(sp_mb.sp - (0.378 * vap_pres))
q_derived = q.to_dataset(name = 'q')

# Only select data with unique days
_, index = np.unique(q_derived['time'], return_index = True)
q_unique = q_derived.isel(time=index)

print('Meteorological data loaded.')

# Select precipitation data from each homogenous rainfall zone
shp_datadir = '/home/ivanov/jupyternb/Monsoon_TW/shape_files/'

NW = gp.read_file(shp_datadir + 'Northwest.shp')
bgd = gp.read_file(shp_datadir + 'bangladesh.shp')

lon = TW_unique.longitude
lat = TW_unique.latitude

NW_mask = regionmask.mask_geopandas(NW,lon,lat)
bgd_mask = regionmask.mask_geopandas(bgd, lon,lat)

print('Shape files loaded.')

# Regional data can be changed to select other subregions of analysis
regions = ['NW','bgd']
masks = [NW_mask,bgd_mask]

### Load annual onset dates
datadir = "/dx01/ivanov/data/CHIRPS/monsoon_humidheat_timing/"
name = 'onset_mondal_allregions.csv'

onset_df = pd.read_csv(datadir+name, index_col = 0)

print('Onset data loaded.')

# Calculate cannonical monsoon onset date for each region

avg_doy = []

for region_name in regions:

    onset_doy = []
    onset_region = onset_df[region_name]

    for yearly_onset in onset_region:

        yearly_datetime = datetime.strptime(yearly_onset,'%Y-%m-%d')
        day_of_year = yearly_datetime.timetuple().tm_yday
        onset_doy.extend([day_of_year])

    avg_doy.extend([np.mean(onset_doy)])
    
print('Avg onset dates calculated.')
    
fig, axs = plt.subplots(2,2,figsize = (13.5,10))
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

print('Ready to plot!')

### Loop through and plot regional data
for i in range(len(regions)):
    
    region = regions[i]
    mask = masks[i]
    
    # Add mask coordinates
    TW_unique.coords[region] = (('latitude', 'longitude'), mask)

    temp_unique.coords[region] = (('latitude', 'longitude'), mask)
    
    q_unique.coords[region] = (('latitude', 'longitude'), mask)
    
    TW_unique_anom.coords[region] = (('latitude', 'longitude'), mask)
    temp_unique_anom.coords[region] = (('latitude', 'longitude'), mask)
    
    # Load regional data
    reg_TW_mean = xr.open_dataset("/dx01/ivanov/data/ERA5/monsoon_humidheat_timing/" + region + "_daily_TW_mean.nc")
    reg_TW_min = xr.open_dataset("/dx01/ivanov/data/ERA5/monsoon_humidheat_timing/" + region + "_daily_TW_min.nc")
    reg_TW_max = xr.open_dataset("/dx01/ivanov/data/ERA5/monsoon_humidheat_timing/" + region + "_daily_TW_max.nc")

    reg_T_mean = xr.open_dataset("/dx01/ivanov/data/ERA5/monsoon_humidheat_timing/" + region + "_daily_T_mean.nc")
    reg_T_min = xr.open_dataset("/dx01/ivanov/data/ERA5/monsoon_humidheat_timing/" + region + "_daily_T_min.nc")
    reg_T_max = xr.open_dataset("/dx01/ivanov/data/ERA5/monsoon_humidheat_timing/" + region + "_daily_T_max.nc")

    reg_q_mean = xr.open_dataset("/dx01/ivanov/data/ERA5/monsoon_humidheat_timing/" + region + "_daily_q_mean.nc")
    reg_q_min = xr.open_dataset("/dx01/ivanov/data/ERA5/monsoon_humidheat_timing/" + region + "_daily_q_min.nc")
    reg_q_max = xr.open_dataset("/dx01/ivanov/data/ERA5/monsoon_humidheat_timing/" + region + "_daily_q_max.nc")
    
    print('Regional data is loaded.')

    ### TOTAL PLOTS
    axs[i,0].plot(reg_TW_mean.dayofyear, reg_TW_mean.TW, markersize = 0.25, c = 'k')
    axs[i,0].fill_between(reg_TW_mean.dayofyear, reg_TW_min.TW, reg_TW_max.TW, color = 'k', alpha = 0.25)
    axs[i,0].set_ylabel('Wet Bulb Temperature (C)')
    axs[i,0].set_xlabel('Day of Year')

    axs[i,0].axvline(x=avg_doy[i], color='k', linestyle = '--', linewidth = 1)
    
    if i == 0:
        axs[i,0].scatter(171,10.8, marker = ">", color = 'k')
        axs[i,0].text(137, 10.2, 'climatological\n monsoon onset', multialignment='center', fontsize = 10)

    masks = [NW_mask,bgd_mask]

    region = regions[i]
    mask = masks[i]

    data_reg = TW_unique['TW'].where(TW_unique[region] == 0).mean(dim= ['latitude','longitude'], skipna=True)
    Tdata_reg = temp_unique['T'].where(temp_unique[region] == 0).mean(dim= ['latitude','longitude'], skipna=True)
    qdata_reg = q_unique['q'].where(q_unique[region] == 0).mean(dim= ['latitude','longitude'], skipna=True)

    onset_dates = onset_df[region]

    onset_dates = list(pd.to_datetime(onset_dates, infer_datetime_format=True) + timedelta(days = -1)) # if want day before onset

    # Select data during monsoon onset and surrounding
    data_onset = data_reg.sel(time = onset_dates)
    data_onset.coords['Day of Year'] = data_onset.time.dt.dayofyear
    data_onset.load()
    
    Tdata_onset = Tdata_reg.sel(time = onset_dates)
    Tdata_onset.coords['Day of Year'] = Tdata_onset.time.dt.dayofyear
    Tdata_onset.load()
    
    qdata_onset = qdata_reg.sel(time = onset_dates)
    qdata_onset.coords['Day of Year'] = qdata_onset.time.dt.dayofyear
    qdata_onset.load() 

    # For total magnitude plots        
    ax2=axs[i,0].twinx()
    ax2.plot(reg_T_mean.dayofyear, reg_T_mean.T - 273.15, c = 'r')
    ax2.scatter(x = Tdata_onset['Day of Year'], y = Tdata_onset - 273.15, c = 'r', s = 10)
    ax2.set_ylabel('Dry Bulb Temperature (C)')
    ax2.yaxis.label.set_color('r')

    [t.set_color('red') for t in ax2.yaxis.get_ticklabels()]

    ax3=axs[i,0].twinx()
    ax3.plot(reg_q_mean.dayofyear, 1000*reg_q_mean.q, c = 'b')
    ax3.scatter(x = qdata_onset['Day of Year'], y = 1000*qdata_onset, c = 'b', s = 10)
    ax3.set_ylabel('Specific Humidity (g/kg)')
    ax3.yaxis.label.set_color('b')
    ax3.tick_params(axis='y', which='major', pad=37, right = False)

    [t.set_color('blue') for t in ax3.yaxis.get_ticklabels()]

    axs[i,0].scatter(x = data_onset['Day of Year'], y = data_onset, c = 'k', s = 10)
    
    axs[i,0].set_xticks([100,125,150,175,200,225])
    axs[i,0].set_xlim([100,225])
    axs[i,0].set_ylim([10,30])
    axs[i,0].set_yticks([10,15,20,25,30])

    ax2.set_ylim([30,55])
    ax2.set_yticks([30,35,40,45,50,55])

    ax3.set_ylim([7,28])
    ax3.set_yticks([7,14,21,28])

    print(str(region) + ' total magnitude plot is plotted.')
    
    ### ANOMALY PLOTS
    
    data_reg = TW_unique_anom['TW'].where(TW_unique_anom[region] == 0).mean(dim= ['latitude','longitude'], skipna=True)
    
    Tdata_reg = temp_unique_anom['T'].where(temp_unique_anom[region] == 0).mean(dim= ['latitude','longitude'], skipna=True)
    
    onset_dates = onset_df[region]
    onset_dates = list(pd.to_datetime(onset_dates, infer_datetime_format=True) + timedelta(days = -1)) # if want day before onset

    # Select data during monsoon onset and surrounding
    data_onset = data_reg.sel(time = onset_dates)
    data_onset.coords['Day of Year'] = data_onset.time.dt.dayofyear
    data_onset.load()
    
    Tdata_onset = Tdata_reg.sel(time = onset_dates)
    Tdata_onset.coords['Day of Year'] = Tdata_onset.time.dt.dayofyear
    Tdata_onset.load()    
    
    # Anomaly plots
    axs[i,1].axvline(x=avg_doy[i], color='k', linestyle = '--', linewidth = 1)

    axs[i,1].scatter(x = data_onset['Day of Year'], y = data_onset, c = 'k', s = 10)
    axs[i,1].set_ylim([-2.5,2.5])
    axs[i,1].set_yticks([ -2,-1,0,1,2])
    axs[i,1].set_ylabel('Wet Bulb Temperature Anomaly (C)')
    axs[i,1].set_xlabel('Day of Year')

    xmin = 100
    xmax = 225
    axs[i,1].set_xlim([xmin,xmax])

    # Add in regression line
    res = stats.linregress(data_onset['Day of Year'],data_onset)

    x = np.linspace(xmin,xmax,50)

    axs[i,1].text(xmax-48, 2.2, 'r = ' + f'{res.rvalue:.2f}' + '; p = ' + f'{res.pvalue:.2f}', fontsize=fontsize)
    axs[i,1].set_xlim([100,225])
    axs[i,1].set_xticks([100,125,150,175,200,225])
    axs[i,1].set_xticklabels([100,125,150,175,200,225])
    
    print(str(res.pvalue))
    
    # Anomaly plots for T
    ax4=axs[i,1].twinx()

    ax4.scatter(x = Tdata_onset['Day of Year'], y = Tdata_onset, c = 'r', s = 10)
    ax4.set_ylim([-6,10])
    ax4.set_yticks([-6,-3,0,3,6,9])
    ax4.set_ylabel('Dry Bulb Temperature Anomaly (C)')
    ax4.yaxis.label.set_color('r')
    ax4.set_xlabel('Day of Year')

    xmin = 100
    xmax = 225
    ax4.set_xlim([xmin,xmax])
    
    [t.set_color('red') for t in ax4.yaxis.get_ticklabels()]

    # Add in regression line
    res = stats.linregress(Tdata_onset['Day of Year'],Tdata_onset)


    x = np.linspace(xmin,xmax,50)

    ax4.set_xlim([100,225])
    ax4.set_xticks([100,125,150,175,200,225])
    ax4.set_xticklabels([100,125,150,175,200,225])
    ax4.text(xmax-48, 8, 'r = ' + f'{res.rvalue:.2f}' + '; p = ' + f'{res.pvalue:.2f}', fontsize=fontsize, color = 'r')

    print(str(res.pvalue))
    
    print(str(region) + ' anomaly plot is plotted.')
    
axs[0,0].text(95, 31, string.ascii_lowercase[0], size=20, weight='bold')
axs[0,1].text(95, 2.65, string.ascii_lowercase[1], size=20, weight='bold')
axs[1,0].text(95, 31, string.ascii_lowercase[2], size=20, weight='bold')
axs[1,1].text(95, 2.65, string.ascii_lowercase[3], size=20, weight='bold')

line1 = Line2D([0], [0], color='k', linestyle = 'solid',linewidth = 3)
line2 = Line2D([0], [0], color= 'r', linestyle = 'solid',linewidth = 3)
line3 = Line2D([0], [0], color='b', linestyle = 'solid',linewidth = 3)

labels_legend = ['Wet Bulb','Dry Bulb','Specific Humidity']

axs[0,0].legend([line1,line2,line3], labels_legend, loc = 'upper left', prop={'size': 12})

plt.tight_layout()
plt.savefig('/home/ivanov/jupyternb/Monsoon_TW/Figure4.png')