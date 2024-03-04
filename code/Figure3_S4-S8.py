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

precip_datadir = '/dx01/data/CHIRPS/daily_precipitation_p25res/*.nc'
precip_data = xr.open_mfdataset(precip_datadir, parallel = True, combine = 'by_coords')

## Prepare masks

# Select precipitation data from each homogenous rainfall zone
shp_datadir = '/home/ivanov/jupyternb/Monsoon_TW/shape_files/'

NW = gp.read_file(shp_datadir + 'Northwest.shp')
bgd = gp.read_file(shp_datadir + 'bangladesh.shp')

# For CHIRPS
lon = precip_data.longitude
lat = precip_data.latitude

NW_mask = regionmask.mask_geopandas(NW,lon,lat)
bgd_mask = regionmask.mask_geopandas(bgd, lon,lat)

## Load data

# Select daily max wet bulb temperatures during these days
TW_dailymax = xr.open_mfdataset('/dx01/data/ERA5/2mTW_dailymax/TW_daily_max_ERA5_historical_an-sfc_*_0UTC.nc')

TW_anom = TW_dailymax.groupby('time.dayofyear').apply(remove_time_mean)

# Only select data with unique days
_, index = np.unique(TW_anom['time'], return_index = True)
TW_unique = TW_anom.isel(time=index)


# Select daily max wet bulb temperatures during these days
temp_dailymax = xr.open_mfdataset('/dx01/data/ERA5/2mtemp_dailymax/*.nc')

temp_anom = temp_dailymax.groupby('time.dayofyear').apply(remove_time_mean)

# Only select data with unique days
_, index = np.unique(temp_anom['time'], return_index = True)
temp_unique = temp_anom.isel(time=index)


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

# Calculate specific humidity anomalies
q_anom_derived = q_derived.groupby('time.dayofyear').apply(remove_time_mean)

# Only select data with unique days
_, index = np.unique(q_anom_derived['time'], return_index = True)
q_unique = q_anom_derived.isel(time=index)

## Mask for subregions

lon = TW_unique.longitude
lat = TW_unique.latitude

NW_mask = regionmask.mask_geopandas(NW,lon,lat)
bgd_mask = regionmask.mask_geopandas(bgd, lon,lat)

# These regions can be changed to plot other subregions
regions = ['NW','bgd']
masks = [NW_mask,bgd_mask]

for i in range(len(regions)):
    
    region = regions[i]
    mask = masks[i]
    
    # Add mask coordinates
    TW_unique.coords[region] = (('latitude', 'longitude'), mask)

    temp_unique.coords[region] = (('latitude', 'longitude'), mask)
    
    q_unique.coords[region] = (('latitude', 'longitude'), mask)


### Plot time evolution over all three periods, here for NW and BGD

# Use CHIRPS dates to identify breaks
data_list = [TW_unique, temp_unique, q_unique]
var_names = ['TW','T','q']
colors = ['k','r','b']
plot_names = ['NW','BGD']
shading_alpha = 0.25

onset_df = pd.read_csv("/dx01/ivanov/data/CHIRPS/monsoon_humidheat_timing/onset_mondal_allregions.csv")

print('Ready to plot!')

fig, axs = plt.subplots(3,2, figsize=([10,14]), facecolor='w', edgecolor='k')

print('Plot initialized.')

for k, ax in enumerate(fig.axes):
    
    print(k)
    
    if k < 2:
        
        if k == 0:
            
            i = 0
            
        if k == 1:
            
            i = 1
            
        region = regions[i]
        plot_title = plot_names[i]

        print('Region started: ' + region)

        mask = masks[i]

        for j in range(len(data_list)):

            data = data_list[j]
            var = var_names[j]
            plot_col = colors[j]

            print("Variable = " + var)

            data_reg = data[var].where(data[region] == 0).mean(dim= ['latitude','longitude'], skipna=True)

            datadir = "/dx01/ivanov/data/CHIRPS/monsoon_humidheat_timing/"
            break_data = 'dry_dates_monsoon_' + region + '.csv'

            break_length = pd.read_csv(datadir+break_data, index_col = 'time')

            break_dates = []

            for index, row in break_length.iterrows():
                count = row['count_below1'] # Adjust based on whether calculating during wet or dry spells (below vs. above)
                date = index
                for addition in range(count):
                    date_add = datetime.strptime(date,'%Y-%m-%d') + timedelta(days = addition)
                    break_dates.extend([date_add])

            break_dates_final = list(set(break_dates))
            break_dates_final.sort()

            # Select data X number of days before and after monsoon onset
            buffer = 7

            break_length.drop_duplicates(subset = ['crossing_below1'], keep = 'first', inplace = True)
            break_length = break_length

            # Add in middle and last day of each break, rounding up if odd number of days
            break_days = []

            for index, row in break_length.iterrows():

                # Buffer before
                date = index
                date_datetime = datetime.strptime(date,'%Y-%m-%d')

                for addition in range(-1*buffer, 0):
                    backward = date_datetime + timedelta(days = addition)
                    break_days.extend([backward])

                #First day
                break_days.extend([date_datetime])

                #Middle day
                tot_len = row['count_below1']
                mid_len = round(row['count_below1']/2)
                mid_day = date_datetime + timedelta(days = mid_len)
                break_days.extend([mid_day])

                #Last day
                last_day = date_datetime + timedelta(days = tot_len)
                break_days.extend([last_day])

                # Buffer after
                for addition in range(1, buffer+1):
                    forward = last_day + timedelta(days = addition)
                    break_days.extend([forward])

            break_norm = list(break_days)

            data_spell = data_reg.sel(time = break_norm)

            # Convert to pandas dataframe
            data_df = data_spell.to_dataframe()

            # Add onset number and day number to dataframe
            buffer_range = 2*buffer + 3
            num_breaks = len(data_df[var])/buffer_range
            break_count = np.arange(1,num_breaks+1,1)

            steps = 2
            days_before = np.arange(-1*buffer, 1,1)
            days_after = np.arange(2*steps + 1, 2*steps + buffer + 1,1)
            days = np.hstack((days_before, [steps,2*steps],days_after))
            day_array = np.tile(days,int(num_breaks))
            onset_array = np.repeat(break_count,buffer_range)

            data_df['Break Number'] = onset_array
            data_df['Day Number'] = day_array

            # Average across all breaks
            group_mean = data_df.groupby('Day Number').mean()
            group_std = data_df.groupby('Day Number').std()
            group_mean['up_2std'] = group_mean[var] + 2*group_std[var]
            group_mean['down_2std'] = group_mean[var] - 2*group_std[var]
            group_mean['up_1std'] = group_mean[var] + group_std[var]
            group_mean['down_1std'] = group_mean[var] - group_std[var]

            # Plot
            labels = [f'Break {i}' for i in range(1, int(num_breaks)+1)]
            labels.append('Average')

            if j == 0:

                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontsize(10)

                mean_line = group_mean.reset_index().plot(ax = ax, x = 'Day Number', y = var, color = plot_col, label = labels[-1], linewidth = 3, kind = 'line')

                shading = ax.fill_between(group_mean.index,group_mean['up_1std'],group_mean['down_1std'], color = plot_col, alpha = shading_alpha)

                ax.axvspan(0, 4, alpha=0.25, color='grey')    
                
                if k == 0:
                    ax.set_ylabel('Wet Bulb Temperature Anomaly (C)', fontsize = 10)

                ax.get_legend().remove()
                ax.set_ylim([-2,2])
                ax.set_xlim([-7,11])
                ax.text(-7.5, 2.2, string.ascii_lowercase[k], size=20, weight='bold')

            if j == 1:
                ax4 = ax.twinx()

                mean_line = group_mean.reset_index().plot(ax = ax4, x = 'Day Number', y = var, color = plot_col, label = labels[-1], linewidth = 3, kind = 'line')

                shading = ax4.fill_between(group_mean.index,group_mean['up_1std'],group_mean['down_1std'], color = plot_col, alpha = shading_alpha)

                if k == 1 :
                    ax4.set_ylabel('Temperature Anomaly (C)', fontsize = 10)
                    ax4.yaxis.label.set_color(plot_col)
                    
                ax4.get_legend().remove()
                ax4.set_ylim([-4,4])
                ax4.tick_params(axis='y', colors=plot_col)

            if j == 2:
                ax5 = ax.twinx()
                group_mean = group_mean*1000

                mean_line = group_mean.reset_index().plot(ax = ax5, x = 'Day Number', y = var, color = plot_col, label = labels[-1], linewidth = 3, kind = 'line')

                shading = ax5.fill_between(group_mean.index,group_mean['up_1std'],group_mean['down_1std'], color = plot_col, alpha = shading_alpha)

                ax5.axhline(y=0, color='white', linestyle='--', linewidth = 2)

                if k == 1:
                    ax5.set_ylabel('Specific Humidity Anomaly (g/kg)', fontsize = 10)
                    ax5.yaxis.label.set_color(plot_col)
                
                ax5.get_legend().remove()
                ax5.set_ylim([-2.5,2.5])

                ax5.tick_params(axis='y', which='major', pad=40, right = False, colors=plot_col)
        
        line1 = Line2D([0], [0], color='k', linestyle = 'solid',linewidth = 3)
        line2 = Line2D([0], [0], color= 'r', linestyle = 'solid',linewidth = 3)
        line3 = Line2D([0], [0], color='b', linestyle = 'solid',linewidth = 3)

        if k == 0:
            labels_legend = ['Wet Bulb','Dry Bulb','Specific Humidity']

            ax4.legend([line1,line2,line3], labels_legend, loc = 'upper left')

        ax.set_xticks([])
        ax.set_xticks([], minor=True)

        ax4.set_xticks([])
        ax4.set_xticks([], minor=True)   

        ax5.set_xticks([])
        ax5.set_xticks([], minor=True)
            
        print(str(region) + ' dry spells plot is plotted')

    if (k >= 2) & (k < 4):
        
        if k == 2:
            
            i = 0
            
        if k == 3:
            
            i = 1
            
        region = regions[i]
        plot_title = plot_names[i]

        print('Region started: ' + region)

        mask = masks[i]

        for j in range(len(data_list)):

            data = data_list[j]
            var = var_names[j]
            plot_col = colors[j]

            print("Variable = " + var)

            data_reg = data[var].where(data[region] == 0).mean(dim= ['latitude','longitude'], skipna=True)

            datadir = "/dx01/ivanov/data/CHIRPS/monsoon_humidheat_timing/"
            break_data = 'wet_dates_monsoon_' + region + '.csv'

            break_length = pd.read_csv(datadir+break_data, index_col = 'time')

            break_dates = []

            for index, row in break_length.iterrows():
                count = row['count_above1'] # Again, adjust these based on whether dry/wet spell
                date = index
                for addition in range(count):
                    date_add = datetime.strptime(date,'%Y-%m-%d') + timedelta(days = addition)
                    break_dates.extend([date_add])

            break_dates_final = list(set(break_dates))
            break_dates_final.sort()

            # Select data X number of days before and after monsoon onset
            buffer = 7

            break_length.drop_duplicates(subset = ['crossing_above1'], keep = 'first', inplace = True)
            break_length = break_length

            # Add in middle and last day of each break, rounding up if odd number of days
            break_days = []

            for index, row in break_length.iterrows():

                # Buffer before
                date = index
                date_datetime = datetime.strptime(date,'%Y-%m-%d')

                for addition in range(-1*buffer, 0):
                    backward = date_datetime + timedelta(days = addition)
                    break_days.extend([backward])

                #First day
                break_days.extend([date_datetime])

                #Middle day
                tot_len = row['count_above1']
                mid_len = round(row['count_above1']/2)
                mid_day = date_datetime + timedelta(days = mid_len)
                break_days.extend([mid_day])

                #Last day
                last_day = date_datetime + timedelta(days = tot_len)
                break_days.extend([last_day])

                # Buffer after
                for addition in range(1, buffer+1):
                    forward = last_day + timedelta(days = addition)
                    break_days.extend([forward])

            break_norm = list(break_days)

            data_spell = data_reg.sel(time = break_norm)

            # Convert to pandas dataframe
            data_df = data_spell.to_dataframe()

            # Add onset number and day number to dataframe
            buffer_range = 2*buffer + 3
            num_breaks = len(data_df[var])/buffer_range
            break_count = np.arange(1,num_breaks+1,1)

            steps = 2
            days_before = np.arange(-1*buffer, 1,1)
            days_after = np.arange(2*steps + 1, 2*steps + buffer + 1,1)
            days = np.hstack((days_before, [steps,2*steps],days_after))
            day_array = np.tile(days,int(num_breaks))
            onset_array = np.repeat(break_count,buffer_range)

            data_df['Break Number'] = onset_array
            data_df['Day Number'] = day_array

            # Average across all breaks
            group_mean = data_df.groupby('Day Number').mean()
            group_std = data_df.groupby('Day Number').std()
            group_mean['up_2std'] = group_mean[var] + 2*group_std[var]
            group_mean['down_2std'] = group_mean[var] - 2*group_std[var]
            group_mean['up_1std'] = group_mean[var] + group_std[var]
            group_mean['down_1std'] = group_mean[var] - group_std[var]

            # Plot
            labels = [f'Break {i}' for i in range(1, int(num_breaks)+1)]
            labels.append('Average')

            if j == 0:

                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontsize(10)

                mean_line = group_mean.reset_index().plot(ax = ax, x = 'Day Number', y = var, color = plot_col, label = labels[-1], linewidth = 3, kind = 'line')

                shading = ax.fill_between(group_mean.index,group_mean['up_1std'],group_mean['down_1std'], color = plot_col, alpha = shading_alpha)

                ax.axhline(y=0, color='white', linestyle='--', linewidth = 2)

                ax.axvspan(0, 4, alpha=0.25, color='grey')    
                
                if k == 2:
                    ax.set_ylabel('Wet Bulb Temperature Anomaly (C)', fontsize = 10)

                ax.get_legend().remove()
                ax.set_ylim([-2,2])
                ax.set_xlim([-7,11])
                ax.text(-7.5, 2.2, string.ascii_lowercase[k], size=20, weight='bold')

            if j == 1:
                ax6 = ax.twinx()

                mean_line = group_mean.reset_index().plot(ax = ax6, x = 'Day Number', y = var, color = plot_col, label = labels[-1], linewidth = 3, kind = 'line')

                shading = ax6.fill_between(group_mean.index,group_mean['up_1std'],group_mean['down_1std'], color = plot_col, alpha = shading_alpha)

                if k == 3:
                    ax6.set_ylabel('Temperature Anomaly (C)', fontsize = 10)
                    ax6.yaxis.label.set_color(plot_col)

                ax6.get_legend().remove()
                ax6.set_ylim([-4,4])
                ax6.tick_params(axis='y', colors=plot_col)

            if j == 2:
                ax7 = ax.twinx()
                group_mean = group_mean*1000

                for label in (ax7.get_xticklabels() + ax7.get_yticklabels()):
                    label.set_fontsize(10)

                mean_line = group_mean.reset_index().plot(ax = ax7, x = 'Day Number', y = var, color = plot_col, label = labels[-1], linewidth = 3, kind = 'line')

                shading = ax7.fill_between(group_mean.index,group_mean['up_1std'],group_mean['down_1std'], color = plot_col, alpha = shading_alpha)

                ax7.axhline(y=0, color='white', linestyle='--', linewidth = 2)

                if k == 3:
                    ax7.set_ylabel('Specific Humidity Anomaly (g/kg)', fontsize = 10)
                    ax7.yaxis.label.set_color(plot_col)
                
                ax7.get_legend().remove()
                ax7.set_ylim([-2.5,2.5])

                ax7.tick_params(axis='y', which='major', pad=40, right = False, colors=plot_col)
                
        ax.set_xticks([])
        ax.set_xticks([], minor=True)

        ax6.set_xticks([])
        ax6.set_xticks([], minor=True)   

        ax7.set_xticks([])
        ax7.set_xticks([], minor=True)
                                
        print(str(region) + ' wet spells plot is plotted')

    if k >= 4:
        
        if k == 4:
            
            i = 0
            
        if k == 5:
            
            i = 1
            
        region = regions[i]
        plot_title = plot_names[i]

        print('Region started: ' + region)

        mask = masks[i]

        for j in range(len(data_list)):

            data = data_list[j]
            var = var_names[j]
            plot_col = colors[j]

            print("Variable = " + var)

            data_reg = data[var].where(data[region] == 0).mean(dim= ['latitude','longitude'], skipna=True)

            onset_dates = onset_df[region]
            onset_dates = onset_df.loc[:,['year',region]]
            onset_dates[region] = pd.to_datetime(onset_dates[region], infer_datetime_format=True)

            if j == 2:
                onset_dates = onset_dates.iloc[:-1,:]
                onset_df = onset_df.iloc[:-1,:]

            buffer = 15
            onset_buffer = []

            for index, row in onset_df.iterrows():
                date = row[region]
                yearly_onset = datetime.strptime(date,'%Y-%m-%d')
                onset_buffer.extend([yearly_onset])

                for addition in range(1, buffer+1):
                    forward = yearly_onset + timedelta(days = addition)
                    backward = yearly_onset + timedelta(days = -1*addition)
                    onset_buffer.extend([backward, forward])

            onset_buffer_sorted = np.sort(list(onset_buffer))

            # Convert monsoon onset dates with/without buffer to datetime objects
            for ind, ts in enumerate(onset_buffer_sorted):
                onset_buffer_sorted[ind] = onset_buffer_sorted[ind].date()

            # Select daily max wet bulb temperatures during monsoon onset and surrounding
            data_onset = data_reg.sel(time = onset_buffer_sorted)

            # Convert to pandas dataframe
            data_df = data_onset.to_dataframe()

            # Add onset number and day number to dataframe
            data_df.sort_index()

            buffer_range = buffer*2+1
            num_years = len(data_df[var])/buffer_range
            year_count = np.arange(1,num_years+1,1)

            days = np.arange(-1*buffer, buffer+1,1)
            day_array = np.tile(days,int(num_years))
            onset_array = np.repeat(year_count,buffer_range)

            data_df['Year Number'] = onset_array
            data_df['Day Number'] = day_array

            if j == 2:
                data_df[var] = 1000*data_df[var]

            # Average across all breaks
            group_mean = data_df.groupby('Day Number').mean()
            group_std = data_df.groupby('Day Number').std()
            group_mean['up_2std'] = group_mean[var] + 2*group_std[var]
            group_mean['down_2std'] = group_mean[var] - 2*group_std[var]
            group_mean['up_1std'] = group_mean[var] + group_std[var]
            group_mean['down_1std'] = group_mean[var] - group_std[var]

            # Plot

            if j == 0:
                
                xticks = [-15,-10,-5,0,5,10,15]
                xlabels = [-15,-10,-5,'Onset',5,10,15]

                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontsize(10)

                mean_line = group_mean.reset_index().plot(ax = ax, x = 'Day Number', y = var, color = plot_col, linewidth = 3, kind = 'line')

                shading = ax.fill_between(group_mean.index,group_mean['up_1std'],group_mean['down_1std'], color = plot_col, alpha = shading_alpha)

                ax.axvspan(0, 17, alpha=0.25, color='grey')    
                
                if k == 4:
                    ax.set_ylabel('Wet Bulb Temperature Anomaly (C)', fontsize = 10)
                  
                ax.get_legend().remove()
                ax.set_ylim([-2,2])
                ax.set_xlim([-15,15])
                ax.text(-16, 2.2, string.ascii_lowercase[k], size=20, weight='bold')

            if j == 1:
                ax2 = ax.twinx()

                mean_line = group_mean.reset_index().plot(ax = ax2, x = 'Day Number', y = var, color = plot_col,  linewidth = 3, kind = 'line')

                shading = ax2.fill_between(group_mean.index,group_mean['up_1std'],group_mean['down_1std'], color = plot_col, alpha = shading_alpha)

                if k == 5:
                    ax2.set_ylabel('Temperature Anomaly (C)', fontsize = 10)
                    ax2.yaxis.label.set_color(plot_col)

                ax2.get_legend().remove()
                ax2.set_ylim([-4,4])
                ax2.tick_params(axis='y', colors=plot_col)

            if j == 2:
                ax3 = ax.twinx()

                mean_line = group_mean.reset_index().plot(ax = ax3, x = 'Day Number', y = var, color = plot_col, linewidth = 3, kind = 'line')

                shading = ax3.fill_between(group_mean.index,group_mean['up_1std'],group_mean['down_1std'], color = plot_col, alpha = shading_alpha)

                ax3.axhline(y=0, color='white', linestyle='--', linewidth = 2)

                if k == 5:
                    ax3.set_ylabel('Specific Humidity Anomaly (g/kg)', fontsize = 10)
                    ax3.yaxis.label.set_color(plot_col)

                ax3.get_legend().remove()
                ax3.set_ylim([-2.5,2.5])

                ax3.tick_params(axis='y', which='major', pad=40, right = False, colors=plot_col)

        print(str(region) + ' monsoon onset is plotted')

# Fixing xlabels

# Row 1
xticks2 = [-6,-4,-2,0,2,4,6,8,10]
xlabels2 = [-6,-4,-2,'Start','Middle','End',2,4,6]

axs[0,0].set_xticks(xticks2)
axs[0,0].set_xticklabels(xlabels2)   
axs[0,0].xaxis.set_tick_params(which='both', labelbottom=True)
axs[0,0].set_xlabel('Time Before/After Dry Spell', fontsize = 10)

for m, t in enumerate(axs[0,0].get_xticklabels()):
    if m > 2 and m < 6:
        t.set_rotation(45)
    else:
        t.set_rotation(0)
        
axs[0,1].set_xticks(xticks2)
axs[0,1].set_xticklabels(xlabels2)   
axs[0,1].xaxis.set_tick_params(which='both', labelbottom=True)
axs[0,1].set_xlabel('Time Before/After Dry Spell', fontsize = 10)

for m, t in enumerate(axs[0,1].get_xticklabels()):
    if m > 2 and m < 6:
        t.set_rotation(45)
    else:
        t.set_rotation(0)

# Row 2

xticks3 = [-6,-4,-2,0,2,4,6,8,10]
xlabels3 = [-6,-4,-2,'Start','Middle','End',2,4,6]

axs[1,0].set_xticks(xticks3)
axs[1,0].set_xticklabels(xlabels3)   
axs[1,0].xaxis.set_tick_params(which='both', labelbottom=True)
axs[1,0].set_xlabel('Time Before/After Event', fontsize = 10)

for m, t in enumerate(axs[1,0].get_xticklabels()):
    if m > 2 and m < 6:
        t.set_rotation(45)
    else:
        t.set_rotation(0)
        
axs[1,1].set_xticks(xticks3)
axs[1,1].set_xticklabels(xlabels3)   
axs[1,1].xaxis.set_tick_params(which='both', labelbottom=True)
axs[1,1].set_xlabel('Time Before/After Event', fontsize = 10)

for m, t in enumerate(axs[1,1].get_xticklabels()):
    if m > 2 and m < 6:
        t.set_rotation(45)
    else:
        t.set_rotation(0)
        
# Row 3
axs[2,0].set_xticks(xticks)
axs[2,0].set_xticklabels(xlabels)   
axs[2,0].xaxis.set_tick_params(which='both', labelbottom=True)
axs[2,0].set_xlabel('Time Before/After Event', fontsize = 10)
axs[0,0].set_title('NW',fontweight = 'bold')

for m, t in enumerate(axs[2,0].get_xticklabels()):
    if m == 3:
        t.set_rotation(45)
    else:
        t.set_rotation(0)
        
axs[2,1].set_xticks(xticks)
axs[2,1].set_xticklabels(xlabels)   
axs[2,1].xaxis.set_tick_params(which='both', labelbottom=True)
axs[2,1].set_xlabel('Time Before/After Event', fontsize = 10)
axs[0,1].set_title('BGD',fontweight = 'bold')

for m, t in enumerate(axs[2,1].get_xticklabels()):
    if m == 3:
        t.set_rotation(45)
    else:
        t.set_rotation(0)
                
plt.tight_layout()
plt.savefig('/home/ivanov/jupyternb/Monsoon_TW/figure3_forsubmission5.png')