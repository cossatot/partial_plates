import sqlite3 as sql
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

import pandas as pd

import sys
sys.path.append('../../../partial_plates/')
import partial_plates as pp

# get data

# velocity data
con = sql.connect('./escape_data.sqlite')

vels = pd.read_sql_query('SELECT * from escape_pts', con, 
                         index_col='ogc_fid')

def wkt_point_parse(wkt_str):
    lonlat = wkt_str.split()[1:]
    lon = float(lonlat[0][1:])
    lat = float(lonlat[1][:-1])
    
    return lon, lat

vels['lon'] = 0.
vels['lat'] = 0.

for i, row in vels.iterrows():
    lon, lat = wkt_point_parse(row.WKT_GEOMETRY)
    vels.ix[i, 'lon'] = lon
    vels.ix[i, 'lat'] = lat

del vels['WKT_GEOMETRY']
vels = vels[vels.lat < 20]

# faults
faults = pd.read_sql_query('SELECT * from escape_faults', con,
                           index_col='ogc_fid')

def wkt_linestring_parse(wkt_str):
    pt0, pt1 = wkt_str.split(',')
    lon0, lat0 = pt0.split()[1:]
    lon0 = lon0[1:]
    lat0 = lat0[:-1]
    
    lon1, lat1 = pt1.split()
    lat1 = lat1[:-1]
    
    return np.float_((lon0, lat0, lon1, lat1))

faults['lon0'] = 0.
faults['lat0'] = 0.
faults['lon1'] = 0.
faults['lat1'] = 0.

for i, row in faults.iterrows():
    lon0, lat0, lon1, lat1 = wkt_linestring_parse(row.WKT_GEOMETRY)
    
    faults.ix[i, 'lon0'] = lon0
    faults.ix[i, 'lat0'] = lat0
    faults.ix[i, 'lon1'] = lon1
    faults.ix[i, 'lat1'] = lat1

# Pole
pole_lon = 30.
pole_lat = -30.

poles_df = pd.DataFrame(index=[0], columns=['lon', 'lat', 'rate_deg_Myr', 
                                            'plate'])
poles_df.ix[0] = [pole_lon, pole_lat, -1., 'P']

# get radius and azimuth for vels and faults
faults['az0'] = 0.
faults['az1'] = 0.

for i, row in faults.iterrows():
    az0, az1 = pp.eulers.azimuth(np.array((row.lon0, row.lon1)), 
                                 np.array((row.lat0, row.lat1)),
                                 pole_lon, pole_lat, input_coords='degrees')
    faults.ix[i, 'az0'] = az0
    faults.ix[i, 'az1'] = az1

faults['r0'] = 0.
faults['r1'] = 0.

for i, row in faults.iterrows():
    r0, r1 = pp.eulers.arc_distance(np.array((row.lon0, row.lon1)), 
                                    np.array((row.lat0, row.lat1)),
                                    pole_lon, pole_lat, input_coords='degrees')
    faults.ix[i, 'r0'] = r0
    faults.ix[i, 'r1'] = r1

vels['r'] = pp.eulers.arc_distance(vels.lon, vels.lat, pole_lon, pole_lat,
                                   input_coords='degrees')
vels['az'] = pp.eulers.azimuth(vels.lon, vels.lat, pole_lon, pole_lat,
                               input_coords='degrees')


# do inversion
fault_rates, mod_vels = pp.faults.do_slip_rate_inversion(vels, faults)

# plot results
def get_cm(val, vmin=0., vmax=1., cmap=cm.viridis):
    norm_val = (val - vmin) / (vmax - vmin)
    return cmap(norm_val)


f = plt.figure(figsize=(11,6))
ax = f.add_subplot(121)

s = ax.scatter(vels.lon, vels.lat, c=mod_vels,
               vmin=0., vmax=1.,
               cmap='plasma', s=30, lw=0)
plt.colorbar(s)

for i, row in faults.iterrows():
    col = get_cm(fault_rates[i], vmin=-1, vmax=1, cmap=cm.PiYG)
    ax.plot((row.lon0, row.lon1), (row.lat0, row.lat1), color=col)

ax1 = f.add_subplot(122, 
                    #polar=True
                    )

s = ax1.scatter(vels.az, vels.r, c=mod_vels,
               vmin=0., vmax=1.,
               cmap='plasma', s=30, lw=0)
plt.colorbar(s)

for i, row in faults.iterrows():
    col = get_cm(fault_rates[i], vmin=-1, vmax=1, cmap=cm.PiYG)
    ax1.plot((row.az0, row.az1), (row.r0, row.r1), color=col)


plt.show()
