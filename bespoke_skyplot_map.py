# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 18:54:36 2021

@author: Satish Narasimhan
"""

from datetime import datetime
from pytz import timezone
import spiceypy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import geocoder
import static as s

# Load the SPICE kernels via a meta file
spiceypy.furnsh('./skyplot/kernel_meta.txt')

# Set the geographic longitude / latitude for Bengaluru, India
loc = 'Bengaluru, India' #Bengaluru, India, Sydney, Australia, Stuttgart, Germany
tzone = 'Europe/Berlin' # Asia/Kolkata, Australia/Sydney, Europe/Berlin

format = "%Y-%m-%d %H:%M:%S"
# Current time in UTC
local_tz = datetime.now(timezone(tzone)) #local time zone
tz = local_tz.strftime(format)
#print(local_tz)
# Convert to local timezone i.e. in this case Asia/Kolkata time zone
datetime_obj = local_tz.astimezone(timezone('UTC')) #Asia/Kolkata, UTC, Asia/Kuwait, Europe/Berlin, Australia/Sydney, Asia/Singapore
#print(datetime_obj.strftime(format))

# Create an initial date-time object that is converted to a string
# datetime_now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
datetime_now = datetime_obj.strftime(format)
#print (datetime_now)

# Convert to Ephemeris Time (ET) using the SPICE function utc2et
datetime_et = spiceypy.utc2et(datetime_now)
#print(datetime_et)

g = geocoder.opencage( loc , key = s.open_cage_api_key)
coord = g.latlng
city = g.city

geo_lat_dec = (round(g.latlng[0],4))
geo_long_dec = (round(g.latlng[1],4))

geo_lat_rad = np.radians(geo_lat_dec)

print (city, geo_lat_dec, geo_long_dec)
print (tz)

# We want to compute the coordinates for different Solar System bodies as seen
# from our planet. First, a pandas dataframe is set that is used to append the
# computed data
solsys_df = pd.DataFrame()

# Add the ET and the corresponding UTC date-time string
solsys_df.loc[:, 'ET'] = [datetime_et]
solsys_df.loc[:, ' Hrs'] = [datetime_now]

# Set a dictionary that lists some body names and the corresponding NAIF ID
# code. Mars has the ID 499, however the loaded kernels do not contain the
# positional information. We use the Mars barycentre instead
# Uranus, Neptune, Pluto cannot be seen without a telescope
SOLSYS_DICT = {'SUN': 10, 'VENUS': 299, 'MOON': 301, 'MARS': 4, 'JUPITER': 5, 'SATURN': 6, 'URANUS':7, 'NEPTUNE':8, 'PLUTO': 9}

# Each body shall have an individual color; set a list with some colors
BODY_COLOR_ARRAY = ['y', 'tab:orange', 'tab:gray', 'tab:red', 'tab:blue', 'tab:purple', 'tab:pink' ,'tab:green', 'tab:brown']

# Now we want the coordinates in equatorial J2000. For this purpose we
# iterate through all celestial bodies
for body_name in SOLSYS_DICT:

    # First, compute the directional vector of the body as seen from Earth in
    # J2000
    solsys_df.loc[:, f'dir_{body_name}_wrt_earth_equ'] = solsys_df['ET'] \
        .apply(lambda x: spiceypy.spkezp(targ=SOLSYS_DICT[body_name], \
                                         et=x, \
                                         ref='J2000', \
                                         abcorr='LT+S', \
                                         obs=399)[0])

    # Compute the longitude and latitude values in equatorial J2000
    # coordinates
    solsys_df.loc[:, f'{body_name}_long_rad_equ'] = solsys_df[f'dir_{body_name}_wrt_earth_equ'] \
                                                        .apply(lambda x: spiceypy.recrad(x)[1])
    solsys_df.loc[:, f'{body_name}_lat_rad_equ'] = solsys_df[f'dir_{body_name}_wrt_earth_equ'] \
                                                        .apply(lambda x: spiceypy.recrad(x)[2])

    # Apply the same logic as shown before to compute the longitudes for the
    # matplotlib figure
    solsys_df.loc[:, f'{body_name}_long_rad4plot_equ'] = \
        solsys_df[f'{body_name}_long_rad_equ'] \
            .apply(lambda x: -1*((x % np.pi) - np.pi) if x > np.pi \
                   else -1*x)
                
# Before we plot the data, let's add the Ecliptic plane for the visualisation.
# In ECLIPJ2000 the Ecliptic plane is the equator line (see corresponding
# figure. The latitude is 0 degrees.

# First, we create a separate dataframe for the ecliptic plane
eclip_plane_df = pd.DataFrame()

# Add the ecliptic longitude and latitude values for the plane. Note: here,
# we need to use pi/2 (90 degrees) as the latitude, since we will apply a
# SPICE function that expects spherical coordinates
eclip_plane_df.loc[:, 'ECLIPJ2000_long_rad'] = np.linspace(0, 2*np.pi, 100)
eclip_plane_df.loc[:, 'ECLIPJ2000_lat_rad'] = np.pi / 2.0

# Compute the directional vectors of the ecliptic plane for the different
# longitude values (the latitude is constant). Apply the SPICE function sphrec
# to transform the spherical coordinates to vectors. r=1 is the distance,
# here in our case: normalised distance
eclip_plane_df.loc[:, 'ECLIPJ2000_direction'] = \
    eclip_plane_df \
        .apply(lambda x: spiceypy.sphrec(r=1, \
                                         colat=x['ECLIPJ2000_lat_rad'], \
                                         lon=x['ECLIPJ2000_long_rad']), \
               axis=1)
            
# Compute a transformation matrix between ECLIPJ2000 and J2000 for a fixed
# date-time. Since both coordinate system are inertial (not changing in time)
# the resulting matrix is the same for different ETs
ecl2equ_mat = spiceypy.pxform(fromstr='ECLIPJ2000', \
                              tostr='J2000', \
                              et=datetime_et)

# Compute the direction vectors of the Ecliptic plane in J2000 using the
# transformation matrix
eclip_plane_df.loc[:, 'j2000_direction'] = \
    eclip_plane_df['ECLIPJ2000_direction'].apply(lambda x: ecl2equ_mat.dot(x))

# Compute now the longitude (and matplotlib compatible version) and the
# latitude values using the SPICE function recrad
eclip_plane_df.loc[:, 'j2000_long_rad'] = \
    eclip_plane_df['j2000_direction'].apply(lambda x: spiceypy.recrad(x)[1])

eclip_plane_df.loc[:, 'j2000_long_rad4plot'] = \
    eclip_plane_df['j2000_long_rad'] \
        .apply(lambda x: -1*((x % np.pi) - np.pi) if x > np.pi \
               else -1*x)

eclip_plane_df.loc[:, 'j2000_lat_rad'] = \
    eclip_plane_df['j2000_direction'].apply(lambda x: spiceypy.recrad(x)[2])
    
# Compute the Horizon in Equatorial Coordiantes
# Some literature
# Horizontal System: http://star-www.st-and.ac.uk/~fv/webnotes/chapter7.htm
# Local Sideral Time: http://star-www.st-and.ac.uk/~fv/webnotes/chapter6.htm
# Greenwich Mean Sideral Time: https://www.astro.umd.edu/~jph/GST_eqn.pdf

# First we define a Dataframe that contains the horizon information, namely
# Elevation: 0 Degrees
# Azimuth: from 0 to 360 degrees
azim_rad = np.linspace(-np.pi, np.pi, 100)
elev_rad = np.zeros(100)

# Local Sideral Time (ST)
GST_func = lambda dt: 6.6208844 \
                      + 0.0657098244 * dt.timetuple().tm_yday \
                      + 1.00273791 * (datetime_obj.hour + (datetime_obj.minute / 60.0))
GST = GST_func(datetime_obj)

# Compute the Local Sideral Time (LST)
# PLease note that the longitude needs to be converted to longitude west (thus 360 degrees minus
# longitude)
LST = (GST - ((360.0 - geo_long_dec) / 360.0) * 24.0)
LST_rad = np.radians((LST / 24.0) * 360.0)

# Lambda functions for declination delta; H1, H2 and H to compute the right ascension alpha
# afterwards
# declination:
dec_rad = np.arcsin(np.sin(elev_rad)*np.sin(geo_lat_rad) \
          + np.cos(elev_rad)*np.cos(geo_lat_rad)*np.cos(azim_rad))

# H functions:
H1 = (-1.0 * np.sin(azim_rad)) * np.cos(elev_rad) / np.cos(dec_rad)
H2 = (np.sin(elev_rad) - np.sin(dec_rad)*np.sin(geo_lat_rad)) \
     / (np.cos(dec_rad)*np.cos(geo_lat_rad))
H = np.arctan2(H1, H2)

# right ascension:
ra_rad = (LST_rad - H) % (2.0 * np.pi)

# Lambda function to convert the alpha values for matplotlib
ra_plot_func = lambda x: -1.0*((x % np.pi) - np.pi) if x > np.pi \
                         else -1.0*x

# Right ascension for plotting
ra_plot = [ra_plot_func(x) for x in ra_rad]
    
# We plot now the data in equatorial J2000. Again with a dark background and
# the same properties as before
plt.style.use('dark_background')
plt.figure(figsize=(15, 12))
plt.subplot(projection="aitoff")
plt.title(f'{city} {tz}', fontsize=10)

# Iterate through the celestial bodies and plot them
for body_name, body_color in zip(SOLSYS_DICT, BODY_COLOR_ARRAY):

    plt.plot(solsys_df[f'{body_name}_long_rad4plot_equ'], \
             solsys_df[f'{body_name}_lat_rad_equ'], \
             color=body_color, marker='o', linestyle='None', markersize=12, \
             label=body_name.capitalize())

# Plot the Ecliptic plane as a blue dotted line
plt.plot(eclip_plane_df['j2000_long_rad4plot'], \
         eclip_plane_df['j2000_lat_rad'], color='tab:blue', linestyle='None', \
         marker='o', markersize=2)
    
# Plot the Horizon as a green line
plt.plot(ra_plot, \
         dec_rad, \
         color='tab:red', marker='o', linestyle='None', markersize=2)


# Convert the longitude values finally in right ascension hours
plt.xticks(ticks=np.radians(np.arange(-150, 180, 30)),
           labels=['10 h', '8 h', '6 h', '4 h', '2 h', '0 h', \
                   '22 h', '20 h', '18 h', '16 h', '14 h'])

# Plot the labels
plt.xlabel('Right ascension in hours \n(South)')
plt.ylabel('Declination in deg. \n(East)')

# Create a legend and grid
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig('j2000_sky_map_horizon.png', dpi=300)