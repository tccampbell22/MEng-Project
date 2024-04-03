

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import pandas as pd

from pyproj import Proj
from math import cos, asin, sqrt


plt.style.use('ggplot')



def calculate_power(speed, speeds, powers):
    """
    Calculate power for a given velocity by interpolation
    """
   
    return np.interp(speed, speeds, powers)


def find_best_locs(arr, n):
    #takes an array and returns a list of indices which would sort the array from high to low
   
    indices = np.argsort(arr)[::-1]#sort and put in decending order

    # Take the first n indices
    best_locs = indices[:n]
    

    return best_locs


def utm_to_latlon(easting, northing, zone_number=30, northern_hemisphere=True):
    # Transform UTM coordinates to latitude and longitude
    utm_proj = Proj(proj="utm", zone=zone_number, ellps="WGS84", north=northern_hemisphere)

    lon, lat = utm_proj(easting, northing, inverse=True)
    
    return lat, lon




def closest(data, lat, lon):
    
    return min(data, key=lambda p: distance(lat, lon, p['lat'], p['lon']))


def distance(lat1, lon1, lat2, lon2):
    
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))






def map_to_bus(lat,lon):

    # maps each generator to the closest location
    df_buses = pd.read_csv('LOPF_data/buses.csv')[:29]
    bus_location = []
    for i in range(len(df_buses)):
        bus_location.append({'lon': df_buses['x'][i], 'lat': df_buses['y'][i]})
    bus_names = df_buses['name'].values

    closest_bus_location = closest(bus_location, lat, lon)
    closest_bus_index = bus_location.index(closest_bus_location)
    bus=bus_names[closest_bus_index]

    return bus










def calculate_magnitude(u, v):
    #calculate the magnitude of the velocity components
    velocity = np.sqrt(u ** 2 + v ** 2)
    return velocity










def extend_range(data,v_data_start):
    #take the fft of the month-long data and use the result to extend it so its a year long
    
    year_len=24*365 
    
    step=1/len(data)
    t_ext=np.arange(0,year_len/len(data),step)

    fft_ = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(fft_), step)  
    
        
    sorted_indices = np.argsort(np.abs(fft_))
    top_indices = sorted_indices[-8:]  # Select the top 4 indices
    
    fft_0=np.zeros_like(fft_, dtype=np.complex128)
    fft_0[top_indices]=fft_[top_indices]
    

    # use the ifft to make a function from the fft values
    ext_data = np.real(ifft(fft_0, frequencies, t_ext, len(data)))
    
    
    year_date = datetime.strptime(v_data_start, '%Y-%m-%d %H:%M:%S')
    n_year = year_date.year+1

    year_end= str(n_year)+'-01-01 00:00:00'
    

    year_end = datetime.strptime(year_end, '%Y-%m-%d %H:%M:%S')
    data_start = datetime.strptime(v_data_start, '%Y-%m-%d %H:%M:%S')
    
    #time in hrs from data time to 31/12
    hours_difference = int((year_end - data_start).total_seconds() / 3600)

    data_dec=ext_data[0:hours_difference+1] #crops sample from data_start date to 31/12
    jan_data=ext_data[hours_difference+1:year_len] #crops sample from 31/12 to data_start
    
    year_data=np.append(jan_data,data_dec)   
    
    return year_data


def ifft(fft_result, frequencies, t, fs):
    
    N = len(fft_result)
    x_t = np.zeros_like(t, dtype=np.complex128)

    for k in range(N):
        x_t += fft_result[k] * np.exp(1j * 2 * np.pi * frequencies[k] * t)

    return x_t / fs  # Scaling by the number of samples




def calculate_direction(u, v, unit="deg"):
    
    direction_rad = np.arctan2(u, v)  # calculates direction of the vector in radians
    if unit == "deg":
        direction_deg = np.degrees(direction_rad)  # converts to degrees if user specifies unit = "deg"
        return direction_deg + 180  # to set north to zero degrees
    return direction_rad + np.pi/2





def _normalize_angle(degree):
    '''
    Normalizes degrees to be between 0 and 360
    
    Parameters
    ----------
    degree: int or float

    Returns
    -------
    new_degree: float
        Normalized between 0 and 360 degrees
    '''
    #taken from mhkit and unedited
    
    # Set new degree as remainder
    new_degree = degree%360
    # Ensure positive
    new_degree = (new_degree + 360) % 360 
    return new_degree






def principal_flow_directions(directions, width_dir):
    '''
    Calculates principal flow directions for ebb and flood cycles
    
    The weighted average (over the working velocity range of the TEC) 
    should be considered to be the principal direction of the current, 
    and should be used for both the ebb and flood cycles to determine 
    the TEC optimum orientation. 

    Parameters
    ----------
    directions: pandas.Series or numpy.ndarray
        Flow direction in degrees CW from North, from 0 to 360
    width_dir: float 
        Width of directional bins for histogram in degrees

    Returns
    -------
    principal directions: tuple(float,float)
        Principal directions 1 and 2 in degrees

    Notes
    -----
    One must determine which principal direction is flood and which is 
    ebb based on knowledge of the measurement site.
    '''
    #taken from mhkit and unedited
    
    if isinstance(directions, np.ndarray):
        directions=pd.Series(directions)
    assert(all(directions>=0) and all(directions<=360),
           'flood must be between 0 and 360 degrees')

    # Number of directional bins 
    N_dir=int(360/width_dir)
    # Compute directional histogram
    H1, dir_edges = np.histogram(directions, bins=N_dir,range=[0,360], density=True) 
    # Convert to perecnt
    H1 = H1 * 100 # [%]
    # Determine if there are an even or odd number of bins
    odd = bool( N_dir % 2  )
    # Shift by 180 degrees and sum
    if odd:
        # Then split middle bin counts to left and right
        H0to180    = H1[0:N_dir//2] 
        H180to360  = H1[N_dir//2+1:]
        H0to180[-1]   += H1[N_dir//2]/2
        H180to360[0]  += H1[N_dir//2]/2
        #Add the two
        H180 = H0to180 + H180to360
    else:
        H180 =  H1[0:N_dir//2] + H1[N_dir//2:N_dir+1]

    # Find the maximum value
    maxDegreeStacked = H180.argmax()
    # Shift by 90 to find angles normal to principal direction
    floodEbbNormalDegree1 = _normalize_angle(maxDegreeStacked + 90.)
    # Find the complimentary angle 
    floodEbbNormalDegree2 = _normalize_angle(floodEbbNormalDegree1+180.)
    # Reset values so that the Degree1 is the smaller angle, and Degree2 the large
    floodEbbNormalDegree1 = min(floodEbbNormalDegree1, floodEbbNormalDegree2)
    floodEbbNormalDegree2 = floodEbbNormalDegree1 + 180.
    # Slice directions on the 2 semi circles
    d1 = directions[directions.between(floodEbbNormalDegree1,
                                       floodEbbNormalDegree2)] 
    d2 = directions[~directions.between(floodEbbNormalDegree1,
                                       floodEbbNormalDegree2)] 
    # Shift second set of of directions to not break between 360 and 0
    d2 -= 180.
    # Renormalize the points (gets rid of negatives)
    d2 = _normalize_angle(d2)
    # Number of bins for semi-circle
    n_dir = int(180/width_dir)
    # Compute 1D histograms on both semi circles
    Hd1, dir1_edges = np.histogram(d1, bins=n_dir,density=True)
    Hd2, dir2_edges = np.histogram(d2, bins=n_dir,density=True)
    # Convert to perecnt
    Hd1 = Hd1 * 100 # [%]
    Hd2 = Hd2 * 100 # [%]
    # Principal Directions average of the 2 bins
    PrincipalDirection1 = 0.5 * (dir1_edges[Hd1.argmax()]+ dir1_edges[Hd1.argmax()+1])
    PrincipalDirection2 = 0.5 * (dir2_edges[Hd2.argmax()]+ dir2_edges[Hd2.argmax()+1])+180.0

    return PrincipalDirection1, PrincipalDirection2 
