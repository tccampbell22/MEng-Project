#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:30:27 2024

@author: thomascampbell
"""

import pandas as pd
import numpy as np
from Support_functions32 import *
import snapshots
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


plt.style.use('ggplot')
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature
from statistics import mean
from adjustText import adjust_text


def site_p_max_pu_point(Tinfo, velocities, site, v_data_start):
    """
    

    Parameters
    ----------
    Tinfo : conatins a dictionary with the turbine name, diameter, nominal power and power curve
    velocities : velocity timeseries
    site :  name of the site being considered
    v_data_start : start date of the v timeseries, if it is to be extended

    Returns
    -------
    cfs : the space averaged capacity factor timeseries for use in PyPSA

    """
    
    if site=='Mor':
            site='Wes-Ang-Dem-Mor' #name change between rsc and future dep scenarios
            
        
    pnom=Tinfo['P']/1E6
    speeds=Tinfo['speeds']
    powers=Tinfo['powers']
    #Take turbine parameters from class defined in Turbine_Power_Curve

    if len(velocities)==2:#i.e. 2 directions not magnitude
        u=velocities[0]
        v=velocities[1]
        if len(u)<(24*365): #can't extend range if a magnitude
            u=extend_range(u,v_data_start)
            v=extend_range(v,v_data_start)
        velocities=calculate_magnitude(u,v)
        velocities=np.append(velocities,velocities[-1])
    else:
        if len(velocities)<(24*365):
            #for magnitude datasets duplicating the set to a year's legnth is most accurate
            velocities=np.tile(velocities,int(24*365/len(velocities)))
    
    #assume a 10% velocity loss due to wake and 15% due to flow impedance, so 25%
    velocities=velocities*0.75
    
    powers=calculate_power(velocities,speeds,powers)
    powers[powers>(pnom)]=pnom
    #power calculated, due to rounding/curve fitting errors power is occassionally fractioanlly higher than pnom
    cfs=powers/pnom
   
    return cfs
    
        

    
def site_p_max_pu_field(Tinfo, capacity, site_name, folder, plotting, Pmax, do_depth, v_data_start):
    """
    

    Parameters
    ----------
    Tinfo : 
    year : year of optimisation
    capacity : the installed capacity for the site, if p_nom_extendable should be left as None 
    site_name : name of site
    folder : Tlocation to retrieve elocity timeseries from
    plotting : "on" or "off", choose whether to plot graphs
    Pmax : the maximum installable capacity for the site

    Returns
    -------
    x_coords : utm x-cooridnates for the site
    y_coords : utm y-coordinates for the site
    site_opt_cf : capacity factors timeseries averaged across the 5 best turbine locations
    avg_cap_factors : time average capacity factor data for use in plotting
    Pmax : the maximum installable capacity for the site

    """
    

    x_coords = np.load(folder+'x_coords.npy')
    y_coords = np.load(folder+'y_coords.npy')
    u_vels=np.load(folder+'u_all_time_cut.npy')
    v_vels=np.load(folder+'v_all_time_cut.npy')
    #load coordinate and velocity data from folder
    
    
    velocities = 0.75*calculate_magnitude(u_vels, v_vels) 
    
    # Determine mean flood and ebb directions
    thetas = calculate_direction(u_vels, v_vels)
    mean_thetas = np.mean(thetas, axis=1)
    ebb, flood = principal_flow_directions(mean_thetas, width_dir=1)
    
    
    principal_dir= 90-0.5*((ebb-180)+flood) #average of flood + ebb direction
    #flood and ebb angles are from north, subtract from 90 to get them from horizontal
    
    pnom = Tinfo['P']/1E6
    turbine_diameter = Tinfo['D']
    speeds = Tinfo['speeds']
    power = Tinfo['powers']
    #retreive turbine characteristics from Tinfo dictionary

    Area_t=0.25*np.pi*turbine_diameter**2

    powers=calculate_power(velocities,speeds,power)
    
    powers[powers>(pnom)] = pnom
    
    cap_factors = powers/pnom

    avg_cap_factors=np.mean(cap_factors, axis=0)
    
    #calculating powers as in site_p_max_pu_point
    
    min_lateral= 2.5*turbine_diameter
    min_principal= 10*turbine_diameter
    #threshold is based on marine energy wales assumptions
    
    
    best_locs=check_Spacing(avg_cap_factors,x_coords,y_coords,turbine_diameter, folder, principal_dir, plotting, do_depth, min_lateral, min_principal)
    
    #check Spacing reduces the number of coordinate locations to only those which are feasible
    
    
    if Pmax == "Get":
        
        Pmax=pnom*len(best_locs)
        print('no of locs:'+str(len(best_locs)))
        print(Pmax)
        
    avg_cfs=[]
    
    if capacity==None:
        
        capacity=np.linspace(pnom,Pmax,5)
        n_turbines=np.round(capacity/pnom,0)
        
        for n in n_turbines:
            
            opt_cfs=cap_factors[:,best_locs[0:int(n)]]
            avg_cf=np.mean(opt_cfs)
            avg_cfs.append(avg_cf)
        
        if plotting=='on':
            plt.plot(capacity,avg_cfs)
            plt.title('Variance of Capacity Factor with Installed Capacity')
            plt.xlabel('Installed Capacity (MW)')
            plt.ylabel('Site-average CF')
            plt.xticks([0,50,100,150,200,Pmax], [0,50,100,150,200,'MAX'])
            plt.xlim(0,Pmax+10)
            plt.savefig('cpvsinstalled.png')
            plt.show()
            #display a graph highlighting the variation in p_max_pu with installed capcity
        
        opt_uvels=u_vels[:,best_locs[0:5]]
        opt_vvels=v_vels[:,best_locs[0:5]]
        #sets the p_max_pu for PyPSA to be the average of 5 best turbine locs
        
        
    else: 
         
         n_turbines=int(capacity/pnom)
    
         opt_uvels=u_vels[:,best_locs[0:n_turbines]]
         opt_vvels=v_vels[:,best_locs[0:n_turbines]]
         #where a set capacity is to be installed, the p_max_pu of the site can be calculated more accurately
    
    velocities_sum=0
    
    for loc in range(len(opt_uvels[0])):
        
        u = extend_range(opt_uvels[:,loc],v_data_start)
        v = extend_range(opt_vvels[:,loc],v_data_start)
        vels=0.75*calculate_magnitude(u,v)
            
        velocities_sum += vels
        
    site_opt_vel=velocities_sum/len(opt_uvels[0])

    powers=calculate_power(site_opt_vel,speeds,power)
    
    powers[powers>(pnom)]=pnom
    
    site_opt_cf=powers/pnom
    
    site_opt_cf=np.append(site_opt_cf,site_opt_cf[-1])
    

    #averages the optimal cfs in space to create a singular timeseries for the site
    
    
    if plotting == "on":

        plot_sitefield(x_coords, y_coords, avg_cap_factors, site_name)
        
        #plots a colourmap of the site overlaid with a map showing its actual location
    
        
    return x_coords,y_coords,site_opt_cf, Pmax




def check_Spacing(avg_cap_factors,x, y, D, folder, principal_dir, plotting, do_depth, min_lateral, min_principal):
        """
    

        Parameters
        ----------
        avg_cap_factors : time-averaged capacity factor for each location
        x : x-coordinate of each location
        y : y-cooridnate of each location
        D : turbine diameter
        threshold : minimum spacing between turbines, 10*D is used in this study
        do_depth : choose whether to include bathymetry data in considering the feasiblility of a turbine location

        Returns
        -------
        best_locs : a list of indices which will return all feasible turbine locations from highest to lowest avg cf

        """
        
        best_locs=find_best_locs(avg_cap_factors,len(x))
        x_sorted=x[best_locs]
        y_sorted=y[best_locs]
        avg_cfs_sorted=avg_cap_factors[best_locs]
        #find the indices which order the location from highest to lowest average capcity factor, the sort each list
        
        
        if do_depth == True: 
            depth=np.load(folder+'bathy.npy')
            #retrieves site bathymetry from same folder as velocity data
            
            depths_sorted=depth[best_locs]
        
            
            min_depth2=D+10
            #min depth is based on EMEC standards
            vals_to_delete=[]
        
            for j in range(len(depths_sorted)):
                min_depth1=0.25*depths_sorted[j]+D+5
                #min depth is based on EMEC standards
                
                if min_depth1>=min_depth2:
                    min_depth=min_depth1
                else:
                    min_depth=min_depth2
                    
                if depths_sorted[j]<min_depth:
                
                    vals_to_delete.append(j)
                    
                    
            depths_sorted=np.delete(depths_sorted,vals_to_delete)
            x_sorted=np.delete(x_sorted,vals_to_delete)
            y_sorted=np.delete(y_sorted,vals_to_delete)
            avg_cfs_sorted=np.delete(avg_cfs_sorted,vals_to_delete)
            best_locs=np.delete(best_locs,vals_to_delete)
            #if location is too shallow, removes it from the list of feasible sites
        
        i=0
        while i < len(x_sorted):
            
            vals_to_delete=[]
                
         
            for j in range(i+1,len(x_sorted)): 
                
                dx=x_sorted[i]-x_sorted[j]
                dy=y_sorted[i]-y_sorted[j]
                phi=np.radians(principal_dir)
                principal_distance=np.abs(dx*np.cos(phi)-dy*np.sin(phi))
                lateral_distance=np.abs(dx*np.sin(phi)+dy*np.cos(phi))
                
                
                if principal_distance < min_principal and lateral_distance < min_lateral:
                    
                    
                    vals_to_delete.append(j)
                    
                
            avg_cfs_sorted=np.delete(avg_cfs_sorted,vals_to_delete)
            x_sorted=np.delete(x_sorted,vals_to_delete)
            y_sorted=np.delete(y_sorted,vals_to_delete)
            best_locs=np.delete(best_locs,vals_to_delete)
            
            i+=1
            #if location is too close to another which has a higher capacity factor, it is removed from the
            #list of feasible sites
            
        if plotting == "on":
        
            plt.scatter(x,y,s=1)
            plt.show()
            
            plt.scatter(x_sorted,y_sorted,s=1)
            plt.show()
            
            #plot a graph of site locations before and after feasibility checks
                           
                    
        return best_locs


def add_tidal_generator(network, site_name, data_type, Tinfo, year, start, end, vdata, time_step, capex='not given', plotting='on',coords="from file", Pmax="Get", new_site_data=None, capacity=None, do_depth=True, v_data_start='2050-08-01 00:00:00'):
    """
    

    Parameters
    ----------
    network : imports the PyPSA network for editting
    site_name : name of site to be updated or added
    data_type : "field" or "point" type data
    Tinfo : turbine_ parameters
    capex : capital cost for the site. The default is 'not given', in which case tstream default will be taken
    plotting : 'on' or 'off', choose to plot graphs
    lat : latitude of site, if field data this can be taken "from file"
    lon : longitude of site, if field data this can be taken "from file"
    Pmax : maximum installable capacity for the site
    new_site_data : output of the function, has to be defined if not altered so is set to None
    capacity : installed capacity for the site if p_nom is set, otherwise this is None
    do_depth :choose whether to consider the site depth
    Returns
    -------
    new_site_data : dictionary contaning the name, coordinates and maximum capacity of the new site

    """
    
    if data_type=='field':
        
        x_coords, y_coords, capacity_factors, max_capacity = site_p_max_pu_field(Tinfo, capacity, site_name, vdata, plotting, Pmax, do_depth, v_data_start)
        
        
    
        pmax_pu = p_max_pu(start, end, time_step, capacity_factors, site_name, year)
        
        #site_p_max_pu_field calculates the average cf time series and max_ capacity for the site, p_max_pu trims 
        #avg cf so that it has the same index range as the PyPSA optimisation
        
        lat, lon = utm_to_latlon(x_coords[0],y_coords[0])
        
        
        
    if data_type=='point':
        
        capacity_factors=site_p_max_pu_point(Tinfo, vdata, site_name, v_data_start)
        
        pmax_pu=p_max_pu(start, end, time_step, capacity_factors, site_name, year)
        
        #site_p_max_pu_point calculates the average cf time series and max_ capacity for the site, p_max_pu trims 
        #avg cf so that it has the same index range as the PyPSA optimisation
        
        max_capacity=Pmax
        

    
        
    if site_name in network.generators.index:
        
        print("Site already in network, p_max_pu for "+ str(site_name) +" updated")
        #this prevents an error where two sites fo the same name are added
        
        
    else:
        
        if coords=="from file":
            lat,lon=utm_to_latlon(x_coords[0],y_coords[0])
        else:
            lat,lon=utm_to_latlon(coords[0],coords[1])
                
        if capex=='not given':
            capex=network.generators.capital_cost['Mey'] #assumes same capex as other TS gens
        
        bus = map_to_bus(lat,lon)
        network.add("Generator", site_name, carrier='Tidal stream', type= 'Tidal stream', bus=bus, ramp_limit_up=1, ramp_limit_down=1, capital_cost=capex)
        #adds the new generator to the network
        
        
        
        if network.generators.loc["Mey", "p_nom_extendable"] == True:
            
            network.generators.loc[site_name,"p_nom_max"] = max_capacity
            network.generators.loc[site_name, "p_nom_extendable"] = True
            new_site_data={'Site Name': site_name, 'lat': lat,'lon': lon, 'max capacity': max_capacity}
            
        
        else: 
            network.generators.loc[site_name,"p_nom"]=capacity
            new_site_data={'Site Name': site_name, 'lat': lat,'lon': lon, 'max capacity': capacity}
            
        
        print(network.generators.loc[site_name])
        
    network.generators_t['p_max_pu'][site_name]=pmax_pu.values
    
    return new_site_data
    

def p_max_pu(start,end, time_step, capacity_factors,site_name,year):
    """
    

    Parameters
    ----------
    start : start time of optimisation
    end : end time of optimisation
    time_step : 'H' or '0.5H', whether data is collected hourly or half-hourly
    capacity_factors : year-long space-averaged capacity factor timeseries
    site_name : name of the new site
    year : year of optimisation

    Returns
    -------
    df_p_max_pu : space-averaged capacity factor timeseries in the range [start:end]

    """
    
    freq = snapshots.write_snapshots(start, end, time_step)
    
    path='../data/renewables/Marine/tidal_stream_'+str(year)+'_full.xlsx'
    df_tidal_stream=pd.read_excel(path,index_col=0)
    #read in existing cf data to add to it
    
    

    df_tidal_stream[site_name]=capacity_factors
    #add new column with new data
    
    
    df_tidal_stream=df_tidal_stream[site_name]
    df_tidal_stream.index = df_tidal_stream.index.round(freq)
    df_tidal_stream = df_tidal_stream.resample(freq).interpolate('polynomial', order=2)
    #take the time index from the other tidal stream dataset, round it to freq 
    #increments so it is compatible with PyPSA
    
    start_ = pd.to_datetime(start)
    end_ = pd.to_datetime(end)
    
    df_p_max_pu=df_tidal_stream.loc[start_:end_]
    #splice the list to get the values in the range of interest
    
    return df_p_max_pu

def plot_sites(map_type, network, extent=[-8.09782, 2.40511, 48.5, 60], new_site=None, marker_scaler=2, zooms=None, color = 'deepskyblue'):
    """
    

    Parameters
    ----------
    map_type : "optimal" (p_nom_opt), "potential" (p_nom_max) or "installed" (p_nom)
    year : Tyear of optimisation
    network : import optimisation network
    extent : the size of grid square to be plotted. The default is [-8.09782, 2.40511, 48.5, 60] (whole UK)
    new_site : new_site info where one is to be plotted

    Returns:
        map plot with scaled markers showing the potential tidal stream sites in the UK
    """
    path='../data/renewables/Marine/tidal_stream_future_deployment_scenarios.xlsx'
    df=pd.read_excel(path,sheet_name='tidal_stream_high')
  
    df.fillna(0, inplace=True)

    lon = df['Lon'].values
    lat = df['Lat'].values
    sites = df['Site ID'].values
    
    minlon=extent[0]
    maxlon=extent[1]
    minlat=extent[2]
    maxlat=extent[3]
    
    if new_site != None:
        if new_site['lat'] > minlat and new_site['lat'] < maxlat and new_site['lon'] > minlon and new_site['lon'] < maxlon:
            lat=np.append(lat,new_site['lat'])
            lon=np.append(lon,new_site['lon'])
            sites=np.append(sites,new_site['Site Name'])
            
    condition = (lat>minlat) & (lat<maxlat) & (lon>minlon) & (lon<maxlon)
    
    on_map=np.where(condition)
    
    
    off_map=np.zeros(0)
    if zooms!= None:
        for j in range(len(zooms)):
                minlon=zooms[j][0]
                maxlon=zooms[j][1]
                minlat=zooms[j][2]
                maxlat=zooms[j][3]
                #if site within zoom square remove it from the list
                condition = (lat>minlat) & (lat<maxlat) & (lon>minlon) & (lon<maxlon)
                on_zoom= np.where(condition)
                off_map=np.append(off_map,on_zoom)
                plot(network, map_type, on_zoom, new_site, zooms[j], lat[on_zoom], lon[on_zoom], sites[on_zoom], color, marker_scaler, zooms ,j)
                
    off_map = np.array(off_map)

    if len(off_map) > 0:
          
        on_map = [item for item in on_map[0] if item not in off_map]

    
    lat=lat[on_map]
    lon=lon[on_map]
    sites=sites[on_map]

            
    plot(network,map_type, on_map,new_site,extent,lat,lon,sites,color,marker_scaler,zooms)
    

def plot(network, map_type,on_map, new_site, extent, lat, lon, sites, color, marker_scaler, zooms=None, j=-1):
    
    plt.rcParams["figure.figsize"] = (15,10)
    fig = plt.figure()
    
    ax = fig.add_subplot(1, 1, 1,
                     projection=ccrs.PlateCarree())
    
    ax.set_extent(extent)

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    
    annotate=True
    
    if map_type == "installed":
        
        sizes = network.generators.loc[network.generators.type=="Tidal stream","p_nom"].values
        sizes = sizes[on_map]
        title = 'Installed Capacity (MWelec)'
        do_sizes=True
    
        
        
    if map_type == "potential":
        
        
        sizes = network.generators.loc[network.generators.type=="Tidal stream","p_nom_max"].values
        sizes = sizes[on_map]
        title= 'Maximum Capacity (MWelec)'
        do_sizes=True
        
        
    if map_type == "optimal":
            
        sizes = network.generators.loc[network.generators.type=="Tidal stream","p_nom_opt"].values
        sizes = sizes[on_map]
        title= 'Optimal Capacity (MWelec)'
        do_sizes=True
        
 
        
    if map_type == "location":
    
        ax.scatter(lon, lat, s=20, c=color, edgecolors='black')
        ax.set_title('Location of Tidal Sites')
        do_sizes=False
        
    if map_type=='map only':
        do_sizes=False
        annotate=False
    
    if do_sizes==True:
        ax.scatter(lon, lat, s=sizes * marker_scaler, c=color, edgecolors='black')
        ax.set_title(title, fontsize=16)
        l1 = ax.scatter([], [], s=min(sizes) * marker_scaler, edgecolors='black', color=color)
        l2 = ax.scatter([], [], s=mean(sizes) * marker_scaler, edgecolors='black', color=color)
        l3 = ax.scatter([], [], s=max(sizes) * marker_scaler, edgecolors='black', color=color)

        label1 = round(min(sizes), 0)
        label2 = round(mean(sizes), 0)
        label3 = round(max(sizes), 0)
        labels = [label1, label2, label3]

        ax.legend([l1, l2, l3], labels, frameon=True, fontsize=12,
                  loc=1, borderpad=1.5, labelspacing=1.5,
                  scatterpoints=1)
    
    if annotate == True:
        annotations=[]
        for i,txt in enumerate(sites):
            annotations.append(ax.annotate(txt, (lon[i], lat[i]), color=color , fontsize=14))

    # Adjusting text labels to avoid overlaps
    adjust_text(annotations, ax=ax)
 
    colors=['black','red','yellow','blue']
    n=0
    if zooms!= None:
        for zoom in zooms:
            n+=1
            x_coords=[zoom[0],zoom[1],zoom[1],zoom[0],zoom[0]]
            y_coords=[zoom[2],zoom[2],zoom[3],zoom[3],zoom[2]]
            plt.plot(x_coords, y_coords, color=colors[n])
       
    plt.gca().spines['top'].set_color(colors[j+1])
    plt.gca().spines['bottom'].set_color(colors[j+1])
    plt.gca().spines['left'].set_color(colors[j+1])
    plt.gca().spines['right'].set_color(colors[j+1])
    
    plt.savefig('sites'+str(extent)+'.png')

    plt.show()

      
def plot_sitefield(x_coords,y_coords, variable, site_name):
    #plots a colourmap for a field site, overlaid on its location in the UK map
    plt.rcParams["figure.figsize"] = (15,10)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1,
                     projection=ccrs.PlateCarree())

    minlat,minlon=utm_to_latlon(np.min(x_coords),np.min(y_coords))
    maxlat,maxlon=utm_to_latlon(np.max(x_coords),np.max(y_coords))
    
    print('site height' + str(np.max(y_coords)-np.min(y_coords)))
    print('site width' + str(np.max(x_coords)-np.min(x_coords)))
    
    UTM=np.vectorize(utm_to_latlon)
    
    lats,lons = UTM(x_coords,y_coords)
    
    extent = [minlon-0.15, maxlon+0.15, maxlat+0.15, minlat-0.15]
    
    ax.set_extent(extent)

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    
    
   
    scatter_plot = ax.scatter(lons, lats, c=variable, cmap='viridis')
   
    ax.set_title('Power Density for site:' + site_name, fontsize=16)
    
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(scatter_plot, label='Average Capacity Factor', cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=18)
    
    plt.savefig('cf_map.png')

    plt.show()



    
    
def alter_p_nom(network,tech,year,scenario):
    """
    

    Parameters
    ----------
    network : network to be altered
    tech : generator type
    year : year for generator pnom data to be taken from
    scenario : affects capacity of renewables installed

    Returns
    -------
    alters the capacity of different generator types to previous levels

    """

    path='../data/P_nom data/'+str(year)+'/generators'+str(year)+scenario+'.csv'
    
    data=pd.read_csv(path,index_col=0)

    curr_data=network.generators.loc[network.generators.type==tech,'p_nom']
    
    prev_data=data.loc[data.type==tech,'p_nom']
        

    for generator in curr_data.index:
        if generator in prev_data.index:
            network.generators.p_nom[generator]= prev_data[generator]
        else:
            network.generators.p_nom[generator]=0
    return


def get_correlation_table(gens,network):
    #make a table of correlation coefficient for a list of generator types 'gens'
    capacity_factors=network.generators_t['p_max_pu']
    gen_types=[]
    #make a list of every generator of a certain generation type
    for gen in gens:
        gen_types.append(network.generators.loc[network.generators.type==gen].index.values)


    all_gens=np.concatenate(gen_types)
        
    correlation_coefficients=np.zeros((len(all_gens),len(all_gens)))

    for i in range(len(all_gens)):
        gen_column=all_gens[i]
        cf_of_column=capacity_factors[gen_column]
    
        for j in range(i+1,len(all_gens)): #corr coefficient of itself would muddle results, this makes it an upper triangular matrix
            # Calculate Pearson correlation coefficient
            gen_index=all_gens[j]
            cf_of_row=capacity_factors[gen_index]
        
            correlation_coefficient, p_value = pearsonr(cf_of_column, cf_of_row)
            correlation_coefficients[i,j]=correlation_coefficient

    df = pd.DataFrame(correlation_coefficients, index=all_gens, columns=all_gens)
    df.to_csv('kylerhea.csv')
    #averages the correlation coefficients in df to be for each generator
    avg_for_Gen_type=[]
    for Gen_type in gen_types:
        for gen_type in gen_types:
            sum_for_Gen_type=0
            for gen in Gen_type:
            
                sum_cc=np.sum(df[gen_type].loc[gen].values)+np.sum(df[gen][gen_type].values) #summing all correlation coefficients between gen and a technology
                if gen in gen_type:
                    avg_cc=sum_cc/(len(gen_type)-1)
                else:
                    avg_cc=sum_cc/len(gen_type)
                sum_for_Gen_type+=avg_cc
            avg_for_Gen_type.append(sum_for_Gen_type/len(Gen_type))
            
    #puts the data into an len(gens)xlen(gens) matrix so its more readable
    avg_ccs=np.array([avg_for_Gen_type[i*len(gens):(i+1)*len(gens)] for i in range(len(gens))])
    df_ccs = pd.DataFrame(avg_ccs, index=gens, columns=gens)
    
    return df_ccs




