"""distance calculator
Allocate points to the zone they belong to based on the divided zones
Reference: https://github.com/calliope-project/uk-calliope
"""

import json
import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

def load_zone(json_file):
    zones_list = list()
    for i in range(len(json_file)):
        try:
            zone = Polygon(json_file[i]['geometry']['coordinates'][0])
        except:
            polygon_list=list()
            for j in range(len(json_file[i]['geometry']['coordinates'])):
                polygon = Polygon(json_file[i]['geometry']['coordinates'][j][0])
                polygon_list.append(polygon)
            zone = MultiPolygon(polygon_list)
        zones_list.append(zone)
    return zones_list


def map_to_zone(df, subzone=None, warm=False):
    file_path = '../data/network/ZonesBasedGBsystem/zone/zones_json.geojson'
    json_file = json.loads(open(file_path).read())['features']
    zones_list = load_zone(json_file)[:20]

    if subzone is not None:
        if type(subzone) is not list:
            subzone = [subzone]
        json_file = [json_file[i] for i in range(len(json_file)) if [(json_file[j]['properties']['Name_1'] in subzone) for j in range(len(json_file))][i]]
        zones_list = [zones_list[i] for i in range(len(json_file)) if [(json_file[j]['properties']['Name_1'] in subzone) for j in range(len(json_file))][i]]
    
    object_to_zone = []
    for i in range(len(df)):
        data_point = Point(df['x'][i],df['y'][i])
        n = 0
        # for j in range(len(json_file)[:20]):
        for j in range(20):
            zone = zones_list[j]
            if zone.contains(data_point):
                n += 1
                object_to_zone.append(json_file[j]['properties']['Name_1'])
                
        if (n == 0) & (not np.isnan(df['x'][i])) & (not np.isnan(df['y'][i])):
            if warm:
                print('point {} is not inside any zone, use the nearest zone instead'.format((df['x'][i],df['y'][i])))
            min_poly = min(zones_list, key=data_point.distance)
            index_min_poly = zones_list.index(min_poly)
            object_to_zone.append(json_file[index_min_poly]['properties']['Name_1'])
            
        elif n != 1:
            if warm:
                print('Error while allocated for point {}, set to nan value'.format((df['x'][i],df['y'][i])))
            object_to_zone.append(float('nan'))

    return object_to_zone


def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


if __name__ == "__main__":
    # file_path = '../data/zone/zones_json.geojson'
    # json_file = json.loads(open(file_path).read())
    # zones_list = load_zone(json_file)

    from storage import read_storage_data
    df = read_storage_data(2050)
    df['x'] = df['x']-10
    print(map_to_zone(df))
    print(map_to_zone(df, subzone='Z7'))



