"""Script for running a network constrained linear optimal power flow of PyPSA-GB
"""

import pypsa
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs

import data_reader_writer

import time

# start = '2050-01-01 00:00:00'
# end = '2050-12-31 23:30:00'
# year = int(start[0:4])

# for scenario in ['Leading The Way', 'Consumer Transformation', 'System Transformation', 'Falling Short']:
#     for demand_dataset in ['eload', 'historical']:
#         for time_step in [1.0]:
#             for year_baseline in [2012, 2013]:
#                 for networkmodel in ['Reduced', 'Zonal']:
#                     for P2G in [True, False]:
#                         print('inputs:', year, scenario, demand_dataset, time_step, year_baseline, networkmodel, P2G)

#                         start_t = time.time()
#                         data_reader_writer.data_writer(start, end, time_step, year, demand_dataset=demand_dataset, 
#                                                     year_baseline=year_baseline, scenario=scenario, FES=2022, 
#                                                     merge_generators=True, scale_to_peak=True, 
#                                                     networkmodel=networkmodel, P2G=P2G,
#                                                     marine_modify=True, marine_scenario='Mid')

#                         network = pypsa.Network()
#                         network.import_from_csv_folder('LOPF_data')

#                         setup_t = time.time()
#                         print((setup_t - start_t) / 60, 'minutes taken to write/read csv files and create network object.')

#                         # print(network.links)

#                         if networkmodel == 'Reduced':
#                             contingency_factor = 4
#                             network.lines.s_max_pu *= contingency_factor
#                         elif networkmodel == 'Zonal':
#                             contingency_factor = 4
#                             network.links.p_nom[:20] *= contingency_factor

#                         network.lopf(network.snapshots, solver_name="gurobi", pyomo=False)

#                         opt_t = time.time()
#                         print((opt_t - setup_t) / 60, 'minutes taken to perform optimisation.')

#                         # print(network.buses_t.marginal_price)
#                         # print(network.generators_t.status)
#                         print(network.generators_t.p)

start = '2050-02-28 00:00:00'
end = '2050-03-01 23:30:00'
year = int(start[0:4])

for scenario in ['System Transformation', 'Falling Short']:
    for demand_dataset in ['eload']:
        for time_step in [1.0]:
            for year_baseline in [2012, 2013]:
                for networkmodel in ['Zonal']:
                    for P2G in [True]:
                        if year % 4 == 0 and year_baseline % 4 != 0:
                            break
                        print('inputs:', year, scenario, demand_dataset, time_step, year_baseline, networkmodel, P2G)

                        start_t = time.time()
                        data_reader_writer.data_writer(start, end, time_step, year, demand_dataset=demand_dataset, 
                                                       year_baseline=year_baseline, scenario=scenario, FES=2022, 
                                                       merge_generators=True, scale_to_peak=True, 
                                                       networkmodel=networkmodel, P2G=P2G,
                                                       floating_wind_scenario='Mid', wave_scenario='Mid', tidal_stream_scenario='Mid')

                        network = pypsa.Network()
                        network.import_from_csv_folder('LOPF_data')

                        setup_t = time.time()
                        print((setup_t - start_t) / 60, 'minutes taken to write/read csv files and create network object.')

                        # print(network.links)

                        if networkmodel == 'Reduced':
                            contingency_factor = 4
                            network.lines.s_max_pu *= contingency_factor
                        elif networkmodel == 'Zonal':
                            contingency_factor = 4
                            network.links[15:111].p_nom *= contingency_factor
                        
                        network.consistency_check()

                        network.lopf(network.snapshots, solver_name="gurobi", pyomo=False)

                        opt_t = time.time()
                        print((opt_t - setup_t) / 60, 'minutes taken to perform optimisation.')

                        # print(network.buses_t.marginal_price)
                        # print(network.generators_t.status)
                        print(network.generators_t.p)

# p_by_carrier = network.generators_t.p.groupby(
#     network.generators.carrier, axis=1).sum()

# storage_by_carrier = network.storage_units_t.p.groupby(
#     network.storage_units.carrier, axis=1).sum()
# # print(network.storage_units_t.p)

# # to show on graph set the negative storage values to zero
# storage_by_carrier[storage_by_carrier < 0] = 0
# p_by_carrier = pd.concat([p_by_carrier, storage_by_carrier], axis=1)

# if year <= 2020:

#     # interconnector exports
#     exports = network.loads_t.p
#     # multiply by negative one to convert it as a generator
#     # i.e. export is a positive load, but negative generator
#     exports['Interconnectors Export'] = exports.iloc[:, -6:].sum(axis=1) * -1
#     interconnector_export = exports[['Interconnectors Export']]

# elif year > 2020:
#     # print(network.links_t.p0)
#     # print(network.links_t.p1)
#     imp = network.links_t.p0.copy()
#     imp[imp < 0] = 0
#     imp['Interconnectors Import'] = imp.sum(axis=1)
#     interconnector_import = imp[['Interconnectors Import']]
#     # print(interconnector_import)
#     p_by_carrier = pd.concat([p_by_carrier, interconnector_import], axis=1)

#     exp = network.links_t.p0.copy()
#     exp[exp > 0] = 0
#     exp['Interconnectors Export'] = exp.sum(axis=1)
#     interconnector_export = exp[['Interconnectors Export']]
#     # print(interconnector_export)

# # group biomass stuff
# p_by_carrier['Biomass'] = (
#     p_by_carrier['Biomass (dedicated)'] + p_by_carrier['Biomass (co-firing)'])

# # rename the hydro and interconnector import
# p_by_carrier = p_by_carrier.rename(
#     columns={'Large Hydro': 'Hydro'})
# p_by_carrier = p_by_carrier.rename(
#     columns={'Interconnector': 'Interconnectors Import'})

# # cols = ["Nuclear", "Coal", "Diesel/Gas oil", "Diesel/gas Diesel/Gas oil",
# #         "Natural Gas", "Sour gas",
# #         'Shoreline Wave', 'Tidal Barrage and Tidal Stream',
# #         'Biomass (dedicated)', 'Biomass (co-firing)',
# #         'Landfill Gas', 'Anaerobic Digestion', 'EfW Incineration', 'Sewage Sludge Digestion',
# #         'Large Hydro', 'Pumped Storage Hydroelectricity', 'Small Hydro',
# #         "Wind Offshore"
# #         ]
# #

# if year > 2020:

#     cols = ["Nuclear", 'Biomass',
#             'Waste', "Oil", "Natural Gas",
#             'Hydrogen', 'CCS Gas', 'CCS Biomass',
#             "Pumped Storage Hydroelectric", 'Hydro',
#             'Battery', 'Compressed Air', 'Liquid Air',
#             "Wind Offshore", 'Wind Onshore', 'Solar Photovoltaics',
#             'Interconnectors Import', 'Unmet Load'
#             ]

# else:
#     cols = ["Nuclear", 'Shoreline Wave', 'Biomass',
#             'EfW Incineration',
#             "Coal", "Oil", "Natural Gas",
#             "Pumped Storage Hydroelectric", 'Hydro',
#             "Wind Offshore", 'Wind Onshore', 'Solar Photovoltaics',
#             'Interconnectors Import'
#             ]

# p_by_carrier = p_by_carrier[cols]

# p_by_carrier.drop(
#     (p_by_carrier.max()[p_by_carrier.max() < 35.0]).index,
#     axis=1, inplace=True)


# colors = {"Coal": "grey",
#           "Diesel/Gas oil": "black",
#           "Diesel/gas Diesel/Gas oil": "black",
#           'Oil': 'black',
#           'Unmet Load': 'black',
#           'Anaerobic Digestion': 'green',
#           'Waste': 'chocolate',
#           'Sewage Sludge Digestion': 'green',
#           'Landfill Gas': 'green',
#           'Biomass (dedicated)': 'green',
#           'Biomass (co-firing)': 'green',
#           'Biomass': 'green',
#           'CCS Biomass': 'darkgreen',
#           'Interconnectors Import': 'pink',
#           "Sour gas": "lightcoral",
#           "Natural Gas": "lightcoral",
#           'CCS Gas': "lightcoral",
#           'Hydrogen': "lightcoral",
#           "Nuclear": "orange",
#           'Shoreline Wave': 'aqua',
#           'Tidal Barrage and Tidal Stream': 'aqua',
#           'Hydro': "turquoise",
#           "Large Hydro": "turquoise",
#           "Small Hydro": "turquoise",
#           "Pumped Storage Hydroelectric": "darkturquoise",
#           'Battery': 'lime',
#           'Compressed Air': 'greenyellow',
#           'Liquid Air': 'lawngreen',
#           "Wind Offshore": "lightskyblue",
#           'Wind Onshore': 'deepskyblue',
#           'Solar Photovoltaics': 'yellow'}

# fig, ax = plt.subplots(1, 1)
# # fig.set_size_inches(12, 6)
# (p_by_carrier / 1e3).plot(
#     kind="area", ax=ax, linewidth=0,
#     color=[colors[col] for col in p_by_carrier.columns])

# # stacked area plot of negative values, prepend column names with '_' such that they don't appear in the legend
# (interconnector_export / 1e3).plot.area(ax=ax, stacked=True, linewidth=0.)
# # rescale the y axis
# ax.set_ylim([(interconnector_export / 1e3).sum(axis=1).min(), (p_by_carrier / 1e3).sum(axis=1).max()])

# # Shrink current axis's height by 10% on the bottom
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])

# # Put a legend below current axis
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=5)

# ax.set_ylabel("GW")

# ax.set_xlabel("")

# plt.show()

# fig, ax = plt.subplots(1, 1)
# fig.set_size_inches(12, 6)

# p_storage = network.storage_units_t.p.sum(axis=1)
# state_of_charge = network.storage_units_t.state_of_charge.sum(axis=1)
# p_storage.plot(label="Pumped hydro dispatch", ax=ax, linewidth=3)
# state_of_charge.plot(label="State of charge", ax=ax, linewidth=3)

# ax.legend()
# ax.grid()
# ax.set_ylabel("MWh")
# ax.set_xlabel("")
# plt.show()

# now = network.snapshots[2]

# print("With the linear load flow, there is the following per unit loading:")
# loading = network.lines_t.p0.loc[now] / network.lines.s_nom
# print(loading.describe())

# fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
# fig.set_size_inches(6, 6)

# network.plot(ax=ax, line_colors=abs(loading), line_cmap=plt.cm.jet, title="Line loading")
# plt.show()

# fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
# fig.set_size_inches(6, 4)

# network.plot(ax=ax, line_widths=pd.Series(0.5, network.lines.index))
# plt.hexbin(network.buses.x, network.buses.y,
#            gridsize=20,
#            C=network.buses_t.marginal_price.loc[now],
#            cmap=plt.cm.jet)

# # for some reason the colorbar only works with graphs plt.plot
# # and must be attached plt.colorbar

# cb = plt.colorbar()
# cb.set_label('Locational Marginal Price (EUR/MWh)')
# plt.show()

# carrier = "Wind Onshore"

# capacity = network.generators.groupby("carrier").sum().at[carrier, "p_nom"]
# p_available = network.generators_t.p_max_pu.multiply(network.generators["p_nom"])
# p_available_by_carrier = p_available.groupby(network.generators.carrier, axis=1).sum()
# p_curtailed_by_carrier = p_available_by_carrier - p_by_carrier
# p_df = pd.DataFrame({carrier + " available": p_available_by_carrier[carrier],
#                      carrier + " dispatched": p_by_carrier[carrier],
#                      carrier + " curtailed": p_curtailed_by_carrier[carrier]})

# p_df[carrier + " capacity"] = capacity
# p_df["Wind Onshore curtailed"][p_df["Wind Onshore curtailed"] < 0.] = 0.
# fig, ax = plt.subplots(1, 1)
# fig.set_size_inches(12, 6)
# p_df[[carrier + " dispatched", carrier + " curtailed"]].plot(kind="area", ax=ax, linewidth=0)
# p_df[[carrier + " available", carrier + " capacity"]].plot(ax=ax, linewidth=0)

# ax.set_xlabel("")
# ax.set_ylabel("Power [MW]")
# # ax.set_ylim([0, 10000])
# ax.legend()
# plt.show()

# end_t = time.time()
# print(end_t - start_t, 'total time taken in seconds.')
# print((end_t - start_t) / 60, 'total time taken in minutes.')
