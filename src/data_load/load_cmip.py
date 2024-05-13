import cdsapi
import os
from calendar import monthrange
from pathlib import Path
import xarray
import shutil
import cdsapi


path = "/download"

years =  [
           '1979', '1980', '1981',
           '1982', '1983', '1984',
           '1985', '1986', '1987',
           '1988', '1989', '1990',
           '1991', '1992', '1993',
           '1994', '1995', '1996',
           '1997', '1998', '1999',
           '2000', '2001', '2002',
           '2003', '2004', '2005',
           '2006', '2007', '2008',
           '2009', '2010', '2011',
           '2012', '2013', '2014'
]
 
# Retrieve all months for a given year.
months = ['01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12']
 
# For valid keywords, check the link:
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6?tab=form
 
vars = [
  "near_surface_air_temperature",
  "daily_maximum_near_surface_air_temperature",
  "daily_minimum_near_surface_air_temperature",
  "precipitation",
  "near_surface_wind_speed"
  ]

models = [
  "CMCC-ESM2",
  "ec_earth3_cc"
  ]

# Log in to CDSAPI. For it put .cdsapirc in $ROOT folder
c = cdsapi.Client()

def main():
    for model in models:
        for var in vars:
            data = c.retrieve(
            'projections-cmip6',
            {
                'format': 'zip',
                'temporal_resolution': 'daily',
                'experiment': 'historical',
                'variable': var,
                'month': months,
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '28', '29', '30',
                    '25', '26', '27',
                    '31',
                ],
                'year': years,
                'model': model,
            },
            os.path.join(path, f"{model}_{var}.zip")
            )
            
            # To unzip
            # shutil.unpack_archive(os.path.join(path, f"{model}_{var}.zip"), path)
            # os.remove(os.path.join(path, f"{model}_{var}.zip"))

if __name__ == "__main__":      
    main()
