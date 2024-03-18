import numpy as np
import os


mean_channels_cmip6 = [8.934635,     # sfcWindmax
                       6.9155507,     # sfcWind
                       2.8345312e-05, # pr
                       100936.055,    # psl
                       280.37646,     # tasmax
                       276.80447,]    # tasmin

std_channels_cmip6 = [4.7078495,
                       3.769951,
                       7.100028e-05,
                       1362.2008,
                       20.841803,
                       20.852707]

mean_channels_cmip5 = [9.64286994934082,        # sfcWindmax
                       7.422664165496826,      # sfcWind
                       2.7714191674022004e-05,  # pr
                       100901.3828125,          # psl
                       280.3889465332031,       # tasmax
                       276.36065673828125,]      # tasmin

std_channels_cmip5 =  [5.953777313232422,
                       4.591133117675781,
                       7.818344602128491e-05,
                       1526.6593017578125,
                       21.043394088745117,
                       21.90244483947754]

elevation_mean = 377.73032
elevation_std =  855.89075