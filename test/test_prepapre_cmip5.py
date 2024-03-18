import random, string
import sys, os
import unittest
import xarray as xr
sys.path.append(os.getcwd())

from src.data_assemble.prepare_cmip5 import CMIP5File
class TestNaming(unittest.TestCase):
    def test_naming(self):
        print("Naming")
        N = random.randint(4, 10)
        N_path_members = random.randint(4, 10)
        random_input =  '/' + os.path.join(*[''.join(random.choices(string.ascii_uppercase + string.digits, k=N)) for i in range(N_path_members)])
        random_var_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(1, 10))) + '_day_'
        random_interim = ''
        for i in range(random.randint(1, random.randint(7, 15))):
            random_interim += (''.join(random.choices(string.ascii_uppercase + string.digits, k=N)) + '_') 
        random_dates = [str(random.randint(1850, 2020)) + '0101', -1]
        random_dates[1] = str(random.randint(int(random_dates[0][:-4]), 2022)) + '1231'
        random_dates = random_dates[0] + '-' + random_dates[1]
        random_filename = random_var_name + random_interim + random_dates + '.nc'
        random_path = os.path.join(random_input, random_filename)
        
        file = CMIP5File(path=random_path)
        file.filename# = os.path.basename(path)
        file.path# = path
        file.variable_name# = file.filename.split('_')[0]
        file.start_date# = file.temporal_subset.split('-')[0]
        file.end_date# = file.temporal_subset.split('-')[1].split('.')[0]
        file.start_year# = int(file.start_date[:4])
        file.end_year# = int(file.end_date[:4])
        self.assertTrue((file.variable_name == random_var_name.split('_')[0]), "Variable name parsed incorrectly")
        self.assertTrue((file.start_date == random_dates.split('-')[0]) and (file.end_date == random_dates.split('-')[1]), "dates are read incorrectly")
        self.assertTrue((file.start_year == int(random_dates.split('-')[0][:4])) and (file.end_year == int(random_dates.split('-')[1][:4])), "years are read incorrectly")
    

if __name__ == "__main__":
    unittest.main()