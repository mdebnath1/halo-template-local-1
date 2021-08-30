import xarray as xr
import pandas as pd
from tsdat.io import AbstractFileHandler
from tsdat import Config
import numpy as np



class HplHandler(AbstractFileHandler):
    """-------------------------------------------------------------------
    Custom file handler for reading *.sta files.

    See https://tsdat.readthedocs.io/ for more file handler examples.
    -------------------------------------------------------------------"""

    def write(self, ds: xr.Dataset, filename: str, config: Config, **kwargs):
        """-------------------------------------------------------------------
        Classes derived from the FileHandler class can implement this method
        to save to a custom file format.

        Args:
            ds (xr.Dataset): The dataset to save.
            filename (str): An absolute or relative path to the file including
                            filename.
            config (Config, optional):  Optional Config object. Defaults to
                                        None.
        -------------------------------------------------------------------"""
        raise NotImplementedError("Error: this file format should not be used to write to.")

    def read(self, filename: str, **kwargs) -> xr.Dataset:
        """-------------------------------------------------------------------
        Classes derived from the FileHandler class can implement this method.
        to read a custom file format into a xr.Dataset object.

        Args:
            filename (str): The path to the file to read in.

        Returns:
            xr.Dataset: An xr.Dataset object
        -------------------------------------------------------------------"""


        # open file
        with open(filename, 'r') as f:
            lines = []
            for line_num in range(11):
                lines.append(f.readline())
                
            # read metadata into strings
            metadata = {}
            for line in lines:
                metaline = line.split(':')
                if 'Start time' in metaline:
                    metadata['Start time'] = metaline[1:]
                else:
                    metadata[metaline[0]] = metaline[1]
                
            # convert some metadata
            num_gates = int(metadata['Number of gates'])
            
            # Read some of the label lines
            for line_num in range(6):
                f.readline()
                
                
            # while there is a file
            df_new = pd.DataFrame()
            
            max_len = 10000
            # initialize arrays
            time            = np.empty(max_len)
            azimuth         = np.empty(max_len)
            elevation       = np.empty(max_len)
            pitch           = np.empty(max_len)
            roll            = np.empty(max_len)
            doppler         = np.empty((max_len,num_gates))
            intensity       = np.empty((max_len,num_gates))
            beta            = np.empty((max_len,num_gates))
            
            time[:]         = np.nan
            azimuth[:]      = np.nan
            elevation[:]    = np.nan
            pitch[:]        = np.nan
            roll[:]         = np.nan
            doppler[:]      = np.nan
            intensity[:]    = np.nan
            beta[:]         = np.nan
            
            index = 0
            
            while True:
                a = f.readline().split()
                if not len(a): # is empty
                    break

                data_point = {}
                time[index]         = float(a[0])
                azimuth[index]      = float(a[1])
                elevation[index]    = float(a[2])
                pitch[index]        = float(a[3])
                roll[index]         = float(a[4])

                for i in range(num_gates):
                    b = f.readline().split()
                    range_gate                  = int(b[0])
                    doppler[index,range_gate]   = float(b[1])
                    intensity[index,range_gate] = float(b[2])
                    beta[index,range_gate]      = float(b[3])
                    
                # increment index
                index += 1

                # print status
                if True and index % 100 == 0:
                    print(time[index-1])
                    

        # Trim data where first time index is nan
        last_ind = np.where(np.isnan(time))[0][0]

        # Convert to Dataset
        dataset = xr.Dataset()

        index = np.arange(len(time[:last_ind]))

        dataset.coords["range_gate"] = np.arange(num_gates)
        dataset['Decimal time (hours)'] = (("time"), time[:last_ind])
        dataset['Azimuth (degrees)'] = (("time"), azimuth[:last_ind])
        dataset['Elevation (degrees)'] = (("time"), elevation[:last_ind])
        dataset['Pitch (degrees)'] = (("time"), pitch[:last_ind])
        dataset['Roll (degrees)'] = (("time"), roll[:last_ind])

        dataset['Doppler'] = (("time","range_gate"), doppler[:last_ind,:]) 
        dataset['Intensity'] = (("time","range_gate"), intensity[:last_ind,:]) 
        dataset['Beta'] = (("time","range_gate"), intensity[:last_ind,:]) 

        # convert date to np.datetime64
        start_time_string = '{}-{}-{}T{}:{}:{}'.format(
            metadata['Start time'][0][1:5],     # year
            metadata['Start time'][0][5:7],     # month
            metadata['Start time'][0][7:9],     # day
            '00',           # hour
            '00',           # minute
            '00.00'         # second
        )
        start_time  = np.datetime64(start_time_string)      
        dtimes      = np.array((dataset['Decimal time (hours)']))       #relative to start of day (start_time)
        tt          = []
        for i, t in enumerate(dtimes):
            tt.append(start_time + np.timedelta64(int(3600 * 1e6 *t),'us'))

        dataset['Timestamp'] = (("time"), np.array(tt))
        dataset.coords["time"] = np.array(tt)

        # Save some attributes
        dataset.attrs['Range gate length (m)']  = float(metadata['Range gate length (m)'])

        return dataset
