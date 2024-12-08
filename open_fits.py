import pandas
from astropy.io import fits

filename = 'DR16Q_v4.fits'
with fits.open(filename) as hdul:
    # Accesses the first extension
    hdu = hdul[1] 
    data_dict = {}

    # Loops over each field in the binary table
    for name in hdu.columns.names:
        column_data = hdu.data[name]
        
        # Tests to see of the column is multi-dimensional
        # If true, flatten the column
        if column_data.ndim > 1:
            # Flattens each row's data and stores in the dictionary
            for i in range(column_data.shape[1]):
                data_dict[f"{name}_{i}"] = column_data[:, i]
        else:
            # If data is not multi-dimensional, stores it as normal
            data_dict[name] = column_data

    df = pandas.DataFrame(data_dict)

    output_filename = 'output_table.csv'
    df.to_csv(output_filename, index=False)


