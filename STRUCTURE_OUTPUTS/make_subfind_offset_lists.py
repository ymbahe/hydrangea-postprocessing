"""Create and save file offset lists for the subfind catalogues."""

import hydrangea as hy
import numpy as np
import os
from pdb import set_trace

def main():
    for isim in range(1, 30):
        print("")
        print(f"Processing simulation {isim}")
        print("")
        
        sim = hy.Simulation(index=isim)
        if not os.path.isdir(sim.run_dir):
            print(f'Cannot find simulation {isim}, skipping it.')
            continue

        for isnap in range(30):
            print(f"Processing snapshot {isim}.{isnap}")
            subfind_file = sim.get_subfind_file(isnap)
            if subfind_file is None:
                print(f'Cannot create subfind file for {isim}.{isnap}, '
                      'skipping it.')
                continue

            process_catalogue(subfind_file)

    print("Done!")

    
def process_catalogue(subfind_file):
    """Analyse one individual subfind catalogue."""

    subfind_fof = hy.SplitFile(subfind_file, 'FOF', read_index=0)
    subfind_sh = hy.SplitFile(subfind_file, 'Subhalo', read_index=0)
    subfind_ids = hy.SplitFile(subfind_file, 'IDs', read_index=0)

    subfind_dir = os.path.dirname(subfind_file)
    output_file = subfind_dir + '/FileOffsets.hdf5'
    print(f"Writing to '{output_file}'.") 

    # Need to read in the file offsets now, before the file is written!
    subfind_fof.file_offsets
    subfind_sh.file_offsets
    subfind_ids.file_offsets

    hy.hdf5.write_data(output_file, 'FOF', subfind_fof.file_offsets,
                       comment="First FOF group stored in file i. The last "
                       "entry gives the total number of FOF groups.")
    hy.hdf5.write_data(output_file, 'Subhalo', subfind_sh.file_offsets,
                       comment="First subhalo stored in file i. The last "
                       "entry gives the total number of subhaloes.")
    hy.hdf5.write_data(output_file, 'IDs', subfind_ids.file_offsets,
                       comment="First ID stored in file i. The last "
                       "entry gives the total number of IDs.")


if __name__ == '__main__':
    main()

