#ifndef IO_HDF5_H
#define IO_HDF5_H

long read_hdf5_attribute_long(std::string sFileName,
			      std::string sAttribName,
			      char nAttribType = 'g',
			      int nAttribIndex = 0);

// Vector version of read_hdf5_attribute_long
std::vector<long> read_hdf5_attribute_vlong(std::string sFileName,
					    std::string sAttribName,
					    char nAttribType = 'g');

double read_hdf5_attribute_double(std::string sFileName,
				  std::string sAttribName,
				  char nAttribType = 'g',
				  int nAttribIndex = 0);

std::string read_hdf5_attribute_string(std::string sFileName,
				       std::string sAttribName,
				       char nAttribType,
				       int nAttribIndex);


void read_hdf5_data(std::string sFileName,             // HDF5 file name
		    std::string sDataSetName,          // Data set to read
		    std::vector<long> &vOut,           // [O] data
		    long pos_start = 0,                // first entry to read...
		    long pos_length = -1,              // ... and how many in total
		    int nDim = 0,                      // Dimension to be read
		    int FlagResizeVec = 1,            // 0: don't modify vOut 
		    long nOutputOffset = 0);

void read_hdf5_data(std::string sFileName,             // HDF5 file name
		    std::string sDataSetName,          // Data set to read
		    std::vector<unsigned long> &vOut,           // [O] data
		    long pos_start = 0,                // first entry to read...
		    long pos_length = -1,              // ... and how many in total
		    int nDim = 0,                      // Dimension to be read
		    int FlagResizeVec = 1,            // 0: don't modify vOut 
		    long nOutputOffset = 0);


void read_hdf5_data(std::string sFileName,             // HDF5 file name
		    std::string sDataSetName,          // Data set to read
		    std::vector<int> &vOut,            // [O] data
		    long pos_start = 0,                // first entry to read...
		    long pos_length = -1,              // ... and how many in total
		    int nDim = 0,
		    int FlagResizeVec = 1,             // Dimension to be read
		    long nOutputOffset = 0);

void read_hdf5_data(std::string sFileName,             // HDF5 file name
		    std::string sDataSetName,          // Data set to read
		    std::vector<unsigned int> &vOut,            // [O] data
		    long pos_start = 0,                // first entry to read...
		    long pos_length = -1,              // ... and how many in total
		    int nDim = 0,
		    int FlagResizeVec = 1,             // Dimension to be read
		    long nOutputOffset = 0);

void read_hdf5_data(std::string sFileName,              // HDF5 file name
		    std::string sDataSetName,           // Data set to read
		    std::vector<float> &vOut,           // [O] data
		    long pos_start = 0,                 // first entry to read...
		    long pos_length = -1,               // ... and how many in total
		    int nDim = 0,                       // Dimension to be read
		    int FlagResizeVec = 1,              
		    long nOutputOffset = 0);
		    
void read_hdf5_data(std::string sFileName,             // HDF5 file name
		    std::string sDataSetName,          // Data set to read
		    std::vector<double> &vOut,         // [O] data
		    long pos_start = 0,                // first entry to read...
		    long pos_length = -1,              // ... and how many in total
		    int nDim = 0,
		    int FlagResizeVec = 1,            // Dimension to be read
		    long nOutputOffset = 0);

void write_hdf5_data(std::string sFileName,             // File name to write to
		     std::string sDataSetName,          // Dataset to write to
		     const std::vector<int> &vData,    // Data to write
		     long pos_start = 0,                // Start position
		     int nDim = 0,                     // Dimension to write (for 2D only)
		     int nVerb = 0);

void write_hdf5_data(std::string sFileName,             // File name to write to
		     std::string sDataSetName,          // Dataset to write to
		     const std::vector<long> &vData,    // Data to write
		     long pos_start = 0,                // Start position
		     int nDim = 0,                     // Dimension to write (for 2D only)
		     int nVerb = 0);

void write_hdf5_data(std::string sFileName,             // File name to write to
		     std::string sDataSetName,          // Dataset to write to
		     const std::vector<unsigned long> &vData,    // Data to write
		     long pos_start = 0,                // Start position
		     int nDim = 0,                     // Dimension to write (for 2D only)
		     int nVerb = 0);

void write_hdf5_data(std::string sFileName,             // File name to write to
		     std::string sDataSetName,          // Dataset to write to
		     const std::vector<unsigned int> &vData,    // Data to write
		     long pos_start = 0,                // Start position
		     int nDim = 0,                     // Dimension to write (for 2D only)
		     int nVerb = 0);

void write_hdf5_data(std::string sFileName,             // File name to write to
		     std::string sDataSetName,          // Dataset to write to
		     const std::vector<char> &vData,    // Data to write
		     long pos_start = 0,                // Start position
		     int nDim = 0,                     // Dimension to write (for 2D only)
		     int nVerb = 0);

void write_hdf5_data(std::string sFileName,             // File name to write to
		     std::string sDataSetName,          // Dataset to write to
		     const std::vector<float> &vData,    // Data to write
		     long pos_start = 0,                // Start position
		     int nDim = 0,                     // Dimension to write (for 2D only)
		     int nVerb = 0);

void write_hdf5_data(std::string sFileName,             // File name to write to
		     std::string sDataSetName,          // Dataset to write to
		     const std::vector<double> &vData,    // Data to write
		     long pos_start = 0,                // Start position
		     int nDim = 0,                     // Dimension to write (for 2D only)
		     int nVerb = 0);


std::vector<long> hdf5_measure_dataset(std::string sFileName, 
				       std::string sDataSetName);


void hdf5_create_file(std::string sFileName);

// Function to create a group in an existing HDF5 file  
void hdf5_create_group(std::string sFileName,
		       std::string sGroupName);

// Function to create a dataset in an *existing* HDF5 file
void hdf5_create_dataset(std::string sFileName,
			 std::string sDataSetName,
			 std::vector<int> vDims,
			 std::string sDataSetType = "float",
			 std::string sDescription = "",
			 int nFillValue = -1);

void hdf5_write_attribute(std::string sFileName,
			  std::string sContName,
			  std::string sAttribName,
			  double dAttribVal,
			  char nAttribType = 'd');

void hdf5_write_attribute(std::string sFileName,
			  std::string sContName,
			  std::string sAttribName,
			  long lAttribVal,
			  char nAttribType = 'd');

void hdf5_write_attribute(std::string sFileName,
			  std::string sContName,
			  std::string sAttribName,
			  int lAttribVal,
			  char nAttribType = 'd');

// STRING version
void hdf5_write_attribute(std::string sFileName,
			  std::string sContName,
			  std::string sAttribName,
			  std::string sAttribVal,
			  char cAttribType);


// Array-LONG version
void hdf5_write_attribute_array(std::string sFileName,
				std::string sContName,
				std::string sAttribName,
				std::vector<long> vAttribVals,
				char nAttribType = 'd');

// Array-INT version
void hdf5_write_attribute_array(std::string sFileName,
				std::string sContName,
				std::string sAttribName,
				std::vector<int> vAttribVals,
				char nAttribType = 'd');

// Array-DOUBLE version
void hdf5_write_attribute_array(std::string sFileName,
				std::string sContName,
				std::string sAttribName,
				std::vector<double> lAttribVals,
				char nAttribType = 'd');



#endif
