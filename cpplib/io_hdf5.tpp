template <typename T>
void hdf5_write_attribute(std::string sFileName,
			  std::string sContName,
			  std::string sAttribName,
			  T AttribVal,
			  char nAttribType) {

  using namespace H5;

  // Open the file
  H5File file(sFileName, H5F_ACC_RDWR);  

  // Create the data space for the attribute.
  hsize_t dims[1] = {1};
  DataSpace attr_dataspace = DataSpace (1, dims);
  
  H5::Attribute attribute;

  if (nAttribType == 'd') {
    DataSet dataset = file.openDataSet(sContName);
    attribute = dataset.createAttribute(sAttribName, PredType::NATIVE_DOUBLE, attr_dataspace);
  }
  else {
    Group group = file.openGroup(sContName);
    attribute = group.createAttribute(sAttribName, PredType::NATIVE_DOUBLE, attr_dataspace);
  }

  // Write the attribute data 
  attribute.write( PredType::NATIVE_DOUBLE, &dAttribVal);

  attribute.close();
  file.close();
  return;
}
