// Helper functions specific to the mapmaker program

int load_input_parameters(int argc, 
			  char *argv[], 
			  std::string &sParamFile);

int read_parameter_file(std::string sParamFile, 
			RunParams &runParams);

int find_ph_level(std::vector<double> vPartBounds,     // Bounding box of particles
		  std::vector<double> &vCellMins,      // [O] Cell lower corner
		  std::vector<int> &vNumCells);         // [O] Number of cells per dimension


void update_cell_info(int nPHLevel,
		      std::vector<double> vPartBounds,
		      std::vector<double> &vCellMins,
		      double &dCellVolume,
		      double &dCellSize,
		      std::vector<int> &vNumCells,
		      long &nNumOfCells);


void print_config_flags();
