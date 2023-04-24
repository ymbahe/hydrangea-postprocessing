// Public functions for tracer_read.cpp

void make_tracer_tpolist(std::vector<int> &vGPOlistT, 
			 std::vector<int> &vTPOlistT, 
			 std::vector<int> &vTGOlistT);

void read_tracer_ids(long nStart, 
		     long nEnd, 
		     std::vector<unsigned long> &vTT_IDsT);

template <typename T>
void find_gal_averages(std::string sDataSetName, 
		       int nDim,
		       const std::vector<long> &vTPOlistS,
		       const std::vector<long> &vFPOlistS,
		       const std::vector<int> &vDestTasks,
		       const std::vector<int> &vDestInds,
		       const std::vector<int> &vGPOlist,
		       const std::vector<int> &vTGOlist,
		       int FlagAstroConv);

void create_output_file(const std::vector<Snepshot> &vSnepshotList);

void create_snepshot_datasets(int nLength);

void verify_id_range(std::vector<unsigned long> vIDs);
void verify_id_range(std::vector<long> vIDs);

#include "tracer_read.tpp"
