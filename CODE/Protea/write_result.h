void make_master_haloloclist(const std::vector<int> &vTHOlist, 
			     const std::vector<int> &vHaloList, 
			     std::vector<int> &vMasterHaloLocList);

void write_result(const Result &SelSendB,                       // The result to be written
		  const std::vector<int> &vMergeTargets,        // Merge list by subhaloes
		  const std::vector<int> &vTHOlistB,                // THOlistB
		  std::vector<int> &vMasterHaloLocListA,        // [O] MHL list (required outside)
		  int &nLengthA,                    // [O] num of galaxies at A
		  const std::vector<int> &vFlagList);



