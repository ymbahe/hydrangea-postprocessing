void extract_ids_for_snipshots(const Result &TracingResult,         // Result structure 
			       const std::vector<int> &vTHOlist,    // vTHOlistA
			       const std::vector<unsigned long> &vIDs,       // vTaskIDsA
			       const std::vector<long> &vPartOffset,  // Full part offset (A)
			       const std::vector<int> &vMatchHaloB,  // Match halo in B
			       const std::vector<int> &vMatchHaloC,  // Match halo in C
			       const std::vector<long> &vOrderB,  // order in B
			       const std::vector<long> &vOrderC,  // order in C
			       const std::vector<int> &vMHLListA,    // SH->Ind
			       int nGalaxiesTot,                // Number of galaxies (at A)
			       int nFlagReverse=0               // [1 = reverse tracing]
			       );    
