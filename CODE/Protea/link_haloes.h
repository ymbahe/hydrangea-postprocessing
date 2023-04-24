void build_taskinfo(const std::vector<long long> &vFullPartOffsetA, 
		    const std::vector<long long> &vFullPartLengthA, 
		    const std::vector<long> &vTHOlistA, 
		    const std::vector<long long> &vTPOlistA,
		    long nCurrFirstHalo, 
		    const std::vector<long> &vTraceListInternal, 

		    std::vector<long long> &vTraceOffset, 
		    std::vector<long long> &vTraceLength);


void link_haloes(const std::vector<long long> &vTraceOffset, 
		 const std::vector<long long> &vTraceLength, 
		 const std::vector<long> &vTaskMatchHaloes, 

		 std::vector<Result> &vResult);
		 
void select_tracinghaloes(const std::vector<long> &vTaskHaloLocList, 
			  const std::vector<long> &vTHOlistA, 

			  std::vector<long> &vTraceListInternal, 
			  std::vector<long> &vTraceLocList);


std::vector<Result> tracing_result(const std::vector<Result> &vTaskResult, 
				   std::vector<long> &vTraceLocList, 
				   std::vector<long> &vHaloList);
  
