void identify_mergers(const Link &LinksAB, 
		      const Result &SelSendB, 
		      std::vector<int> &vMergeTargets);


void output_links(const Link &lLink,
		  std::string sFileName);


void build_links(const std::vector<int> &vTaskMatchHaloesAB,
		 const std::vector<long> &vFullPartOffsetA, 
		 const std::vector<int> &vTHOlistA,
		 std::vector<int> &vLinkSenderAB, 
		 std::vector<int> &vLinkReceiverAB, 
		 std::vector<int> &vLinkRankAB, 
		 std::vector<int> &vLinkNumPartAB,
		 std::vector<float> &vLinkSenderFractionAB,
		 int nCheck = -1);

void invert_links(Link &Link,
		  const std::vector<long> &vFullPartOffsetList);


void evaluate_link_network(const Link &LinksAB, 
			   const Link &LinksBC, 
			   const Link &LinksAC, 
			   Result &SelSendB,                            // [0] Main result! 
			   std::vector<long> vFullPartOffsetList,
			   Result &SelRecOut);



void select_links(const Link &Links,
		  Result &ss,
		  Result &sr,
		  int length);


Link build_link_struct(std::vector<int> &vLinkSender,
		       std::vector<int> &vLinkReceiver,
		       std::vector<int> &vLinkRank,
		       std::vector<int> &vLinkNumPart,
		       std::vector<float> &vLinkSenderFraction,
		       int nHaloes,
		       int nVerb = 1);

std::vector<int> flag_contaminated_subhaloes
(const std::vector<int> &vHaloesB,   // B-halo for A-particles
 const std::vector<long> &vOrderB,    // B-order for A-particles
 const std::vector<int> &vTHOlist,  // Tasks-->Haloes(A)
 const std::vector<long> &vFullPartOffset, // Halo-->Particles(A)
 const Result &TracingResult,  // Linking result
 const Link &LinksAB);
