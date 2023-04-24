int g_numtasks = -1;
int g_rank = -1;
int g_nMultiplex = 1;
int g_function_indent = 3;
int g_nLinkRatio = 5;
int g_nHaloesA, g_nHaloesB, g_nHaloesC;
int g_ttSnap;
int g_nSnapshots;
int g_nSnapA = -1, g_nSnapB = -1, g_nSnapC = -1;

int g_nGalaxies = -1;
int g_CurrSnapInd = -1;

std::string g_rp, g_sFileNameRoot, g_sFileNameOut;
std::vector<double> g_vFunctionStartTimes;
RunParams Params;

std::string g_sSubDirA, g_sSubDirB, g_sSubDirC;
double g_hubble_curr, g_aexp_curr;

const int g_ML_DISRUPTED = -10;
