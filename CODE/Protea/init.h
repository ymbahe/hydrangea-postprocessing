// Declarations et al.

#ifndef ENUMS_STRUCTS_ARE_DECLARED
#define ENUMS_STRUCTS_ARE_DECLARED

#include "/u/ybahe/cpplib/init_hydrangea.h"

#include <vector>

enum Codes
  {
    MERGED = -5,
    BYPASSED = -9,
    BYPASSED2 = -8,
    TRACEFAIL = -10,
    IDENTFAIL = -20,
    ML_DISRUPTED = -10,
    
  };
  
struct Link
{
  std::vector<int> Sender;
  std::vector<int> Receiver;
  std::vector<int> Rank;
  std::vector<int> Choice;
  std::vector<float> SenderFraction;
  std::vector<float> ReceiverFraction;
  std::vector<int> NumPart;
  std::vector<int> SortedByRecv;
  std::vector<int> SenderOffset;
  std::vector<int> ReceiverOffset;
};
 

struct Result
{
  std::vector<int> Match;
  std::vector<int> Length;
  std::vector<int> Rank;
  std::vector<int> Choice;
  std::vector<long> LinkInd;
  std::vector<float> SenderFraction;
  std::vector<float> ReceiverFraction;
};
  

struct RunParams
{
  int nSnapIni;
  int nSnapFin;

  char cCodaOut[1000];
  char cSimBaseDir[1000];
  char cOutputDir[1000];

  int  nSnapshotListEntries;
  char cSnapshotListName[1000];
  char cRunType[20];

  int nSnepshotListEntries;
  char cSnepshotListName[1000];

  double dMinFracLowChoice;
  double dMinRecvFrac;
  double dMinSendFrac;

  int nMaxTracers;
  
};



#endif
