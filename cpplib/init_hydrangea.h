// Definitions of structs etc. generic to Hydrangea/Eagle use

#ifndef INIT_HYDRANGEA_H
#define INIT_HYDRANGEA_H

struct Snepshot
{
  int Index;
  int RootIndex;
  int Num;
  int Type;
};

enum SnepType
  {
    SNEPSHOT_SNAP = 0,
    SNEPSHOT_SNIP = 1
  };

enum OutputType
  {
    EAGLE_SNAPSHOT = 0,
    EAGLE_SUBDIR = 1,
    EAGLE_SUBPART = 2
  };


#endif
