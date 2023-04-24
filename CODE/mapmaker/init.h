// init.h
// File that contains definitions and declarations for SNIPLOCATE program
// Started 20 NOV 2015

#ifndef ENUMS_STRUCTS_ARE_DECLARED
#define ENUMS_STRUCTS_ARE_DECLARED

// Put numerical codes that are NOT user-definable here
// Everything adjustable should instead be #defined in Config.h

enum Codes
  {
    REPLACEME = 100,
    REPLACEMETOO = 1000
  };

enum Constants
  {
    NUM_PARTTYPES = 6
  }; 

// This is a structure containing variables read in from the configuration file
// NB: It must not contain vectors to enable easy broadcasting to all tasks

struct RunParams
{
  char cSimBaseDir[1000];        // The base directory of the simulation

  int  nSnepshotListEntries;     // How many entries expected in sneplist
  char cSnepshotListName[1000];  // The name of an (ASCII) file containing the snepshot list in 2-column format
  int nSnepIni;                  
  int nSnepFin;
  int nSnepshotListLoc;          // [0] if in SimBaseDir, [1] if in output dir

  char cOutputFileName[1000];       // The (full) file name to which write (all) the results
  char cOutputDir[1000];         // The base directory to write output to (can be different from Sim Base)

  int nTypeCode;                 // Code for all particles to be processed (Sum(2^i))
  int nDesNumPerCell;            // Ideal average number of particle per cell
  double dBoxSize;               // Size of (full) simulation box, in code units
  double dMaxCellSize;           // Maximum cell sidelength (in code units)
  int nMinNumOfCells;            // MINIMUM number of cells in map
  int nMaxNumOfCells;            // MAXIMUM number of cells in map
  int nMaxPHLevel;               // Maximum allowed PH level

  char cSimLabel[1000];          // Label for simulation (written into output)
  int nFlagCreateSnepList;       // Flag to automatically create sneplist when it does not exist yet.
  int nRedo;                     // Flag to ignore pre-existing maps.

};



/*
struct Snepshot
{
  int Index;
  int Num;
  int Type;
};
*/

#endif
