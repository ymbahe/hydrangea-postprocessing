// These switches can modify the behaviour of the code!

//#define VERBOSE
//#define DEBUG
//#define DEBUG_SELECT_LINKS
//#define DEBUG_IDLHIST
//#define VERBOSE_READ_HDF5_DATA_LONG
//#define DEBUG_HDF5

#define VERBOSE_SNIPSHOTS

//#define DENSE_IDS

#define BARRIER_AT_KATAMARAN_LOOPEND
#define STRICT_LINK_LIMIT 20

// --------------------------
// Options for link selection
// --------------------------

#define EXHAUST_LINKS 10
#define LINK_MIN_RECVFRAC      // enable minimum ReceiverFraction for link to be selected.
#define LINK_MAX_CHOICE 10     // Maximum choice level considered
// #define LINK_MAX_RANK 10    // Maximum rank level (not currently set)

#define USE_LONGLINKS    // not actually doing anything
#define USE_LOWER_RANKS  // not actually doing anything


#define EXTRACT_IDS_FOR_SNIPSHOTS
#define EXTRACT_IDS_FOR_SNIPSHOTS_REVERSE
#define NUM_SNIPIDS 20

//#define FIXED_ID_RANGE   3e9   // Set this to assume a fixed number of particles (USE WITH CAUTION!!!!!)
