"""
Pathfinder - identifies galaxies in SpiderWeb link output

This is essentially a modified version of the second part of Protea.
At basic level, it goes through snapshot intervals one-by-one, evaluates
the Level1 link network, and builds up the galaxy lifelines.

Started 22-Nov-2017
"""

import yb_utils as yb
import numpy as np
from pdb import set_trace
import sim_tools as st
import time
import os
import ctypes as c
from copy import copy
import argparse
import calendar
import hydrangea_tools as ht

prog_stime = time.time()
parser = argparse.ArgumentParser()

parser.add_argument("dir", help = "The directory in which the simulation is sitting")
parser.add_argument("-n", "--num_snaps", help = "Number of snapshots to process", type = int)
parser.add_argument("-s", "--spiderloc", help = "SpiderWeb output file")
parser.add_argument("-o", "--outloc", help = "Output file name")
parser.add_argument("-p", "--selection_priority", help = "'rank' (default) or 'choice' priority in selecting links")
parser.add_argument("--sim_frac", help = "Similarity fraction for link rejection (default = 2/3)", type = float)
parser.add_argument("--stolen_threshold", help = "Stolen fraction threshold for subhalo to be skipped in tracer IDs (default = 1/3)", type = float)

parser.add_argument("-fd", "--default_flags", help = "Enable all standard code options", action = "store_true")
parser.add_argument("-fll", "--long_links", help = "Use long links", action = "store_true")
parser.add_argument("-flr", "--lower_ranks", help = "Use lower ranks", action = "store_true")
parser.add_argument("-flc", "--lower_choice", help = "Usel lower choice", action = "store_true")
parser.add_argument("-fli", "--limit_links", help = "Exclude weak or suspicious links", action = "store_true")
parser.add_argument("-fch", "--compensate_history", help = "Compensate previous mass exchanges", action = "store_true")
parser.add_argument("-fxh", "--exact_history", help = "Exact history compensation (VERY SLOW, EXPERIMENTAL)", action = "store_true")
parser.add_argument("-fxl", "--exact_limited", help = "(sub-option for exact history mode)", action = "store_true")
parser.add_argument("-fo",  "--allow_orphans", help = "Allow creation of orphan galaxies", action = "store_true")
parser.add_argument("-fx",  "--extract_tracers", help = "Extract tracer particles", action = "store_true")
parser.add_argument("-fns", "--no_spectre_descendants", help = "Disable long-links from spectres", action = "store_true")
parser.add_argument("-ftp", "--trace_prebirth", help = "Extract tracers before first alive snapshot", action = "store_true")
parser.add_argument("-fss", "--skip_stolen", help = "Skip snapshots where a subhalo is largely stolen in tracer list", action = "store_true")
parser.add_argument("-fpr", "--prevent_revenge", help = "Disable long-links to merger target", action = "store_true")
parser.add_argument("-fcs", "--consistent_selection", help = "Use long-links to improve time-consistency in link selection", action = "store_true")
parser.add_argument("-fdh", "--dual_history", help = "Split history compensation into old and new parts", action = "store_true")
parser.add_argument("-fml", "--allow_multiple_longlinks", help = "Allow multiple long-links with similar core particle number", action = "store_true")
parser.add_argument("-fxa", "--exclude_all_linktargs", help = "Disable long-links to descendants of any galaxy that received short-links from the sender (not just merger target)", action = "store_true")
parser.add_argument("-frc", "--require_core", help = "Only allow tracing along links containing at least 3 core particles, or 2/3 of all particles")
parser.add_argument("-fmx", "--mix_history", help = "Only compensate 'old' exchanges in proportion to sender's mass going to target", action = "store_true") 
parser.add_argument("-fbr", "--blockAllRevenge", help = "Disable links to a galaxy that has hidden a galaxy previously", action = "store_true")
parser.add_argument("-fqh", "--quickHistory", help = "Enable quickHistory mode", action = "store_true")
parser.add_argument("-fqhl", "--quickHistoryLight", help = "Enable simple quickHistory mode", action = "store_true")
parser.add_argument("-fbo", "--bridgeInOrphans", help = "Include orphans of L2 links instead of disconnecting them", action = "store_true")
parser.add_argument("-fpl", "--protectWithLongLinks", help = "Conquering long-links must outweigh all shorter long-links to subhalo's progenitors", action = "store_true")

args = parser.parse_args()

if not args.dual_history:
    args.lim_recent = 0

if args.default_flags:
    args.long_links = True
    args.lower_ranks = True
    args.lower_choice = True
    args.limit_links = True
    args.compensate_history = True
    args.allow_orphans = True
    args.extract_tracers = True
    args.no_spectre_descendants = True
    args.trace_prebirth = True
    args.skip_stolen = True
    args.prevent_revenge = True
    args.consistent_selection = True
    args.dual_history = True # <---- (Not) CHANGED!!!! 
    args.lim_recent = 5 # <---- (Not) CHANGED !!!
    args.allow_multiple_longlinks = True
    args.exclude_all_linktargs = True
    args.require_core = True   # New 28-Apr-2018
    args.mix_history = True    # New 28-Apr-2018
    args.blockAllRevenge = True # New 9-May-2018
    args.bridgeInOrphans = True  # Added 14-May-2018
    args.protectWithLongLinks = True # Added 14-May-2018
    args.quickHistory = True       # Added 14-May-2018
    args.quickHistoryLight = True  # Addd 14-May-2018

rundir = args.dir
if args.num_snaps is None:
    nsnap = 30
else:
    nsnap = args.num_snaps

if args.spiderloc is None:
    spiderloc = rundir + '/highlev/SpiderwebLinks.hdf5'
else:
    spiderloc = args.spiderloc

if args.outloc is None:
    outloc = ht.clone_dir(rundir) + '/highlev/SpiderwebTablesMay18.hdf5'
else:
    outloc = args.outloc

if args.selection_priority is None:
    selectionPriority = 'rank'
else:
    selectionPriority = args.selection_priority

if args.sim_frac is None:
    sim_frac = 2/3
else:
    sim_frac = args.sim_frac

if args.stolen_threshold is None:
    stolen_skip_threshold = 1/3
else:
    stolen_skip_threshold = args.stolen_threshold


flag_interaction_list =        True
flag_computeFree =             True
flag_computeStolenFrac =       True     # Compute fraction stolen from others

xlist = []
snaps = []
allLinks = []
partialExchangeHistory = []

print("")
print("="*len(rundir))
print(rundir)
print("="*len(rundir))
print("")

tstring = "Start timestamp: " + time.strftime("%A, %d %B %Y at %H:%M:%S (NL)", time.localtime())

print('-'*len(tstring))
print(tstring)
print('-'*len(tstring))
print("")

class Status:
    
    """
    Class to hold global 'current status' variables
    """

    def __init__(self):
        
        self.jsnap = -1
        self.isnap = -1
        self.maxgal = -1
        self.iorig = -1
        self.nback = -1

    def update_snap(self, jsnap):
        self.jsnap = jsnap
        self.isnap = jsnap-1

    def update_maxgal(self, num_new):
        self.maxgal = self.maxgal + num_new

    

class Snapshot:

    """
    Class to represent one individual snapshot
    """

    def __init__(self, isnap, nHaloes):
        self.isnap = isnap
        self.nHaloes = nHaloes
        self.maxgal = -1

        self.sdtype_local = [('Galaxy', 'i'), ('Flags', 'i'), ('fracStolen', 'd'), ('TempNewGalFlag', 'i1'), ('LongLinkMinMass', 'd')]

        self.sdtype_rev = [('Link', 'i4'),            
                ('SHI', 'i4'),                   
                ('Length', 'i1'),
                ('CoreRank', 'i4'), 
                ('Rank', 'i4'), 
                ('Choice', 'i4'), 
                ('UncompensatedChoice', 'i4'), 
                ('CompensatedNormRF', 'd'), 
                ('NormRF', 'd'), 
                ('NormRFn', 'd'), 
                ('FreeMassFrac', 'd'),          # Previously free fraction, by mass 
                ('FreeNumFrac', 'd'),           # Previously free fraction, by particle number
                ('FreeMassFracPrev', 'd'),
                ('FreeNumFracPrev', 'd'),
                ('NormSF', 'd'), 
                ('NormCSF', 'd'), 
                           ('CoreNumPart', 'i8'),
                           ('LinkMass', 'd'),
                           ('LinkMassUncomp', 'd'),
                           ('NumShortLinks', 'i4')]


        self.sdtype_fwd = [('Link', 'i4'), # Link index of the connection
                ('SHI', 'i4'),                   # Subhalo index of descendant
                ('Length', 'i1'),                # Number of snaps to descendant
                ('CoreRank', 'i4'),              # Core rank of connection link
                ('Rank', 'i4'),                  # (Full) rank of connection link
                ('Choice', 'i4'),                # Choice (mass-weighted, compensated) of connection link
                ('UncompensatedChoice', 'i4'),   # Choice (mass-weighted, non-compensated) of c.l.
                ('CompensatedNormRF', 'd'),      # m_link / Sum (m_link_i) over all links to descendant
                ('NormRF', 'd'),                 # As above, but without exchange compensation
                ('NormRFn', 'd'),                # As above, but based on particle numbers, not mass
                ('FreeFrac', 'd'),               # (Number) fraction of particles that are not in any SH 
                ('FreeCFrac', 'd'),              # As above, but only for core particles
                ('FreeFracNext', 'd'),
                ('FreeCFracNext', 'd'),
                ('NormSF', 'd'),                 # N_link / Sum (N_link_i) over all links from this subhalo
                ('NormCSF', 'd'),                # As above, but only for core particles
                           ('CoreNumPart', 'i8'),           # Number of core particles in connection link
                           ('NumShortLinks', 'i4'),
                           ('NumSpawned', 'i4'),
                           ('LengthTemp', 'i1'),  # temp during LL selection
                           ('SHITemp', 'i4'),      # temp during LL selection
                           ('SHI_CR0', 'i4')]  # receiver of (core-)rank 0

        self.local = np.zeros(nHaloes, dtype = self.sdtype_local)
        self.local[:] = (-1, 0, 0, 1, 0)  # Initial: everything new galaxy!
        self.reverse = np.zeros(nHaloes, dtype = self.sdtype_rev)
        self.reverse[:] = (-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0)
        self.forward = np.zeros(nHaloes, dtype = self.sdtype_fwd)
        self.forward[:] = (-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1)


        self.fdtype = [('flag', 'U32'), ('xpl', 'U200')]
        self.flag_list = np.array([("NOPROGENITOR_SHORT", "subhalo has no progenitor in previous snapshot (may still be long-linked)"),
                                   ("NORECVLINK_SHORT", "subhalo does not receive any short-link"),
                              ("NODESCENDANT_SHORT", "subhalo has no descendant in next snapshot (may still be long-linked)"),
                              ("NOSENTLINK_SHORT", "subhalo does not send any short-links"),
                              ("NEW_GAL", "newly emerged galaxy"),
                              ("LASTSNAP", "last snapshot of this galaxy"),
                              ("MINOR_MERGER", "galaxy underwent at least one minor merger (1:10 - 1:3) in previous snapshot interval"),
                              ("MAJOR_MERGER", "galaxy underwent at least one major merger (1:3 - 2:3) in previous snapshot interval"),
                              ("NEAREQUAL_MERGER", "galaxy underwent at least one near-equal merger (> 2:3) in previous snapshot interval"),
                              ("MISSEDCONNECT_L2_REV", "no link back to progenitor 2 snapshots ago"),
                              ("MISSEDCONNECT_L3_REV", "no link back to progenitor 3 snapshots ago"),
                              ("MISSEDCONNECT_L4_REV", "no link back to progenitor 4 snapshots ago"),
                              ("MISSEDCONNECT_L5_REV", "no link back to progenitor 5 snapshots ago"),

                              ("MISSEDCONNECT_L2_FWD", "no link back to progenitor 2 snapshots ago"),
                              ("MISSEDCONNECT_L3_FWD", "no link back to progenitor 3 snapshots ago"),
                              ("MISSEDCONNECT_L4_FWD", "no link back to progenitor 4 snapshots ago"),
                              ("MISSEDCONNECT_L5_FWD", "no link back to progenitor 5 snapshots ago"),


                              ("LOWCORENUM", "best back-link has < 3 core particles"),
                              ("NOCORE", "best back-link does not have any core particles"),
                              ("CHOICE_AFFECTED_BY_COMPENSATION", "choice of best back-link changed during history compensation"),
                              ("PROG_SPAWNED", "the progenitor subhalo spawned at least one other subhalo, which may contain mass missing from this one"),
                              ("MERGED_ONTO_NEW", "subhalo merges onto newly formed galaxy in next snapshot interval"),
                              ("SPAWNED_FROM_MERGED", "subhalo is spawned from a galaxy that has since merged"),
                              ("NEW_WITH_MERGER", "newly formed galaxy that has already experienced mergers"),
                            ("LONG_OVER_LONG", "long link overwrote a (shorter) long link"),
                            ("LONG_OVER_SHORT", "long link overwrote a short-link"),
                                   ("ORPHAN", "detached from original progenitor because this led to long-link match"),
                                   ("SPECTRE", "new galaxy that is mostly formed from another galaxy's particles"),
                                   ("STOLEN", "estimated stolen mass fraction is > {:.2f}" .format(stolen_skip_threshold)),
                                   ("LONG_OVERRULED", "subhalo had a long-link match which was overruled by a longer link to an earlier snapshot"),
                                   ("SHORT_OVERRULED", "subhalo had a short-link match which was overruled by a long link to an earlier snapshot")

                               ], dtype = self.fdtype)

        self.nFlags = len(self.flag_list)
            

    def write_flag_list(self, outloc):
        flag_str_arr = np.zeros(len(self.flag_list), dtype = '|S100')#np.string_)
        flagXpl_str_arr = np.zeros(len(self.flag_list), dtype = '|S1000')# np.string_)

        for ii in range(len(flag_str_arr)):
            flag_str_arr[ii] = np.string_(self.flag_list['flag'][ii])
            flagXpl_str_arr[ii] = np.string_(self.flag_list['xpl'][ii])
        
        yb.write_hdf5(flag_str_arr, outloc, "Header/Flags", comment = "Flag names")
        yb.write_hdf5(flagXpl_str_arr, outloc, "Header/FlagDescriptions", comment = "Flag descriptions")
        

    def update_reverse(self, links, update = None, ind_link = None, connection = False):
        
        if update is None:
            update = np.arange(self.nHaloes, dtype = int)

        if ind_link is None:
            ind_link = self.reverse['Link'][update]
        else:
            self.reverse['Link'][update] = ind_link

        self.reverse['SHI'][update] = links.links[ind_link]['sender']
        self.reverse['LinkMass'][update] = links.links[ind_link]['compensatedMass']
        subind_newLLMM = np.nonzero(links.links[ind_link]['compensatedMass'] > self.local['LongLinkMinMass'][update])[0]
        self.local['LongLinkMinMass'][update[subind_newLLMM]] = links.links[ind_link[subind_newLLMM]]['compensatedMass']

        self.reverse['LinkMassUncomp'][update] = links.links[ind_link]['mass']
        
        if connection:
            self.reverse['Length'][update] = links.length

            if flag_computeFree:
                self.reverse['FreeMassFrac'][update] = links.freeRecv['Mass'][update]
                self.reverse['FreeNumFrac'][update] = links.freeRecv['Num'][update]

            
        self.reverse['CoreRank'][update] = links.links['coreRank'][ind_link]
        self.reverse['Rank'][update] = links.links['rank'][ind_link]
        self.reverse['Choice'][update] = links.links['compensatedChoice'][ind_link]
        self.reverse['UncompensatedChoice'][update] = links.links['choice'][ind_link]
        self.reverse['CompensatedNormRF'][update] = links.links['compensatedNormRF'][ind_link]
        self.reverse['NormRF'][update] = links.links['normRF'][ind_link]
        self.reverse['NormRFn'][update] = links.links['normRFn'][ind_link]
        self.reverse['CoreNumPart'][update] = links.links['coreNumPart'][ind_link]
        self.reverse['NormSF'][update] = links.links['normSF'][ind_link]
        self.reverse['NormCSF'][update] = links.links['normCSF'][ind_link]


    def update_forward(self, links, update = None, ind_link = None, connection = False):
        
        if update is None:
            update = np.arange(self.nHaloes, dtype = int)
            
        if ind_link is None:
            ind_link = self.forward['Link'][update]
        else:
            self.forward['Link'][update] = ind_link

        if connection:
            self.forward['Length'][update] = links.length

            if flag_computeFree:
                self.forward['FreeFrac'][update] = links.freeSend['All'][update]
                self.forward['FreeCFrac'][update] = links.freeSend['Core'][update]

        self.forward['SHI'][update] = links.links[ind_link]['receiver']
        self.forward['CoreRank'][update] = links.links['coreRank'][ind_link]
        self.forward['Rank'][update] = links.links['rank'][ind_link]
        self.forward['Choice'][update] = links.links['compensatedChoice'][ind_link]
        self.forward['UncompensatedChoice'][update] = links.links['choice'][ind_link]
        self.forward['CompensatedNormRF'][update] = links.links['compensatedNormRF'][ind_link]
        self.forward['NormRF'][update] = links.links['normRF'][ind_link]
        self.forward['NormRFn'][update] = links.links['normRFn'][ind_link]
        self.forward['CoreNumPart'][update] = links.links['coreNumPart'][ind_link]
        self.forward['NormSF'][update] = links.links['normSF'][ind_link]
        self.forward['NormCSF'][update] = links.links['normCSF'][ind_link]

    def build_revGal(self):
        self.maxgal = self.local['Galaxy'].max()
        self.revGal = np.zeros(self.maxgal+1, dtype = int)-1
        ind_goodGal = np.nonzero(self.local['Galaxy'] >= 0)[0]
        self.revGal[self.local['Galaxy'][ind_goodGal]] = ind_goodGal
                                                      
    def gal_to_sh(self, gal): 
        
        ind_in_range = np.nonzero(gal < self.revGal.shape[0])[0]
        sh_in_range = self.revGal[gal[ind_in_range]]

        sh_out = np.zeros(len(gal), dtype = np.int32)-1
        sh_out[ind_in_range] = sh_in_range

        return sh_out


    def set_forward_flags(self):
            
        # Next line works because we call it in the next snapshot, when
        # *only* short-links have been connected
        self.add_flag(np.nonzero(self.forward['Length'] == 0)[0], "NODESCENDANT_SHORT")
        self.add_flag(np.nonzero(self.forward['NumShortLinks'] == 0)[0], "NOSENTLINK_SHORT")

    def set_reverse_flags(self, match_ji):

        # Next line does not work analogous to above, because short-links
        # may have been overridden by long links
        self.add_flag(np.nonzero(match_ji < 0)[0], "NOPROGENITOR_SHORT")
        self.add_flag(np.nonzero(self.reverse['NumShortLinks'] == 0)[0], "NORECVLINK_SHORT")
        
        ind_lowcore = np.nonzero((self.reverse['Length'] > 0) & (self.reverse['CoreNumPart'] < 3) & (self.reverse['CoreNumPart'] > 0))[0]
        self.add_flag(ind_lowcore, "LOWCORENUM")
        
        ind_nocore = np.nonzero((self.reverse['Length'] > 0) & (self.reverse['CoreNumPart'] == 0))[0]
        self.add_flag(ind_nocore, "NOCORE")

        ind_compchange = np.nonzero((self.reverse['Length'] > 0) & (self.reverse['Choice'] != self.reverse['UncompensatedChoice']))[0]
        self.add_flag(ind_compchange, "CHOICE_AFFECTED_BY_COMPENSATION")

        n_matched = np.count_nonzero(self.reverse['Length'] > 0)
        self.nMatchedRev = n_matched
        
        if n_matched > 0:
            print("   ---> {:d} galaxies ({:.2f}%) have matched choice affected by history compensation" 
              .format(len(ind_compchange), len(ind_compchange)/n_matched*100)) 
            print("   ---> {:d} galaxies ({:.2f}%) are linked back by 0 < x < 3 core particles" 
              .format(len(ind_lowcore), len(ind_lowcore)/n_matched*100)) 
            print("   ---> {:d} galaxies ({:.2f}%) are linked back without any core particles" 
              .format(len(ind_nocore), len(ind_nocore)/n_matched*100)) 

        else:
            print(" ========= WARNING!! ============ ")
            print(" --- no subhalo match at all! --- ")
            print(" ================================ ")
        

    def initialize_new_galaxies(self, strict_check = False):
        
        ind_new = np.nonzero(self.local['TempNewGalFlag'] == 1)[0]
        n_new = len(ind_new)

        if n_new == 0:
            return

        self.add_flag(ind_new, "NEW_GAL")
        self.local['Galaxy'][ind_new] = np.arange(main.maxgal+1, main.maxgal+n_new+1, dtype = int)
        main.update_maxgal(n_new)

        self.verify_galaxyID_unique(strict = strict_check)
        self.build_revGal()

        self.local['TempNewGalFlag'][ind_new] = 0
            

    def verify_galaxyID_unique(self, strict = False):

        if strict:
            if len(np.unique(self.local['Galaxy'])) != self.nHaloes:
                print("Galaxies are not unique!!!")
                set_trace()
        
        else:
            ind_real = np.nonzero(self.local['Galaxy'] >= 0)[0]
            if len(ind_real) > 0:
                if len(np.unique(self.local['Galaxy'][ind_real])) != len(ind_real):
                    set_trace()


    def _flagBit(self, flag):
        
        flagBit = np.nonzero(self.flag_list['flag'] == flag)[0]
        
        if len(flagBit) != 1:
            print("Could not find specified flag '" + flag + "'")
            set_trace()

        return flagBit[0]


    def add_flag(self, shi, flag):

        """
        Set a flag bit for the subhaloes listed in shi
        """
        
        self.local['Flags'][shi] |= 2**self._flagBit(flag)


    def remove_flag(self, shi, flag):

        """
        Remove a flag bit (set to zero)
        """

        ind_flag_is_set = np.nonzero(self.local['Flags'][shi] & 2**self._flagBit(flag))[0]
        if len(ind_flag_is_set) > 0:
            self.local['Flags'][shi[ind_flag_is_set]] -= 2**self._flagBit(flag) 


    def check_flag(self, flag, shi = None):
        
        """
        Check whether the FLAG for subhaloes SHI is set
        """

        if shi is None:
            shi = np.arange(self.nHaloes, dtype = np.int32)

        return (self.local['Flags'][shi] & 2**self._flagBit(flag)) > 0

            
    def test_flag(self, flag, shi = None):
        
        """
        Return a list of all subhaloes in shi for which FLAG is set
        """

        if shi is None:
            shi = np.arange(self.nHaloes, dtype = np.int32)

        return np.nonzero(self.check_flag(flag, shi))[0]
    
    def get_flags(self, shi):
        
        """
        Helper/debug function to return the flag NAMES for one subhalo.
        """

        retlist = []
        for flag in self.flag_list['flag']:
            if self.check_flag(flag, shi):
                retlist.append(flag)

        return retlist

    def update_longLink_threshold(self, links):

        """
        Update the threshold for a long-link to connect to a subhalo
        in snapshot SELF. 
        
        Find current-best-guess-galaxy and then determine its long-link.
        
        In contrast to check_backwards_consistency() below, this is called
        during the long-link evaluation already.
        """

        # Find current best match for j subhaloes:
        sh_match = self.reverse['SHI']
        sh_length = self.reverse['Length']
        best_gal_guess = np.zeros(len(sh_match), dtype = np.int32)-1
        
        for nback in np.unique(sh_length):
            
            # Exclude unmatched galaxies:
            if nback == 0: continue

            ind_this_back = np.nonzero(sh_length == nback)[0]
            curr_o = self.isnap-nback
            best_gal_guess[ind_this_back] = snaps[curr_o].local['Galaxy'][sh_match[ind_this_back]]

        # Find corresponding subhalo in snapshot o (links origin snap):
        # (N.B.: SHI function already takes care of out-of-range entries)
        sh_in_o = snaps[links.isnap].SHI(best_gal_guess)

        # And now find those subhaloes (now) that have a subhalo in o
        # with the same (best-guess) galaxy number
        # N.B.: ind_galexist_o is an index into FULL CURRENT subhalo list
        #     (not into subset thereof as in check_backwards_consistency)
        
        ind_galexist_o = np.nonzero(sh_in_o >= 0)[0] 
        marklist_linked = np.zeros(self.nHaloes, dtype = np.int8)
        marklist_linked[ind_galexist_o] = 1
        
        # find_links() returns all links between given sender and receiver
        gallinks = links.find_links(sh_in_o[ind_galexist_o], ind_galexist_o)
        
        # Now update the threshold:
        ind_strongerBound = np.nonzero(links.links['compensatedMass'][gallinks] > self.local['LongLinkMinMass'][links.links['receiver'][gallinks]])[0]
        self.local['LongLinkMinMass'][links.links['receiver'][gallinks[ind_strongerBound]]] = links.links['compensatedMass'][gallinks[ind_strongerBound]]
        
        
        
    def check_backwards_consistency(self, links):
        
        """
        Check whether galaxy progenitor is connected to its current subhalo
        by a link/connection. Now includes L1 as well, as consistency check
        """

        # Load galaxies at check snapshot (o) and set up reverse list:
        galaxies_o = snaps[links.isnap].local['Galaxy']
        
        # Find subhaloes in o corresponding to current subhaloes,
        # from established connections.
        # Can only do this for subhaloes that have been matched already
        # (non-negative Galaxy entry) and that exist in o snapshot
        
        sh_in_j = np.nonzero(self.local['Galaxy'] >= 0)[0]
        sh_in_o = snaps[links.isnap].SHI(self.local['Galaxy'][sh_in_j])

        if links.length >= 2:

            # And now find those subhaloes (now) that have a subhalo in o
            # with the same galaxy number, but without a long-link connection
            # N.B.: ind_galexist_o is an index into CURRENT subhalo list

            ind_galexist_o = np.nonzero(sh_in_o >= 0)[0] 
            marklist_linked = np.zeros(self.nHaloes, dtype = np.int8)
            marklist_linked[sh_in_j[ind_galexist_o]] = 1
        
            # find_links() returns all links between given sender and receiver
            gallinks = links.find_links(sh_in_o[ind_galexist_o], sh_in_j[ind_galexist_o])
        
            # Mark connections where a link could be found
            marklist_linked[links.links['receiver'][gallinks]] = 2

            # New bit added 14-May-2018: also increase 'threshold'
            # for long-link connections if self-long-link is stronger
            # than current best connection

            if args.protectWithLongLinks:
                ind_strongerBound = np.nonzero(links.links['compensatedMass'][gallinks] > self.local['LongLinkMinMass'][links.links['receiver'][gallinks]])[0]
                self.local['LongLinkMinMass'][links.links['receiver'][gallinks[ind_strongerBound]]] = links.links['compensatedMass'][gallinks[ind_strongerBound]]

            sh_missedConnect_j = np.nonzero(marklist_linked == 1)[0]
            
            print("   --> {:d}/{:d} (={:.2f}%) galaxies are not long-linked between snaps {:d}-->{:d}!" .format(len(sh_missedConnect_j), len(ind_galexist_o), len(sh_missedConnect_j)/len(ind_galexist_o)*100, links.isnap, self.isnap))
            
            self.add_flag(sh_missedConnect_j, "MISSEDCONNECT_L{:d}_REV" .format(links.length))

            sh_missedConnect_o = snaps[links.isnap].SHI(self.local['Galaxy'][sh_missedConnect_j])

            if len(sh_missedConnect_j) > 0:
                if np.min(sh_missedConnect_o) < 0:
                    print("Why is there a missed connection to a non-existing galaxy?")
                    set_trace()

            snaps[links.isnap].add_flag(sh_missedConnect_o, "MISSEDCONNECT_L{:d}_FWD" .format(links.length))


        if links.length == 1:

            # In this case, it's more serious: the galaxies must be 
            # progenitor and descendant of each other!

            ind_l1 = np.nonzero(self.reverse['Length'][sh_in_j] == 1)[0]
            if len(ind_l1) > 0:
                if np.count_nonzero(self.reverse['SHI'][sh_in_j[ind_l1]] != sh_in_o[ind_l1]) > 0:
                    print("Inconsistent galaxy matches (rev)!")
                    set_trace()

            ind_l1_o = np.nonzero(snaps[links.isnap].forward['Length'] == 1)[0] 
            gal_l1_o = snaps[links.isnap].local['Galaxy'][ind_l1_o]
            sh_l1_j = self.SHI(gal_l1_o)

            if np.count_nonzero(snaps[links.isnap].forward['SHI'][ind_l1_o] != sh_l1_j) > 0:
                print("Inconsistent galaxy matches (fwd)!")
                set_trace()


    def write(self, outloc):

        """
        Write to output file
        """

        sdir = 'Subhalo/Snapshot_' + str(self.isnap).zfill(3) + '/'

        # ---------------------
        # 1.) LOCAL properties
        # ---------------------
        
        flagComment = "Quality flags, stored as Sum(2**i), with i == 1 if the flag is set and i == 0 otherwise. Individual flags are:"
        for iflag in range(self.nFlags):
             flagComment += " {:d} (" .format(iflag) + self.flag_list['flag'][iflag] + ") - " + self.flag_list['xpl'][iflag]
             if iflag != self.nFlags-1:
                 flagComment += ";"
            
        yb.write_hdf5(self.local['Flags'], outloc, sdir + 'Flags', comment = flagComment)

        yb.write_hdf5(self.local['Galaxy'], outloc, sdir + 'Galaxy', comment = "Galaxy corresponding to this subhalo. The descendant or (main) progenitor in other snapshots is the subhalo with the same galaxy number.")

        yb.write_hdf5(self.local['fracStolen'], outloc, sdir + 'StolenFraction', comment = "Estimated mass fraction of this subhalo that is temporarily stolen from other galaxies. If this is large, the mass and/or position of this subhalo may not be reliable.")

        # ---------------------
        # 2.) REVERSE properties
        # ---------------------
        
        sdir = 'Subhalo/Snapshot_' + str(self.isnap).zfill(3) + '/Reverse/'

        yb.write_hdf5(self.reverse['Length'], outloc, sdir + 'Length', comment = "How many snapshots back the progenitor could be identified. Standard is 1, and 0 indicates that this is a newly formed galaxy.")
        
        yb.write_hdf5(self.reverse['SHI'], outloc, sdir + 'SubHaloIndex', comment = "Subhalo index of progenitor in match snapshot (i.e. 'Length' snapshots backwards). For new galaxies, the subhalo contributing most of its mass, if any.")

        yb.write_hdf5(self.reverse['Link'], outloc, sdir + 'LinkIndex', comment = "Link index connecting this subhalo to its descendant, or choice-0 link in case of a new galaxy")

        yb.write_hdf5(self.reverse['CoreRank'], outloc, sdir + 'CoreRank', comment = "*Core* rank of link, i.e. how many other subhaloes in this snapshot receive more core particles from the identified progenitor than this one.")

        yb.write_hdf5(self.reverse['Rank'], outloc, sdir + 'FullRank', comment = "*Full* rank of link, i.e. how many other subhaloes in this snapshot receive more particles from the identified progenitor than this one.")

        yb.write_hdf5(self.reverse['Choice'], outloc, sdir + 'Choice', comment = "Choice of link, i.e. how many other subhaloes in match snapshot send more mass to this one than the identified progenitor, accounting for history compensation.")

        yb.write_hdf5(self.reverse['UncompensatedChoice'], outloc, sdir + 'UncompensatedChoice', comment = "Choice of link, i.e. how many other subhaloes in match snapshot send more mass to this one than the identified progenitor, *but without* accounting for history compensation.")

        yb.write_hdf5(self.reverse['CompensatedNormRF'], outloc, sdir + 'ReceiverFraction', comment = "Mass fraction of this subhalo received from progenitor, normalised to mass received from all subhaloes and including history compensation.")

        yb.write_hdf5(self.reverse['NormRF'], outloc, sdir + 'UncompensatedReceiverFraction', comment = "Mass fraction of target received from this link, normalised to mass received from all subhaloes but NOT including history compensation.")

        yb.write_hdf5(self.reverse['NormRFn'], outloc, sdir + 'ReceiverNumberFraction', comment = "Particle number fraction of target received from this link, normalised to number received from all subhaloes (does not include history compensation).")
    
        yb.write_hdf5(self.reverse['FreeNumFracPrev'], outloc, sdir + 'FreeNumberFractionPrev', comment = "Particle number fraction of this subhalo that was not bound to any subhalo in the previous snapshot.")

        yb.write_hdf5(self.reverse['FreeMassFracPrev'], outloc, sdir + 'FreeMassFractionPrev', comment = "Mass fraction of particles in this subhalo that were not bound to any subhalo in the previous snapshot.")

        yb.write_hdf5(self.reverse['FreeNumFrac'], outloc, sdir + 'FreeNumberFraction', comment = "Particle number fraction of this subhalo that was not bound to any subhalo in the match snapshot.")

        yb.write_hdf5(self.reverse['FreeMassFrac'], outloc, sdir + 'FreeMassFraction', comment = "Mass fraction of particles in this subhalo that were not bound to any subhalo in the match snapshot.")

        yb.write_hdf5(self.reverse['NormSF'], outloc, sdir + 'SenderFraction', comment = "(Number) fraction of all particles of the progenitor subhalo that are bound to this subhalo, normalised to the total number of these particles that are now bound to subhaloes. For new galaxies, the corresponding subhalo is that providing the most mass to this subhalo.") 
        
        yb.write_hdf5(self.reverse['NormCSF'], outloc, sdir + 'CoreSenderFraction', comment = "(Number) fraction of all core particles of the progenitor subhalo that are bound to this subhalo, normalised to the total number of these particles that are now bound to subhaloes. For new galaxies, the corresponding subhalo is that providing the most mass to this subhalo.")
        
        yb.write_hdf5(self.reverse['CoreNumPart'], outloc, sdir + 'CoreNumPart', comment = "Number of core particles of the progenitor that are bound to this subhalo. For new galaxies, the corresponding subhalo is that providing the most mass to this subhalo.")

        # ---------------------
        # 3.) FORWARD properties
        # ---------------------
        
        sdir = 'Subhalo/Snapshot_' + str(self.isnap).zfill(3) + '/Forward/'

        yb.write_hdf5(self.forward['Link'], outloc, sdir + 'LinkIndex', comment = "Level1 link connecting this subhalo to its descendant, or rank-0 link in case of merged or hidden subhaloes")
        
        yb.write_hdf5(self.forward['Length'], outloc, sdir + 'Length', comment = "How many snapshots forward the progenitor could be identified. Standard is 1, and 0 indicates that this is the last snapshot of this galaxy.")

        yb.write_hdf5(self.forward['SHI'], outloc, sdir + 'SubHaloIndex', comment = "Subhalo index of descendant in match snapshot (i.e. 'Length' snapshots forward). For merged galaxies (Length == 0), the subhalo with which it merged.")

        yb.write_hdf5(self.forward['SHI_CR0'], outloc, sdir + 'SubHaloIndexCR0', comment = "Subhalo index of trivial descendant in match snapshot (i.e. 'Length' snapshots forward), i.e. the receiver of the core-rank 0 link. If no links are sent by a subhalo at all, this is set to -1.")

        yb.write_hdf5(self.forward['CoreRank'], outloc, sdir + 'Rank', comment = "*Core* rank of link, i.e. how many other subhaloes receive more core particles from this subhalo than the target. Always 0 for merged galaxies")
        
        yb.write_hdf5(self.forward['Rank'], outloc, sdir + 'FullRank', comment = "*Full* rank of link, i.e. how many other subhaloes receive more particles *in total* from this subhalo than the target. This may be non-zero even for merged galaxies.")
 
        yb.write_hdf5(self.forward['Choice'], outloc, sdir + 'Choice', comment = "Choice of link, i.e. how many other subhaloes send more mass to the target than this, accounting for history compensation.")

        yb.write_hdf5(self.forward['UncompensatedChoice'], outloc, sdir + 'UncompensatedChoice', comment = "As Choice, but without history compensation.")

        yb.write_hdf5(self.forward['CompensatedNormRF'], outloc, sdir + 'ReceiverFraction', comment = "Mass fraction of target received from this subhalo, normalised to mass received from all subhaloes and including history compensation.")

        yb.write_hdf5(self.forward['NormRF'], outloc, sdir + 'UncompensatedReceiverFraction', comment = "Mass fraction of target received from this link, normalised to mass received from all subhaloes but NOT including history compensation.")

        yb.write_hdf5(self.forward['NormRFn'], outloc, sdir + 'ReceiverNumberFraction', comment = "Particle number fraction of target received from this link, normalised to number received from all subhaloes (does not include history compensation).")

        yb.write_hdf5(self.forward['FreeFracNext'], outloc, sdir + 'FreeFractionNext', comment = "Particle number fraction of this subhalo that is not bound to any subhalo in the next snapshot.")

        yb.write_hdf5(self.forward['FreeCFracNext'], outloc, sdir + 'FreeCoreFractionNext', comment = "Fraction of core particles in this subhalo that is not bound to any subhalo in the next snapshot.")
        
        yb.write_hdf5(self.forward['FreeFrac'], outloc, sdir + 'FreeFraction', comment = "Particle number fraction of this subhalo that is not bound to any subhalo in the match snapshot.")

        yb.write_hdf5(self.forward['FreeCFrac'], outloc, sdir + 'FreeCoreFraction', comment = "Fraction of core particles in this subhalo that is not bound to any subhalo in the match snapshot.")
        
        yb.write_hdf5(self.forward['NormSF'], outloc, sdir + 'SenderFraction', comment = "Fraction of all particles of this subhalo that is bound to the target subhalo in the next snapshot.")
        
        yb.write_hdf5(self.forward['NormCSF'], outloc, sdir + 'CoreSenderFraction', comment = "Fraction of all core particles of this subhalo that is bound to the target subhalo in the next snapshot.")
        
        yb.write_hdf5(self.forward['CoreNumPart'], outloc, sdir + 'CoreNumPart', comment = "Number of core particles of this subhalo that are bound to the target subhalo in the next snapshot.")
        
        yb.write_hdf5(self.forward['NumSpawned'], outloc, sdir + 'NumSpawned', comment = "Number of new galaxies in the next snapshot that derive more mass from this subhalo than from any other.")

                      
        #"Quality flags, stored as Sum(2**i) with i == 1 if the flag is set. Individual flags are: 0 - no progenitor in previous snapshot; 1 - no link to any subhalo in previous snapshot; 2 - no descendant in following snapshot; 3 - no link to any subhalo in following snapshot; 4 - new galaxy; 5 - last snapshot of galaxy; 6/7/8 - subhalo had at least one minor (1:10-1:3) / major (1:3-2:3) / near-equal (>2:3) merger since last snapshot; 9/10/11/12 - subhalo does not have any link to its progenitor 2/3/4/5 snapshots ago; 13 - subhalo contains less than three of its progenitor's core particles; 14 - subhalo contains none of its progenitor's core particles; 15 - match choice is affected by history compensation; 16 - progenitor has spawned galaxies in this snapshot; 17 - subhalo merges with a newly formed galaxy by the next snapshot; 18 - new galaxy spawned from a galaxy that has merged; 19 - new galaxy that is the target of a galaxy that has merged")
        

    # override is called for jsnap
    def override(self, match):
        
        """
        Process situations where a new (long-)link overrides a previous
        match that is currently registered in this subhalo's tables.

        This also deals with the stupidities i.v.m. orphans.

        N.B.: We have not updated any of the current snap info yet. 
              So self.reverse['Length'/'SHI'] still refer to the 
              OLD = NOW-TO-BE-OVERRIDDEN connection.
        """
        
        ind_matched = np.nonzero(match >= 0)[0]
        if len(ind_matched) == 0:
            return

        # 'override' refers to (j) subhalo where override occurs
        # 'overruled' refers to (<j) subhalo that is now overridden
        ind_override = np.nonzero(self.reverse['Length'][ind_matched] > 0)[0]
        sh_override = ind_matched[ind_override]
        sh_overruled = self.reverse['SHI'][sh_override]

        # Set flags, depending on whether previous link is long/short:
        subind_over_long = np.nonzero(self.reverse['Length'][sh_override] > 1)[0]
        if len(subind_over_long) > 0:
            self.add_flag(sh_override[subind_over_long], "LONG_OVER_LONG")

            print("   ---> {:d} override previous long matches)" 
                  .format(len(subind_over_long)))
                    
        subind_over_short = np.nonzero(self.reverse['Length'][sh_override] == 1)[0]
        if len(subind_over_short) > 0:
            self.add_flag(sh_override[subind_over_short], "LONG_OVER_SHORT")
            
            print("   ---> {:d} override previous short matches)" 
                  .format(len(subind_over_short)))
     
            sh_old_sender = self.reverse['SHI'][sh_override[subind_over_short]]

            # IMPORTANT! ONLY set the length to zero and leave
            # everything else intact. That way it's consistent
            # with what's done for other merged galaxies.
            # (length needs to be updated immediately after L1-selection
            # to enable L1-flags to be written)

            snaps[main.isnap].forward['Length'][sh_old_sender] = 0
                   

        # Now we need to also look at the senders of the overruled links.
        # Two things need doing: set flags, and reverse orphanization 
        # (where necessary)

        for iback in np.unique(self.reverse['Length'][sh_override]):
        
            ind_this_back = np.nonzero(self.reverse['Length'][sh_override] == iback)[0]
            iold = self.isnap - iback

            # Set appropriate flags at the senders:
            if iback == 1:
                snaps[iold].add_flag(sh_overruled[ind_this_back], "SHORT_OVERRULED")
            else:
                snaps[iold].add_flag(sh_overruled[ind_this_back], "LONG_OVERRULED")
            
            # Check whether the just overruled sender had a descendant
            # and we are at least at iback == 2.
            # (if iback == 1, this is just a normal merger)
            # If yes, this was a Case-B long-link and the descendant is now
            # an orphan, which needs to be re-united with its original parent
            # N.B.: This works because forward['Length'] has not been updated
            # yet and so still reflects the pre-orphan state 

            if iback > 1:
                
                ind_exOrphan_parent = np.nonzero(snaps[iold].forward['Length'][sh_overruled[ind_this_back]] == 1)[0]
                
                sh_exOrphan_parent = sh_overruled[ind_this_back[ind_exOrphan_parent]]
                if len(ind_exOrphan_parent) > 0:

                    sh_exOrphan = snaps[iold].forward['SHI'][sh_exOrphan_parent]

                    # Safety check:
                    if np.count_nonzero(snaps[iold+1].local['Flags'][sh_exOrphan] & 2**snaps[iold+1]._flagBit('ORPHAN') == 0) > 0:
                        print("Inconsistency in orphan flags...")
                        set_trace()
    
                    # Do what is necessary to de-orphanize ex-orphan:
                    snaps[iold+1].reverse['Length'][sh_exOrphan] = 1
                    snaps[iold+1].local['TempNewGalFlag'][sh_exOrphan] = 0

                    # N.B.: No need to adjust galaxy IDs, since these have
                    # not been changed yet for the orphan.
                    # But the orphan flag is set immediately to enable safety-
                    # checks, so we need to manually unset it now:
                    snaps[iold+1].remove_flag(sh_exOrphan, "ORPHAN")
                


        
    def make_orphans(self, match, connMode):

        """
        Create orphans, i.e. deal with situations where a (dying)
        galaxy's progenitor could be Case-B long-link-matched
        
        There is NO NEED to deal with the "associated stupidities"
        here already, since we're not sure the orphan will stay an 
        orphan until the end of the long-link selection iteration...
        """
    
        # New check added 14-May-18: do not include 'bridged' galaxies
        #    (connMode == 2)
        ind_matched = np.nonzero((match >= 0) & (connMode < 2))[0]
        if len(ind_matched) == 0:
            return
            
        # Safety checks first:
        if np.count_nonzero(snaps[main.iorig].forward['Length'][match[ind_matched]] > 1) > 0:
            print("Why is there a long-link from already long-linked site??")
            set_trace()
                
        if self.isnap != main.iorig+1:
            print("Inconsistent snapshots in orphans()")
            set_trace()

        # Now identify orphan parents and their (ex)-offspring:
        ind_orphanParent = np.nonzero(snaps[main.iorig].forward['Length'][match[ind_matched]] == 1)[0]

        if len(ind_orphanParent) > 0:

            # Need to modify existing subahlo fields to 'orphanize' the 
            # descendant of the selected senders. 
            # This may NOT be permanent, since the long-link may
            # later be overruled at the receiver, but we deal with 
            # that when needed (in function "override()")
            # So we here ONLY set its length to 0 and mark it as 
            # 'new'. This can both be undone later if need be.

            sh_orphanParent = match[ind_matched[ind_orphanParent]]
            sh_orphan = snaps[main.iorig].forward['SHI'][sh_orphanParent]

            self.reverse['Length'][sh_orphan] = 0
            self.local['TempNewGalFlag'][sh_orphan] = 1

            # All the "special fun" will be dealt with later, once
            # we're sure which galaxies are actually orphans
            # All we do here is set the warning flag (for consistency checks):
            
            self.add_flag(sh_orphan, "ORPHAN")

    def SHI(self, galaxies):

        """
        Function to look up the subhalo for a given set of galaxies 
        in current snapshot. Deals properly with out-of-range conditions.
        """
                                                    
        shi = np.zeros(len(galaxies), dtype = int)-1
        ind_in_range = np.nonzero((galaxies <= self.maxgal) & (galaxies >= 0))[0]
        
        if len(ind_in_range) > 0:
            shi[ind_in_range] = self.revGal[galaxies[ind_in_range]]

        return shi
                                                                        


    def orphan_consequences_old(self, sh_orphan, snap_orphan):

        """
        Things that need to be done with orphans BEFORE their galaxy ID 
        is changed.
        """

        if len(sh_orphan) == 0:
            return

        old_gal = snaps[snap_orphan].local['Galaxy'][sh_orphan]

        for iorig in range(snap_orphan-5, snap_orphan-1):

            if iorig < 0:
                continue
                    
            sh_thisorig = snaps[iorig].SHI(old_gal)
            ind_in_this_orig = np.nonzero(sh_thisorig >= 0)[0]
                
            if len(ind_in_this_orig) > 0:
                snaps[iorig].remove_flag(sh_thisorig[ind_in_this_orig], "MISSEDCONNECT_L{:d}_FWD" .format(snap_orphan-iorig))
                
            

    def orphan_consequences_new(self, sh_orphan, snap_orphan):

        """
        This function deals with all the complications arising from
        out-of-sync new galaxy creation (i.e. orphans). 
        
        It is only called after the long-link selection is completely
        finished, and we know for sure which subhaloes are actually orphans.

        Also, at the calling point, their IDs have already been updated. 
        Any modification requiring the old IDs are in ..._old()
        """
        
        if len(sh_orphan) == 0:
            return

        
        # Safety tests first:
        if snap_orphan < 1:
            print("Why are there orphans before snapshot 1?")
            set_trace()
        
        sh_orphanParent = snaps[snap_orphan].reverse['SHI'][sh_orphan]
        if np.count_nonzero(snaps[snap_orphan-1].forward['Length'][sh_orphanParent] + snap_orphan-1 != main.jsnap) > 0:
            print("Why is an orphan progenitor not matched to current snapshot?")
            set_trace()


        # ---------------------------------------------------
        # Special fun, Pt.I: need to add just-orphaned links 
        # to exchange and history lists...
        # ---------------------------------------------------

        old_xlist_num = xlist[snap_orphan].nXlink
        xlist[snap_orphan].add_new(snaps[snap_orphan].reverse['Link'][sh_orphan])
        if snap_orphan - args.lim_recent >= 0:
            exchange.update(xlist[snap_orphan], first = old_xlist_num)


        # -------------------------------------------------
        # Special fun, Pt.II: need to update spawn-counters
        # -------------------------------------------------
        
        snaps[snap_orphan-1].forward['NumSpawned'][sh_orphanParent] += 1

        
    

    def update_galaxyID(self, sh_j, sh_o, iorig):

        """
        Central function to update local['Galaxy'] counter.
        """
        
        # Safety check:
        if np.max(sh_j) >= self.nHaloes or np.min(sh_j) < 0:
            print("Wrong input subhalo in update_galaxyID!")
            set_trace()

        if np.max(sh_o) >= snaps[iorig].nHaloes or np.min(sh_o) < 0:
            print("Wrong input subhalo in update_galaxyID!")
            set_trace()
            
        self.local['Galaxy'][sh_j] = snaps[iorig].local['Galaxy'][sh_o]
        
        self.build_revGal()
        self.verify_galaxyID_unique()

        self.local['TempNewGalFlag'][sh_j] = 0


class ExchangeHistory:

    """
    Class to hold, compute, and provide information on past mass
    exchanges between galaxies.
    """

    def __init__(self):

        self.xdtype = [('gal_i', 'i'), ('gal_j', 'i'), ('mass', 'd')]
        self.xlist = np.zeros(0, dtype = self.xdtype)
        self.nX = 0
        self.qktime = 0  # perfmon
        self.qktime_python = 0 # perfmon
        self.qktime_ccat = 0
        self.qktime_sort = 0
        self.npairs = 0  # perfmon
        self.nruns  = 0  # perfmon

    def _reSort(self):
        
        """
        (re)-sort the list by i-galaxy entries.
        """

        self.xlist.sort(order = ['gal_i', 'gal_j'])
                

    def _katamaran_search_ccat(self, a, aIsSorted = False, useCtypes = True):
        
        """
        Hopefully faster version of search using C
        """
        
        if len(a) == 0:
            return np.zeros(0, dtype = np.int32)-1

        if self.nX == 0:
            return np.zeros(len(a), dtype = np.int32)-1
        
        tc = time.time()
    
        if not aIsSorted:
            set_trace()
            sorter_a = np.argsort(a, order = ['gal_i', 'gal_j'])
        """
        else:
            sorter_a = np.arange(len(a), dtype = int)
        """

        blocksize_a = int(np.sqrt(len(a)))
        if blocksize_a*blocksize_a != len(a):
            print("Inconsistent size of a")
            set_trace()
            

        if not useCtypes:
            locs_a_tmp = ckat.ckat(a['gal_i'][sorter_a],
                               a['gal_j'][sorter_a],
                               self.xlist['gal_i'],
                               self.xlist['gal_j'],
                               self.xOffset,
                               blocksize_a,
                               len(a),
                               self.nX,
                               len(self.xOffset))

        else:
            
            locs_a_tmp = np.zeros(len(a), dtype = int)-1
            ObjectFile = "/u/ybahe/EXPERIMENTAL/Spiderweb/ccat.so"
            lib = c.cdll.LoadLibrary(ObjectFile)
            ccat = lib.ccat
    
            vAI = a['gal_i']
            vAJ = a['gal_j']
            vXI = self.xlist['gal_i']
            vXJ = self.xlist['gal_j']
            vXOffset = self.xOffset

            vAI_p = vAI.ctypes.data_as(c.c_void_p)
            vAJ_p = vAJ.ctypes.data_as(c.c_void_p)
            vXI_p = vXI.ctypes.data_as(c.c_void_p)
            vXJ_p = vXJ.ctypes.data_as(c.c_void_p)
            vXOffset_p = vXOffset.ctypes.data_as(c.c_void_p)
        
            blocksize_c = c.c_int(blocksize_a)
            na_c = c.c_int(len(a))
            nx_c = c.c_int(self.nX)
            nXOff_c = c.c_int(len(self.xOffset))
            locs_a_p = locs_a_tmp.ctypes.data_as(c.c_void_p)

            myargv = c.c_void_p * 10
            argv = myargv(vAI_p, vAJ_p, vXI_p, vXJ_p,
                          vXOffset_p,
                          c.addressof(blocksize_c),
                          c.addressof(na_c),
                          c.addressof(nx_c),
                          c.addressof(nXOff_c),
                          locs_a_p)


            succ = ccat(10, argv)

            
        #self.qktime_loop += (time.time() - tc)
        #tc = time.time()

        #locs_a = np.zeros(len(a), dtype = int)-1
        #ind_nz = np.nonzero(locs_a_tmp >= 0)[0]

        #locs_a[sorter_a[ind_nz]] = locs_a_tmp[ind_nz]

        self.qktime_ccat += (time.time() - tc)
                
        return locs_a_tmp


    def _katamaran_search(self, a, aIsSymmetric = False, aIsSorted = False):

        """
        Perform an accelerated katamaran search to locate
        the index (if any) of the elements of a in xlist

        This assumes that the elements in a are unique, and
        that self.xlist is sorted
        """
        

        if len(a) == 0:
            return np.zeros(0, dtype = int)-1

        locs_a = np.zeros(len(a), dtype = int)-1

        if self.nX == 0:
            return locs_a
        
        tc = time.time()

        if not aIsSorted:
            sorter_a = np.argsort(a, order = ['gal_i', 'gal_j'])
        else:
            sorter_a = np.arange(len(a), dtype = int)

        
        if aIsSymmetric:
            blocksize_a = int(np.sqrt(len(a)))
            if blocksize_a*blocksize_a != len(a):
                print("Inconsistent size of a")
                set_trace()

        ind_x = ind_a = 0

        a_i = a['gal_i'][sorter_a[ind_a]]
        a_j = a['gal_j'][sorter_a[ind_a]]
        x_i = self.xlist['gal_i'][ind_x]
        x_j = self.xlist['gal_j'][ind_x]


                
        while(True):
            
            if a_i > x_i:

                # Need to check if jump would go beyond end of list
                if a_i > len(self.xOffset)-1:
                    break
                
                ind_x = self.xOffset[a_i]
                if ind_x >= self.nX:
                    break
                x_i = self.xlist['gal_i'][ind_x]
                x_j = self.xlist['gal_j'][ind_x]
                continue
          
            if a_i < x_i:
                if aIsSymmetric:
                    ind_a = (ind_a // blocksize_a + 1) * blocksize_a
                else:
                    ind_a += 1
                if ind_a >= len(a):
                    break
                a_i = a['gal_i'][sorter_a[ind_a]]
                a_j = a['gal_j'][sorter_a[ind_a]]
                continue

            # At this point, a_i == x_i, so we now check j

            if a_j == x_j:
                locs_a[sorter_a[ind_a]] = ind_x
                ind_a += 1
                ind_x += 1

                if ind_x >= self.nX or ind_a >= len(a):
                    break

                a_i = a['gal_i'][sorter_a[ind_a]]
                a_j = a['gal_j'][sorter_a[ind_a]]
                x_i = self.xlist['gal_i'][ind_x]
                x_j = self.xlist['gal_j'][ind_x]
                
                continue
                
            if a_j < x_j:
                ind_a += 1
                if ind_a >= len(a):
                    break
                a_i = a['gal_i'][sorter_a[ind_a]]
                a_j = a['gal_j'][sorter_a[ind_a]]
                continue

            if a_j > x_j:
                ind_x += 1
                if ind_x >= self.nX:
                    break
                x_i = self.xlist['gal_i'][ind_x]
                x_j = self.xlist['gal_j'][ind_x]
                continue

        self.qktime_python += (time.time() - tc)

        return locs_a


    def update(self, xlist_curr, first = None, last = None):

        """
        Update and extend the exchange list, to include exchanges in 
        last snapshot interval.
        
        This also re-orders the list by i-galaxy, and creates
        a new offset list.
        """
        stime = time.time()

        if first is None:
            first = 0
        if last is None:
            last = len(xlist_curr.xlist)-1

        subind = np.arange(first, last+1, dtype = int)

        print("   ... updating exchange totals...")

        # First check which are new, and which are old
        inds_in_x = self._katamaran_search(xlist_curr.xlist[subind])

        ind_old = np.nonzero(inds_in_x >= 0)[0]
        ind_new = np.nonzero(inds_in_x < 0)[0]

        print("   ---> {:d} updates, {:d} new exchanges..." 
              .format(len(ind_old), len(ind_new)))

        # For old ones, just add masses
        self.xlist['mass'][inds_in_x[ind_old]] += xlist_curr.xlist['mass'][subind[ind_old]]
        
        # For new ones, make new entries
        self.xlist = np.concatenate((self.xlist, np.zeros(len(ind_new), dtype = self.xdtype)))
        self.xlist[self.nX:]['gal_i'] = xlist_curr.xlist[subind[ind_new]]['gal_i']
        self.xlist[self.nX:]['gal_j'] = xlist_curr.xlist[subind[ind_new]]['gal_j']
        self.xlist[self.nX:]['mass'] = xlist_curr.xlist[subind[ind_new]]['mass']

        self.nX += len(ind_new)

        # Sort xlist
        self._reSort()

        # And build new offset list
        self._build_offset_list()

        print("   <> done updating exchange totals ({:.3f} sec.)"
              .format(time.time() - stime))

    def _build_offset_list(self, max_elem = 0):

        """
        (re-)build the offset list by i-galaxy entries.
        """

        max_i = np.max(self.xlist['gal_i'])
        numbins = np.max((max_elem, max_i))+1  # +1 because if maxgal = 10, there are 11
        self.xOffset = np.zeros(numbins+1, dtype = int) # ... and another one extra for coda
        
        # ... and here +1 is necessary to get proper behavious out of histogram
        # N.B.: final [0] only selects actual histogram, not bin edges
        histogram = np.histogram(self.xlist['gal_i'], bins = max_i+1, range = [0, max_i+1])[0]
        self.xOffset[1:1+len(histogram)] = np.cumsum(histogram)
        
        # If there are more offset_list entries than histogram entries, fill all those 
        # with the coda value
        self.xOffset[1+len(histogram):] = self.xOffset[len(histogram)]


    def query_pairs(self, pairs):
        
        """
        Retrieve exchanged mass between specified pairs
        Input must have correct field names already
        """
        
        ngal = len(pairs)
        sorter = np.argsort(pairs, order = ('gal_i', 'gal_j'))
        
        inds_in_x = self._katamaran_search(pairs[sorter], aIsSymmetric = False, aIsSorted = True)

        ind_found = np.nonzero(inds_in_x >= 0)[0]
        pairs['mass'][sorter[ind_found]] = self.xlist['mass'][inds_in_x[ind_found]]

        return

        

    def query(self, galaxies):

        """
        Retrieve all the exchanges between the galaxies named in 
        the input.
        """

        if self.nX == 0:
            return np.zeros(0, dtype = [('gal_i', 'i'), ('gal_j', 'i'), ('ind_i', 'i'), ('ind_j', 'i'), ('mass', 'd')])

        ngal = len(galaxies)

        pairs = np.zeros(ngal*ngal, dtype = [('gal_i', 'i'), ('gal_j', 'i'), ('ind_i', 'i'), ('ind_j', 'i'), ('mass', 'd')])

        sorter_gal = np.argsort(galaxies)
        
        pairs['gal_i'] = np.repeat(galaxies[sorter_gal], ngal)
        pairs['gal_j'] = np.tile(galaxies[sorter_gal], ngal)
        pairs['ind_i'] = np.repeat(sorter_gal, ngal)
        pairs['ind_j'] = np.tile(sorter_gal, ngal)
        
        ktime = time.time()
        
        if len(pairs) < 2500:
            inds_in_x = self._katamaran_search(pairs, aIsSymmetric = True, aIsSorted = True)
        else:
            inds_in_x = self._katamaran_search_ccat(pairs, aIsSorted = True)

        self.qktime += time.time()-ktime
        self.npairs += len(pairs)
        self.nruns  += 1

        ind_found = np.nonzero(inds_in_x >= 0)[0]
        pairs['mass'][ind_found] = self.xlist['mass'][inds_in_x[ind_found]]

        return pairs[ind_found]


    def find_paired_exchanges(self):

        """
        Find all exchanges in current list that are paired, i.e.
        that have both an entry i-->j and j-->i
        """

        statusPaired = np.zeros(self.nX, dtype = np.int8)
        pairCandidate = np.zeros(self.nX, dtype = self.xdtype)

        pairCandidate['gal_i'][:] = self.xlist['gal_j']
        pairCandidate['gal_j'][:] = self.xlist['gal_i']
        pairCandidate['mass'][:] = -1

        if np.count_nonzero(self.xlist['gal_i'] == self.xlist['gal_j']) > 0:
            print("Exchanges to and from same galaxy???")
            set_trace()

        ind_sel = np.nonzero(pairCandidate['gal_i'] > pairCandidate['gal_j'])[0]
        pairCandidate = pairCandidate[ind_sel]
        self.query_pairs(pairCandidate)

        ind_isPaired = np.nonzero(pairCandidate['mass'] > 0)[0]
        numPaired = len(ind_isPaired)
        
        retlist = np.zeros(numPaired*2, dtype = self.xdtype)

        ind_i = np.arange(numPaired*2, step = 2, dtype = int)
        ind_j = ind_i+1

        retlist[ind_i] = self.xlist[ind_sel[ind_isPaired]]
        retlist[ind_j] = pairCandidate[ind_isPaired]

        return retlist


class Links:

    """ 
    Class to represent a link network between two snapshots.
    """

    def __init__(self, isnap, length, norm = True, compensate = args.compensate_history, pre = "", find_permitted = True, filter_long = True, quiet = False, blockAllRevenge = True):

        """
        Initialise the link network, i.e. read the info from disk
        """

        self.quiet = quiet

        if not quiet:
            print(pre + "... reading links...", end = "")

        self._pre = pre
        self.eqtime = 0
        self.recent_time = 0
        self.rx_findTimeA = 0
        self.rx_findTimeB = 0
        self.luTime = 0
        self.isinTime = 0
        self.rx_calcTime = 0
        self.errorCheckTime = 0

        self.blockAllRevenge = blockAllRevenge

        lstime = time.time()

        h5dir = 'Level' + str(length) + '/Snapshot_' + str(isnap).zfill(3) + '/'
        subdir = st.form_files(rundir, isnap+length)
        #sh_numPart = st.eagleread(subdir, 'Subhalo/SubLength', astro = False)

        sender = yb.read_hdf5(spiderloc, h5dir + 'Sender')
        
        self.nLinks = sender.shape[0]
        
        self.isnap = isnap
        self.jsnap = isnap+length
        self.length = length
        self.links = np.zeros(self.nLinks, dtype = [('sender', 'i'), ('receiver', 'i'), ('rank', 'i'), ('coreRank', 'i'), ('choice', 'i'), ('fullChoice', 'i'), ('senderFraction', 'd'), ('coreSenderFraction', 'd'), ('receiverFraction', 'd'), ('receiverFractionN', 'd'), ('numPart', 'i8'), ('coreNumPart', 'i8'), ('mass', 'd'), ('normSF', 'd'), ('normCSF', 'd'), ('normRF', 'd'), ('normRFn', 'd'), ('sortByRecv', 'i'), ('compensatedMass', 'd'), ('compensatedChoice', 'i'), ('compensatedNormRF', 'd'), ('status', 'i1'), ('compensatedNum', 'i')])
        
        self.links['sender'] = sender
        self.links['receiver'] = yb.read_hdf5(spiderloc, h5dir + 'Receiver')
        self.links['rank'] = yb.read_hdf5(spiderloc, h5dir + 'Rank')
        self.links['coreRank'] = yb.read_hdf5(spiderloc, h5dir + 'CoreRank')
        self.links['choice'] = yb.read_hdf5(spiderloc, h5dir + 'Choice')
        self.links['fullChoice'] = yb.read_hdf5(spiderloc, h5dir + 'FullChoice')
        self.links['senderFraction'] = yb.read_hdf5(spiderloc, h5dir + 'SenderFraction')
        self.links['coreSenderFraction'] = yb.read_hdf5(spiderloc, h5dir + 'CoreSenderFraction')
        self.links['receiverFraction'] = yb.read_hdf5(spiderloc, h5dir + 'ReceiverFraction')

        if flag_computeFree:
            self.sendNTot = yb.read_hdf5(spiderloc, h5dir + 'SenderTotNum')
            self.sendNCoreTot = yb.read_hdf5(spiderloc, h5dir + 'SenderTotCoreNum')
            self.recvNTot = yb.read_hdf5(spiderloc, h5dir + 'ReceiverTotNum')
            self.recvMTot = yb.read_hdf5(spiderloc, h5dir + 'ReceiverTotMass')


        self.links['numPart'] = yb.read_hdf5(spiderloc, h5dir + 'NumPart')

        if flag_computeFree:
            self.links['receiverFractionN'] = self.links['numPart']/self.recvNTot[self.links['receiver']]
        
        self.links['coreNumPart'] = yb.read_hdf5(spiderloc, h5dir + 'CoreNumPart')
        self.links['mass'] = yb.read_hdf5(spiderloc, h5dir + 'Mass')
        self.links['sortByRecv'] = yb.read_hdf5(spiderloc, h5dir + 'SortByRecv')

        self.recvOffset = yb.read_hdf5(spiderloc, h5dir + 'ReceiverOffset')
        self.sendOffset = yb.read_hdf5(spiderloc, h5dir + 'SenderOffset')

        self.links['status'][:] = 0

        self.nHaloesA = self.sendOffset.shape[0]-1
        self.nHaloesB = self.recvOffset.shape[0]-1

        self.norm_ratios = np.zeros(self.nHaloesB*10)
        self.norm_ratio_mass = np.zeros(self.nHaloesB*10)
        self.norm_ratio_size = self.nHaloesB*10
        self.norm_ratio_counter = 0

        self.recent_fracs = np.zeros(self.nHaloesB*100)
        self.recent_frac_size = self.nHaloesB*100
        self.recent_frac_counter = 0
        self.recentMTot = 0
        self.MTot = 0

        if not quiet:
            print(" done ({:.3f} sec.)"
                  .format(time.time()-lstime))

        ppstime = time.time()

        if not quiet:
            print(pre + "... post-processing links...")

        if norm:
            self._normalise_fraction('receiver', 'mass', 'normRF')
            self._normalise_fraction('sender', 'mass', 'normSF')
            self._normalise_fraction('sender', 'coreNumPart', 'normCSF')
            self._normalise_fraction('receiver', 'numPart', 'normRFn')

        if flag_computeFree:
            self.compute_free_fraction()

        # Perform history compensation, if needed:
        if compensate:
        
            if args.exact_history:
                self._compensate_history_exact()
            else:
                self._compensate_history()
        
            if norm:
                self._normalise_fraction('receiver', 'compensatedMass', 'compensatedNormRF')
            
            if np.max(self.links['compensatedNormRF']) > (1+1e6):
                print("Problem: compensatedNormRF > 1 (max={:.5f}" .format(np.max(self.links['compensatedNormRF'])))
                set_trace()
        
        else:
            self._copy_compensated_fields()


        if not quiet:
            print("   <> done post-processing links ({:.3f} sec.)"
                  .format(time.time()-ppstime))

        # Can also directly identify permitted links, to make actual program
        # tidier
        if find_permitted:
            self.find_permitted_links()
            
        if filter_long and self.length > 1:
            self.filter_long_links()

        #print(pre + "<> finished link set-up ({:.3f} sec.)"
        #      .format(time.time()-lstime))


    # Define internal functions

    def _copy_compensated_fields(self):
        self.links['compensatedMass'] = self.links['mass']
        self.links['compensatedChoice'] = self.links['choice']
        self.links['compensatedNormRF'] = self.links['normRF']

    def _normalise_fraction(self, side, quant, output, zeroval = 0):
        
        """
        Calculate the normalised fraction of a quantity.
        zeroval indicates what value to write where total is zero.
        """

        # Compute the total of 'quant' for each subhalo
        if side == 'sender':
            nSH = self.nHaloesA
        elif side == 'receiver':
            nSH = self.nHaloesB
        else:
            print("input for side not understood.")
            set_trace()
        
        totals = np.zeros(nSH)
        for iilink in range(self.nLinks):
            totals[self.links[side][iilink]] += self.links[quant][iilink]

        # Check whether there are divide-by-zero instances:
        self.links[output][:] = zeroval
        ind_good = np.nonzero(totals[self.links[side]] > 0)[0]
        
        # Then just divide the 'quant' entry of each link by its total
        self.links[output][ind_good] = self.links[quant][ind_good]/totals[self.links[side][ind_good]]

        return


    def compute_free_fraction(self):
        
        """
        Compute the fraction of particles from the sender/receiver
        subhaloes that are not bound in the receiver/sender snapshot
        """

        self.freeSend = np.zeros(self.nHaloesA, dtype = [('All', 'd'), ('Core', 'd')])
        self.freeRecv = np.zeros(self.nHaloesB, dtype = [('Mass', 'd'), ('Num', 'd')])
        
        cumsum_ncore_send = np.zeros(self.nLinks+1, dtype = int)
        cumsum_ntot_send = np.zeros(self.nLinks+1, dtype = int)
        
        cumsum_ncore_send[1:] = np.cumsum(self.links['coreNumPart'])
        cumsum_ntot_send[1:] = np.cumsum(self.links['numPart'])
        
        nCoreTot_send = cumsum_ncore_send[self.sendOffset[1:]]-cumsum_ncore_send[self.sendOffset[:-1]]
        nTot_send = cumsum_ntot_send[self.sendOffset[1:]]-cumsum_ntot_send[self.sendOffset[:-1]]

        self.freeSend['All'] = 1-nTot_send/self.sendNTot

        ind_anycore = np.nonzero(self.sendNCoreTot > 0)[0]
        self.freeSend['Core'][ind_anycore] = 1-nCoreTot_send[ind_anycore]/self.sendNCoreTot[ind_anycore]
        self.freeSend['Core'][self.sendNCoreTot == 0] = 0

        cumsum_ntot_recv = np.zeros(self.nLinks+1, dtype = int)
        cumsum_mtot_recv = np.zeros(self.nLinks+1)

        cumsum_ntot_recv[1:] = np.cumsum(self.links['numPart'][self.links['sortByRecv']])
        cumsum_mtot_recv[1:] = np.cumsum(self.links['mass'][self.links['sortByRecv']])

        nTot_recv = cumsum_ntot_recv[self.recvOffset[1:]]-cumsum_ntot_recv[self.recvOffset[:-1]] 
        mTot_recv = cumsum_mtot_recv[self.recvOffset[1:]]-cumsum_mtot_recv[self.recvOffset[:-1]]  

        self.freeRecv['Num'] = 1 - nTot_recv/self.recvNTot
        self.freeRecv['Mass'] = 1 - mTot_recv/self.recvMTot


    def _compensate_history_exact(self):
        
        """
        EXPERIMENTAL version of history compensation that works 
        with an explicity particle-level table.
    
        Beginning is identical to 'standard' version below.
        """

        # Initialise compensated mass/choice to original values
        self.links['compensatedMass'] = self.links['mass']
        self.links['compensatedChoice'] = self.links['choice']
        
        # Cannot compensate at first snapshot...
        if self.isnap == 0:
            return

        print(self._pre + "   ... start exact history compensation...")

        # Load galaxy table at sender snapshot
        galaxies_at_send = snaps[self.isnap].local['Galaxy']

        # If this is a *short-link*, then we need to identify 'restricted' subhaloes 
        # that may only send compensation, but not receive it.
        # See notebook, 19-Jan-18, point 4
        
        compStatusMask = np.zeros(self.nLinks, dtype = int) # 0 indicates send+receive, =default

        if self.length == 1:
            ind_sendCompOnly = np.nonzero((self.links['coreRank'] > 0) & 
                                          (self.links['coreNumPart'] < sim_frac * self.links['coreNumPart'][self.sendOffset[self.links['sender']]]))[0]
            
            compStatusMask[ind_sendCompOnly] = 1
            
            print(self._pre + "      ---> ({:d}/{:d} (={:.2f}%) links cannot receive compensation)" 
                  .format(len(ind_sendCompOnly), self.nLinks, len(ind_sendCompOnly)/self.nLinks*100))

        # Now loop through each B-subhalo in turn...

        #print(self._pre + '      ', end = '', flush = True)
        dotgap = next_dot = self.nHaloesB/50

        counter_links = self.recvOffset[1:]-self.recvOffset[:-1]
        ind_toCheck = np.nonzero(counter_links >= 2)[0]

        for irecv in ind_toCheck:   # irecv is a RECEIVER SUBHALO

            print("Correcting irecv={:d}" .format(irecv))

            if irecv > next_dot:
                #print(".", end = '', flush = True) 
                next_dot += dotgap
                
            ind_first = self.recvOffset[irecv]
            ind_beyondlast = self.recvOffset[irecv+1]
            n_curr_links = ind_beyondlast-ind_first

            curr_links = self.links['sortByRecv'][ind_first:ind_beyondlast]
            curr_sender = self.links['sender'][curr_links]
            curr_galaxies = galaxies_at_send[curr_sender] # Gal. num. of sender

            
            """
            # Two more safety checks: make sure all these links come from different
            # senders and have unique galaxy numbers
            if len(np.unique(curr_sender)) != len(curr_sender):
                print("Double sender??")
                set_trace()

            if len(np.unique(curr_galaxies)) != len(curr_galaxies):
                print("Double galaxies???")
                set_trace()
            """


        
            # Setup done.

            # --------------------------------------------
            # From here on this version differs from below
            # --------------------------------------------

            # Find indices in PSS for current subhalo

            if args.exact_limited:
                pss_inds = np.nonzero((pss.table[:, self.isnap+self.length] == irecv) & (np.isin(pss.table[:, self.isnap], curr_sender)))[0]    
            else:
                pss_inds = np.nonzero(pss.table[:, self.isnap+self.length] == irecv)[0]

            if len(pss_inds) == 0:
                print("?? No particles ??")
                set_trace()

            corrected_prog = np.zeros(len(pss_inds), dtype = int)-1

            # Loop through previous snapshots
            for iorig in range(isnap):
                
                ind_notyetfound = np.nonzero(corrected_prog < 0)[0]
                if len(ind_notyetfound) == 0:
                    break
                
                shi_o = pss.table[pss_inds[ind_notyetfound], iorig]
                ind_bound_o = np.nonzero(shi_o >= 0)[0]

                gal_o = snaps[iorig].local['Galaxy'][shi_o[ind_bound_o]]
                in_targ = np.isin(gal_o, curr_galaxies)
                ind_targ = np.nonzero(in_targ)[0]

                for igal, targgal in enumerate(curr_galaxies):
                    ind_this = np.nonzero(gal_o[ind_targ] == targgal)[0]
                    corrected_prog[ind_notyetfound[ind_bound_o[ind_targ[ind_this]]]] = igal
            
            # Sanity check: ALL particles should be allocated at the end
            if args.exact_limited and corrected_prog.min() < 0:
                print("Why are some particles never in their target galaxy??")
                set_trace()

            # Now collect results:
            
            for ilink in range(len(curr_galaxies)):
                ind_thislink = np.nonzero(corrected_prog == ilink)[0]
                self.links['compensatedNum'][ilink] = len(ind_thislink)
                self.links['compensatedMass'][ilink] = np.sum(pss.mass[pss_inds[ind_thislink]])
                    
            # Final step: we need to convert this into a new compensatedChoice
            # Also need to take core particle order into account!

            ind_goodcore = np.nonzero(self.links['coreNumPart'][curr_links] >= 3)[0]
            ind_badcore = np.nonzero(self.links['coreNumPart'][curr_links] < 3)[0]
            
            # N.B.: minus sign is there to sort in descending order
            sorter_good = np.argsort(-self.links['compensatedMass'][curr_links[ind_goodcore]])
            self.links['compensatedChoice'][curr_links[ind_goodcore[sorter_good]]] = np.arange(len(ind_goodcore), dtype = int) 

            sorter_bad = np.argsort(-self.links['compensatedMass'][curr_links[ind_badcore]])
            self.links['compensatedChoice'][curr_links[ind_badcore[sorter_bad]]] = np.arange(len(ind_badcore), dtype = int)+len(ind_goodcore) 

    

    def _compensate_history(self):

        eeh_time = 0
        hstime = time.time()

        # Initialise compensated mass/choice to original values
        self.links['compensatedMass'] = self.links['mass']
        self.links['compensatedChoice'] = self.links['choice']
        
        # Cannot compensate at first snapshot...
        if self.isnap == 0:
            return

        print(self._pre + "   ... start history compensation...")

        # Load galaxy table at sender snapshot
        galaxies_at_send = snaps[self.isnap].local['Galaxy']

        # If this is a *short-link*, then we need to identify 'restricted' 
        # subhaloes that may only send compensation, but not receive it.
        # See notebook, 19-Jan-18, point 4
        
        compStatusMask = np.zeros(self.nLinks, dtype = int) 
        # (0 indicates send+receive, =default)

        if self.length == 1:
            ind_sendCompOnly = np.nonzero((self.links['coreRank'] > 0) & 
                                          (self.links['coreNumPart'] < sim_frac * self.links['coreNumPart'][self.sendOffset[self.links['sender']]]))[0]
            
            compStatusMask[ind_sendCompOnly] = 1
            
            print(self._pre + "      ---> ({:d}/{:d} (={:.2f}%) links cannot receive compensation [{:.3f} sec.])" 
                  .format(len(ind_sendCompOnly), self.nLinks, len(ind_sendCompOnly)/self.nLinks*100, time.time()-hstime))

        # Safety check to make sure each link gets compensated only once...
        # [[ only during testing ]]
        #checked_mask = np.zeros(self.nLinks, dtype = int)
        
        if args.dual_history:
            
            lll_stime = time.time()
            # Need to load the (basic) long-link info
            longLinks = []

            #            longLinks.append([]) # '0'
            #            longLinks.append([]) # '1'

            # N.B.: iorig here is STARTING snap of the long link, go backwards
            for iorig in range(self.isnap+self.length, self.isnap+self.length-args.lim_recent-1, -1):
                if iorig < 0:
                    break
                
                if iorig >= self.isnap:
                    longLinks.append([])
                else:
                    longLinks.append(Links(iorig, self.isnap-iorig+self.length, norm = False, compensate = False, find_permitted = False, filter_long = False, quiet = True, blockAllRevenge = False))
            
            print(self._pre + "     ---> loaded long-link info, took {:.3f} sec."
                  .format(time.time()-lll_stime))


        # Now loop through each B-subhalo in turn and compensate its links

        exchange.qktime = 0
        exchange.qktime_python = 0
        exchange.qktime_ccat = 0
        exchange.qktime_sort = 0
        exchange.npairs = 0
        exchange.nruns = 0
        uniquetime = 0
        rescaletime = 0
        choicetime = 0
        setuptime = 0
        dctime = 0
        seltime = 0
        sumtime = 0
        hm_time = 0
        num_skipped = 0


        # Set up progress indicator:
        #print(self._pre + '      ', end = '', flush = True)
        dotgap = next_dot = self.nHaloesB/50

        # Identify which (receiver) subhaloes actually need checking:
        # only those that receive more than one link
        counter_links = self.recvOffset[1:]-self.recvOffset[:-1]
        ind_toCheck = np.nonzero(counter_links >= 2)[0]

        lstime = time.time()
        setuptime += (lstime - hstime)
        print(self._pre + "     ---> setup done, took {:.3f} sec."
              .format(setuptime))

        if args.quickHistory:
            
            # --- New bit added 11-May-18 ---
        
            # Pre-load all sender galaxies, 
            # to see which receivers actually need to be dealt with.

            if args.quickHistoryLight:

                nCombiTotal = np.sum(counter_links[ind_toCheck])
        
                fullSendGalList = np.zeros(nCombiTotal, dtype = np.int32)
                fullMatchList = np.zeros(nCombiTotal, dtype = np.int8)
                fullSendOffset = np.zeros(len(ind_toCheck)+1, dtype = int)

                currOffset = 0
                for iirecv, irecv in enumerate(ind_toCheck):

                    ind_first = self.recvOffset[irecv]
                    ind_beyondlast = self.recvOffset[irecv+1]
                    n_curr_links = ind_beyondlast-ind_first
                    
                    # Extract all links going to current receiver subhalo
                    curr_links = self.links['sortByRecv'][ind_first:ind_beyondlast]
                    curr_sender = self.links['sender'][curr_links]
                    curr_galaxies = galaxies_at_send[curr_sender] # Gal. num. of sender
                    fullSendGalList[currOffset:currOffset+n_curr_links] = curr_galaxies
                    
                    currOffset += n_curr_links
                    fullSendOffset[iirecv+1] = currOffset 
                
                # Now check which are in fullExchange.xlist['gal_i']:
                iistime = time.time()
                subind_inExchange = np.nonzero(np.isin(fullSendGalList, fullExchange.xlist['gal_i']))[0]
                #print("--> isin pre-check took {:.3f} sec." .format(time.time()-iistime))
                #set_trace()
            else:    
                
                nCombiTotal = np.sum(counter_links[ind_toCheck]**2)
        
                dtype_combi = [('gal_i', 'i4'), ('gal_j', 'i4')]
                fullSendGalList = np.zeros(nCombiTotal, dtype = dtype_combi)
                fullMatchList = np.zeros(nCombiTotal, dtype = np.int8)
                fullSendOffset = np.zeros(len(ind_toCheck)+1, dtype = int)

                xclist = np.zeros(len(fullExchange.xlist['gal_i']), dtype = dtype_combi)
                xclist['gal_i'] = fullExchange.xlist['gal_i']
                xclist['gal_j'] = fullExchange.xlist['gal_j']

                currOffset = 0
                for iirecv, irecv in enumerate(ind_toCheck):

                    ind_first = self.recvOffset[irecv]
                    ind_beyondlast = self.recvOffset[irecv+1]
                    n_curr_links = ind_beyondlast-ind_first
                
                    # Extract all links going to current receiver subhalo
                    curr_links = self.links['sortByRecv'][ind_first:ind_beyondlast]
                    curr_sender = self.links['sender'][curr_links]
                    curr_galaxies = galaxies_at_send[curr_sender] # Gal. num. of sender
                    curr_gal_i = np.repeat(curr_galaxies, n_curr_links)
                    curr_gal_j = np.tile(curr_galaxies, n_curr_links)
                
                    fullSendGalList[currOffset:currOffset+n_curr_links**2]['gal_i'] = curr_gal_i #curr_galaxies
                    fullSendGalList[currOffset:currOffset+n_curr_links**2]['gal_j'] = curr_gal_j #curr_galaxies
              
                    currOffset += n_curr_links**2
                    fullSendOffset[iirecv+1] = currOffset 
                
                # Now check which are in fullExchange.xlist['gal_i']:
                iistime = time.time()
                subind_inExchange = np.nonzero(np.isin(fullSendGalList, xclist))[0]  #fullExchange.xlist['gal_i']))[0]
                #print("--> isin pre-check took {:.3f} sec." .format(time.time()-iistime))

            fullMatchList[subind_inExchange] = 1
            
            matchCumSum = np.cumsum(fullMatchList)
            matchCumSum = np.concatenate(([0], matchCumSum))
            nMatchPerToCheck = matchCumSum[fullSendOffset[1:]]-matchCumSum[fullSendOffset[:-1]]
            
            subind_reallyToCheck = np.nonzero(nMatchPerToCheck > 0)[0]
            n_reallyToCheck = len(subind_reallyToCheck)
            print(self._pre + "      --> QuickHistory check result: retain {:d}/{:d} galaxies (={:.2f}%; {:.3f} sec.)" .format(n_reallyToCheck, len(ind_toCheck), n_reallyToCheck/len(ind_toCheck)*100, time.time()-lstime)) 
            ind_toCheck = ind_toCheck[subind_reallyToCheck]


        # *******************************************
        # **** MAIN LOOP OVER RECEIVER SUBHALOES ****
        # *******************************************

        eeh_times = np.zeros(len(ind_toCheck))
        hm_status = np.zeros(len(ind_toCheck), dtype = np.int8)
        print(self._pre + '      ', end = '', flush = True)
        
        for iirecv, irecv in enumerate(ind_toCheck):

            # Progress indicator:
            if irecv > next_dot:
                print(".", end = '', flush = True) 
                next_dot += dotgap
                
            ind_first = self.recvOffset[irecv]
            ind_beyondlast = self.recvOffset[irecv+1]
            n_curr_links = ind_beyondlast-ind_first

            # Check for double-compensation first, in case there are *any* 
            # links to this subhalo
            # [[ only during testing ]]

            """
            if ind_beyondlast > ind_first:
                if np.sum(checked_mask[self.links['sortByRecv'][ind_first:ind_beyondlast]]) > 0:
                    print("Seems that we are double-compensating the same link...?")
                    set_trace()

                checked_mask[self.links['sortByRecv'][ind_first:ind_beyondlast]] = 1
            """

            # Extract all links going to current receiver subhalo
            curr_links = self.links['sortByRecv'][ind_first:ind_beyondlast]
            curr_sender = self.links['sender'][curr_links]
            curr_galaxies = galaxies_at_send[curr_sender] # Gal. num. of sender

            ec = time.time()
            seltime += (ec - lstime)
            
            """
            # Two more safety checks: make sure all these links come from different
            # senders and have unique galaxy numbers
            if len(np.unique(curr_sender)) != len(curr_sender):
                print("Double sender??")
                set_trace()

            if len(np.unique(curr_galaxies)) != len(curr_galaxies):
                print("Double galaxies???")
                set_trace()
            """

            fc = time.time()
            uniquetime += (fc-ec)
            
            # Setup done. We now need to work out the past mass exchanges between
            # these galaxies

            if args.dual_history:
                history_matrix = self._extract_dual_exchange_history(irecv, curr_galaxies, self.links['mass'][self.links['sortByRecv'][self.recvOffset[irecv]:self.recvOffset[irecv+1]]], longLinks, curr_links)
            else:
                history_matrix = self._extract_exchange_history(curr_galaxies) 

            gc = time.time()
            eeh_time += (gc - fc)
            eeh_times[iirecv] = (gc-fc)

            # In the history_matrix array, element [i,j] says how much mass 
            # gal i HAS GIVEN (already) to gal j, so how much j should give 
            # back to i. 
            #
            # To find the total that j must return (to all i's), we must 
            # therefore sum over axis 0. To find the total that i gets
            # back (from all other galaxies), we sum over axis 1.

            # Need to set exchanges to links which are blocked from getting
            # compensation to zero. From above, this means zeroing all 
            # elements corresponding to individual i-entries (=receivers).

            history_matrix[compStatusMask[curr_links] == 1, :] = 0

            hm_stime = time.time()
            hm_tot = np.sum(history_matrix)
            hm_time += (time.time()-hm_stime)

            if hm_tot == 0:
                num_skipped += 1
                hm_status[iirecv] = 1
                continue
 
            total_toSend = np.sum(history_matrix, axis = 0)

            # Need to make sure that a galaxy does not give more compensation 
            # than its entire link mass to current subhalo.
            # If this happens, re-scale all its compensations so total is 
            # equal to its link mass.

            ind_over_request = np.nonzero(total_toSend > self.links[curr_links]['mass'])[0]
            if len(ind_over_request) > 0:
                rescale = self.links[curr_links[ind_over_request]]['mass']/total_toSend[ind_over_request]
            
                # This can affect multiple compensations to the same galaxy, so 
                # need to do this in a loop
                for irsc, curr_scale in enumerate(rescale):
                    history_matrix[:, ind_over_request[irsc]] *= curr_scale

                # ... and now need to re-compute send-total after re-scaling
                total_toSend = np.sum(history_matrix, axis = 0)

            hc = time.time()
            rescaletime += (hc - gc)


            total_toReceive = np.sum(history_matrix, axis = 1)

            self.links['compensatedMass'][curr_links] = self.links['mass'][curr_links] - total_toSend + total_toReceive

            # Make double-sure this does not lead to negative masses anywhere:
            if np.min(self.links['compensatedMass'][curr_links]) < -1e-6:
                print("negative compensated mass??")
                set_trace()

            ic = time.time()
            sumtime += (ic - gc)
                
            # Final step: we need to convert this into a new compensatedChoice
            # Also need to take core particle order into account!

            binind_zero = (self.links['coreNumPart'][curr_links] < 3)
            ind_goodcore = np.arange(len(curr_links))[~binind_zero]
            ind_badcore = np.arange(len(curr_links))[binind_zero]

            #ind_goodcore = np.nonzero(self.links['coreNumPart'][curr_links] >= 3)[0]
            #ind_badcore = np.nonzero(self.links['coreNumPart'][curr_links] < 3)[0]
            
            # N.B.: minus sign is there to sort in descending order
            sorter_good = np.argsort(-self.links['compensatedMass'][curr_links[ind_goodcore]])
            self.links['compensatedChoice'][curr_links[ind_goodcore[sorter_good]]] = np.arange(len(ind_goodcore), dtype = int) 

            sorter_bad = np.argsort(-self.links['compensatedMass'][curr_links[ind_badcore]])
            self.links['compensatedChoice'][curr_links[ind_badcore[sorter_bad]]] = np.arange(len(ind_badcore), dtype = int)+len(ind_goodcore) 

            lstime = time.time()
            choicetime += (lstime - ic)

        # Ends loop over individual B-subhaloes
        print("")

        etime = time.time()
        
        # Almost done, just output some statistics:
        ind_varied = np.nonzero(self.links['compensatedChoice'] != self.links['choice'])[0]
        n_changed = len(ind_varied)
        
        print(self._pre + "      ---> {:d}/{:d} links have changed choice (={:.2f}%)"
              .format(n_changed, self.nLinks, n_changed/self.nLinks*100))

        recentFracs = self.recent_fracs[:self.recent_frac_counter] 
        
        if len(recentFracs) > 0:
            print(self._pre + "      ---> median recent exchange fraction is {:.2f} ({:.2f} - {:.2f})"
                  .format(*np.percentile(recentFracs, [50, 15, 85])))
            print(self._pre + "            (mass-weighted average = {:.3f})"
                  .format(self.recentMTot/self.MTot))
            
        else:
            print(self._pre + " --------------  WARNING  --------------")
            print(self._pre + "      ---> found no exchanges at all (?)")
            print(self._pre + " --------------  WARNING  --------------")

        normRatios = self.norm_ratios[:self.norm_ratio_counter]
        normRatioMasses = self.norm_ratio_mass[:self.norm_ratio_counter]
        
        if len(normRatios) > 0:
            print(self._pre + "      ---> {:.2f}% of corrections below 0.1 ({:.2f}% by mass)"
              .format(np.count_nonzero(normRatios < 0.1)/self.norm_ratio_counter*100, np.sum(normRatioMasses[normRatios < 0.1])/np.sum(normRatioMasses)))
            print(self._pre + "      ---> median of remaining is {:.2f} ({:.2f} - {:.2f})"
              .format(*np.percentile(normRatios[normRatios >= 0.1], [50, 15, 85])))
        else:
            if self.length < args.lim_recent:
                print(self._pre + " --------------  WARNING  --------------")
                print(self._pre + "      ---> found no (recent) exchanges  ")
                print(self._pre + " --------------  WARNING  --------------")

        print(self._pre + "   <> finished history compensation ({:.3f} sec.) <>" 
              .format(etime-hstime))


        print(self._pre + "      (recent-exchanges     = {:.3f} sec. / {:.2f}%)"
              .format(self.recent_time, self.recent_time/(etime-hstime)*100))
        print(self._pre + "      (old-exchanges        = {:.3f} sec. / {:.2f}%)"
              .format(self.eqtime, self.eqtime/(etime-hstime)*100))


        #if self.isnap == 4:
        #    set_trace()
        
        """
        print(self._pre + "      (query-kat-python     = {:.3f} sec.)"
              .format(exchange.qktime_python))
        print(self._pre + "      (query-kat-ccat       = {:.3f} sec.)"
              .format(exchange.qktime_ccat))
        #print(self._pre + "      (query-kat-sort      = {:.3f} sec.)"
        #      .format(exchange.qktime_sort))

        print(self._pre + "      (query-kat-other      = {:.3f} sec. / Npairs = {:d} / NKat = {:d})"
              .format(exchange.qktime-exchange.qktime_python-exchange.qktime_ccat, exchange.npairs, exchange.nruns))
        print(self._pre + "      (query-other          = {:.3f} sec.)"
              .format(self.eqtime - exchange.qktime))
        print(self._pre + "      (_extract-other       = {:.3f} sec.)"
              .format(eeh_time - self.eqtime))
        print(self._pre + "      (other                = {:.3f} sec.)"
              .format(etime-hstime-eeh_time))
        #print(self._pre + "            (setup          = {:.3f} sec.)"
        #      .format(setuptime))

        #print(self._pre + "            (double-test    = {:.3f} sec.)"
        #      .format(dctime))
        #print(self._pre + "            (unique-test    = {:.3f} sec.)"
        #      .format(uniquetime))
        #print(self._pre + "            (selection      = {:.3f} sec.)"
        #      .format(seltime))
        print(self._pre + "            (rescaling      = {:.3f} sec.)"
              .format(rescaletime))
        print(self._pre + "            (summing        = {:.3f} sec.)"
              .format(sumtime-rescaletime))
        print(self._pre + "            (choicing       = {:.3f} sec.)"
              .format(choicetime))
        print(self._pre + "            (rest           = {:.3f} sec.)"
              .format(etime-hstime-eeh_time-choicetime-sumtime))
        
        #print(self._pre + "        (after setup        = {:.3f} sec.)"
        #      .format(etime-lstime))
        """

    def _extract_exchange_history(self, curr_galaxies):
    
        """
        Function to actually compute the integrated exchange history between 
        a set of galaxies ('curr_galaxies') up to a specified snapshot
        ('target_snap') from the (pre-computed) exchange lists.
    
        This is now quite simple, because we keep a running total of
        exchanges to speed up the code.
        """

        n_galaxies = len(curr_galaxies)
        history_matrix = np.zeros((n_galaxies, n_galaxies))
        
        qtime = time.time()

        # .query() now creates a sorter internally
        exchangePairs = exchange.query(curr_galaxies)
        self.eqtime += (time.time() - qtime)

        #if self.isnap == 2 and self.length == 2: 
        #    set_trace()

        history_matrix[exchangePairs['ind_i'], exchangePairs['ind_j']] = exchangePairs['mass']

        # And that's almost it.
    
        # Sanity check: there should be no on-diagonal elements
        if np.trace(history_matrix) > 0:
            print("Problem - history matrix has on-diagonal elements...!?")
            set_trace()

        # Balance the history matrix (i.e. subtract exchanges
        # from i->j from those j->i)
        # The second step removes all the negative entries, since the matrix
        # is now anti-symmetric

        diff_matrix = history_matrix - np.transpose(history_matrix)
        history_matrix = np.clip(diff_matrix, 0, None)
            
        return history_matrix


    def _compute_recent_exchanges(self, curr_recv, curr_galaxies, history_matrix, linkmass, longLinks):

        """
        Extract exchanges for a few recent snapshots from 
        detailed exchange list.
        """

        min_m_self = linkmass

        # N.B.: Have to stop already in time that the snapshot at the start
        # of the furthest-back interval is still connected by a long-link
        # to current (j) snap
        for iorig in range(self.isnap, self.isnap+self.length-args.lim_recent, -1):
            if iorig < 1:
                break

            tta = time.time()
            xc = xlist[iorig]
      
            max_xc = len(xc.offset)-2  # max gal in list
            ind_curr_pre = np.nonzero(curr_galaxies <= max_xc)[0]
            subind_curr_inX = np.nonzero(xc.offset[curr_galaxies[ind_curr_pre]+1] > xc.offset[curr_galaxies[ind_curr_pre]])[0]
            if len(subind_curr_inX) == 0: continue
            
            ind_inX = ind_curr_pre[subind_curr_inX]

            # Set up comparison long-links: only care about those to 
            # current receiver (in j)
            
            ll = longLinks[self.isnap-iorig+1+self.length]

            if curr_recv+1 >= len(ll.recvOffset):
                set_trace()

            ind_ll = ll.links['sortByRecv'][ll.recvOffset[curr_recv]:ll.recvOffset[curr_recv+1]]
            # Find gal of sender - note that this is snap iorig-1!
            send_gal_ll = snaps[iorig-1].local['Galaxy'][ll.links['sender'][ind_ll]]

            self.rx_findTimeA += (time.time()-tta)
            
            # We now only loop through those galaxies that actually 
            # have at least one exchange in current list:

            curr_gal_to_process = curr_galaxies[ind_inX]
            sh_ii_all = snaps[iorig].revGal[curr_gal_to_process]

            for ind_ii, ii in enumerate(curr_gal_to_process):

                ttb = time.time()
                sh_ii = sh_ii_all[ind_ii]

                # Galaxy might have been skipped in current orig snapshot
                if sh_ii >= 0:
                    if snaps[iorig].reverse['LinkMassUncomp'][sh_ii] < min_m_self[ind_inX[ind_ii]]:
                        min_m_self[ind_inX[ind_ii]] = snaps[iorig].reverse['LinkMassUncomp'][sh_ii]

                xind = xc.offset[ii]+np.arange(xc.offset[ii+1]-xc.offset[ii], dtype = int)
                cstime = time.time()
                if np.count_nonzero(xc.xlist['gal_i'][xind] != ii) > 0:
                    print("Error with offsets in xlist!")
                    set_trace()
                self.errorCheckTime += (time.time()-cstime)

                jj = xc.xlist['gal_j'][xind]
                ii_stime = time.time()
                ind_useful = np.nonzero(np.isin(jj, curr_galaxies, assume_unique = True))[0]
                self.isinTime += (time.time()-ii_stime)
                if len(ind_useful) == 0: continue

                # ind_useful now has all the exchanges we care about
                # Need to reconstruct which entry in input galaxy list these
                # match galaxies correspond to
                
                lu_stime = time.time()
                galind_jj_useful = [np.nonzero(curr_galaxies == this_jj)[0][0] for this_jj in jj[ind_useful]]
                self.luTime += time.time()-lu_stime
                    
                ttc = time.time()
                self.rx_findTimeB += (ttc-ttb)
                
                m_useful = xc.xlist['mass'][xind[ind_useful]]
                sum_mx = np.sum(m_useful)

                # Now we need to compare this against the long-link
                # from current galaxy (ii) in iorig to curr_recv in j.
                ind_this_ll = np.nonzero(send_gal_ll == ii)[0]
                if len(ind_this_ll) == 0:
                    continue

                if len(ind_this_ll) > 1:
                    print("Why is there more than one long-link between two galaxies??")
                    set_trace()
                    
                # Have the one LL we care about - need its mass
                m_ll = ll.links['mass'][ind_ll[ind_this_ll[0]]]

                norm_ratio = (m_ll - min_m_self[ind_inX[ind_ii]])/sum_mx 
                if norm_ratio < 0:
                    norm_ratio = 0

                if norm_ratio > 1:
                    norm_ratio = 1

                if self.norm_ratio_counter+1 >= self.norm_ratio_size:
                    self.norm_ratios = np.concatenate((self.norm_ratios, np.zeros(self.norm_ratio_size)))
                    self.norm_ratio_mass = np.concatenate((self.norm_ratio_mass, np.zeros(self.norm_ratio_size)))
                    self.norm_ratio_size += self.norm_ratio_size

                self.norm_ratios[self.norm_ratio_counter] = norm_ratio
                self.norm_ratio_mass[self.norm_ratio_counter] = sum_mx
                self.norm_ratio_counter += 1

                history_matrix[ind_inX[ind_ii], galind_jj_useful] += m_useful*norm_ratio

                self.rx_calcTime += (time.time()-ttc)

    def _extract_dual_exchange_history(self, curr_recv, curr_galaxies, linkmass, longLinks, currLinksToRecv):
    
        """
        Function to actually compute the integrated exchange history between 
        a set of galaxies ('curr_galaxies') up to a specified snapshot
        ('target_snap').
    
        In CONTRAST to the 'simple' procedure (function above), this one
        uses the exchange history only for 'old' exchanges, and extracts
        'recent' ones directly from the exchange list, checking against
        the relevant long links.

        Started 8-Feb-2018
        Adapted 28-Apr-2018 to weight 'old' exchanges by fraction given
             from sender to receiver.
        """

        n_galaxies = len(curr_galaxies)
        history_matrix = np.zeros((n_galaxies, n_galaxies))

        # First compute recent exchanges, but load that into 'total' matrix
        # That way we can later just add the 'old' exchanges to it.

        re_stime = time.time()
        self._compute_recent_exchanges(curr_recv, curr_galaxies, history_matrix, linkmass, longLinks)
        self.recent_time += (time.time()-re_stime)

        history_matrix_recent = np.copy(history_matrix)

        # Recent now contains the recent exchanges, already tested by
        # long-link confirmation.
        # We now need to add old exchanges to full list. 

        qtime = time.time()
        # .query() now creates a sorter internally
        exchangePairs = exchange.query(curr_galaxies)
        self.eqtime += (time.time() - qtime)

        #if self.isnap == 2 and self.length == 2: 
        #    set_trace()

        if args.mix_history and len(exchangePairs['ind_i']) > 0:
            
            # New bit (28-Apr-2018):  
            # 
            # Multiply (=reduce) the old exchange mass by the sender mass
            # fraction, i.e. the mass fraction of the sender that goes
            # to the receiver.
            # 
            # ind_j denotes the (prior) exchange receiver, i.e. the galaxy
            # that needs to 'return' some mass. If this one does not actually
            # donate all its mass to curr_recv, this return is reduced
            # accordingly.

            sendMassFrac = self.links['normSF'][currLinksToRecv] 

            history_matrix[exchangePairs['ind_i'], exchangePairs['ind_j']] += exchangePairs['mass'] * sendMassFrac[exchangePairs['ind_j']]
                          
        else:
            # 'Old'/simple method: just add all prior mass exchanges.
            history_matrix[exchangePairs['ind_i'], exchangePairs['ind_j']] += exchangePairs['mass']

        ind_all_nonzero = np.nonzero(history_matrix > 0)

        if len(ind_all_nonzero[0]) > 0:


            recent_frac = history_matrix_recent[ind_all_nonzero]/history_matrix[ind_all_nonzero]

            if self.recent_frac_counter + len(ind_all_nonzero[0]) >= self.recent_frac_size:
                print("Update recent_fracs...")
                self.recent_fracs = np.concatenate((self.recent_fracs, np.zeros(self.recent_frac_size)))
                self.recent_frac_size += self.recent_frac_size

            self.recent_fracs[self.recent_frac_counter:self.recent_frac_counter + len(ind_all_nonzero[0])] = recent_frac
            self.recent_frac_counter += len(ind_all_nonzero[0])
            self.recentMTot += np.sum(history_matrix_recent)
            self.MTot += np.sum(history_matrix)

        # And that's almost it.
    
        # Sanity check: there should be no on-diagonal elements
        if np.trace(history_matrix) > 0:
            print("Problem - history matrix has on-diagonal elements...!?")
            set_trace()

        # Balance the history matrix (i.e. subtract exchanges
        # from i->j from those j->i)
        # The second step removes all the negative entries, since the matrix
        # is now anti-symmetric

        diff_matrix = history_matrix - np.transpose(history_matrix)
        history_matrix = np.clip(diff_matrix, 0, None)
            
        return history_matrix


    def filter_long_links(self):

        """
        Return the indices of 'permitted' long-links.

        I.e. those that actually connect a faded galaxy
        to its potential re-appearance.

        Substantially re-written 8/9 Feb 2018
        """

        # Helper array to list senders that are allowed
        self.sender_mask = np.zeros(self.nHaloesA, dtype = np.int8)

        # --------------------------------------------
        # -- Case-A: senders that are (still) faded --
        # --------------------------------------------

        ind_still_faded = np.nonzero(snaps[self.isnap].forward['Length'] == 0)[0]
        self.sender_mask[ind_still_faded] = 1

        
        # --------------------------------------------------------
        # --- Case-B: senders whose progenitor is faded
        # --- (these follow the same rules, with some extra ones)
        # --------------------------------------------------------

        if args.allow_orphans:

            # Find all subhaloes in following snap that are unmatched
            # N.B. 1: includes cases where a match from a shorter long-link
            # has been made in current round
            # N.B. 2: also require these to have progenitors in self.isnap!

            ind_still_faded_in_next = np.nonzero((snaps[self.isnap+1].forward['Length'] == 0) & (snaps[self.isnap+1].reverse['Length'] == 1))[0]

            sh_faded_in_this = snaps[self.isnap+1].reverse['SHI'][ind_still_faded_in_next]
            # Safety check:
            if np.count_nonzero(sh_faded_in_this < 0) > 0:
                print("Why do we have non-existent irrlicht progenitors?")
                set_trace()

            ind_alive_in_this = np.nonzero(sh_faded_in_this >= 0)[0]

            self.sender_mask[sh_faded_in_this] = 2
            
        # -------------------------------------------------------------------
        # Now select the actual permissible links. Criteria are:
        # 1) (almost) faded sender (sender_mask[sender] > 0)   AND
        # 2) at least 3 core particles                         AND
        # 3) (compensated) mass higher than that of best current matched link
        #    (of shorter length), if there is any              AND
        # 4) not going to any descendant of the subhaloes that the sender
        #    sends (short-)links to [if args.prevent_revenge]  AND
        # 4b) [for Case-B]: higher coreRank than (possible) tentative
        #    match for immediate descendant                    AND
        # 5) Either the highest coreRank link satisfying these criteria, or
        #    within 2/3 of its coreNumPart
        # -------------------------------------------------------------------

        # Selection is done on a PER-SENDER basis
        nlinks_halo = self.sendOffset[1:] - self.sendOffset[:-1]

        # First cut: select senders that MAY be long-linked [--> 1)]
        senders = np.nonzero((self.sender_mask >= 1) &
                             (nlinks_halo > 0))[0]
        
        # "Result" is a per-link mask 
        permitted_mask = np.zeros(self.nLinks, dtype = np.int8)

        # Now loop through senders to find the long-link candidate(s),
        # if there are any
        for isend in senders:

            # Make a list of ALL (long) links sent by current sender...
            links_curr = self.sendOffset[isend]+np.arange(nlinks_halo[isend], dtype = int)
            
                                              # ... and see which ones are feasible [--> 2) and 3)]
            # (N.B.: ind_goodlinks is implicitly sorted by coreRank)
            
            # PREVIOUS version:
            #ind_goodlinks = np.nonzero((self.links[links_curr]['coreNumPart'] >= 3) & (self.links['compensatedMass'][links_curr] > snaps[self.isnap+self.length].reverse['LinkMass'][self.links['receiver'][links_curr]]))[0]

            # NEW version (14-May-2018):
            ind_goodlinks = np.nonzero((self.links[links_curr]['coreNumPart'] >= 3) & (self.links['compensatedMass'][links_curr] > snaps[self.isnap+self.length].local['LongLinkMinMass'][self.links['receiver'][links_curr]]))[0]
            
            if len(ind_goodlinks) == 0:
                continue

            # Safety check:
            if np.count_nonzero(self.links['coreRank'][links_curr[ind_goodlinks]] != ind_goodlinks) > 0:
                print("Inconsistent link ordering in filter_long_links...")
                set_trace()

            # Set up a temporary mark list to flag forbidden links
            marklist = np.zeros(len(ind_goodlinks), dtype = np.int8)

            # Below, we find the 'candidate' = index into ind_goodlinks
            # of best link candidate. Initialise to 'none_found'
            candidate = -1

            if not args.prevent_revenge:
                candidate = 0  
            else:

                # --------------------------------------------------------
                # Now the tricky bit: mark certain targets as 'off-limits'
                # --------------------------------------------------------
            
                # Find all short links from current sender
                # (N.B.: self.isnap+1 to access links TO self.isnap+1!)
                nShortLinks_curr = allLinks[self.isnap+1].sendOffset[isend+1]-allLinks[self.isnap+1].sendOffset[isend]

                # No shortlinks --> no problem 
                # (this cannot happen with case-B)
                if nShortLinks_curr == 0:
                    candidate = 0
                else:
                
                    # Need to find the target galaxy numbers of all (good)links

                    # Following bit was updated 22-Apr-18 to actually
                    # make revenge prevention work

                    # Need to construct a temporary list of 
                    # 'current best guess' galaxy IDs in target snapshot
                    # (the definitive assignment is only done later)
                    tempGal = np.zeros(snaps[self.isnap+self.length].nHaloes, dtype = np.int32)-1
                    
                    # Need to loop over already covered connection lengths
                    # (i.e. those where connections might have been made)
                    for itback in range(1, 6):
                        
                        ind_thisback = np.nonzero(snaps[self.isnap+self.length].reverse['Length'] == itback)[0]
                        if itback >= self.length and len(ind_thisback) > 0:
                            print("Error: found prior connections from current snapshot length or larger?!?")
                            set_trace()
                        
                        if len(ind_thisback) > 0:
                            shi_thisback = snaps[self.isnap+self.length].reverse['SHI'][ind_thisback]
                            tempGal[ind_thisback] = snaps[self.isnap+self.length-itback].local['Galaxy'][shi_thisback]

                    gal_goodlinks = tempGal[self.links[links_curr[ind_goodlinks]]['receiver']]

                    if args.exclude_all_linktargs:
                        shortLinks_curr = allLinks[self.isnap+1].sendOffset[isend]+np.arange(nShortLinks_curr, dtype = int)
                        
                        # Target SHI of these shortlinks...
                        shi_short = allLinks[self.isnap+1].links[shortLinks_curr]['receiver']
                        # ... and their galaxy numbers (self.isnap+1 is correct!)
                        gal_short = snaps[self.isnap+1].local['Galaxy'][shi_short]
                        marklist[np.isin(gal_goodlinks, gal_short)] = 1
     
                    else:
                        # Exclude merger descendant only:
                        if self.sender_mask[isend] == 1:
                            gal_targ = snaps[self.isnap+1].local['Galaxy'][snaps[self.isnap].forward['SHI'][isend]]
                        else:
                            gal_targ = snaps[self.isnap+2].local['Galaxy'][snaps[self.isnap+1].forward['SHI'][snaps[self.isnap].forward['SHI']]]

                        marklist[gal_goodlinks == gal_targ] = 1


                    # EXTRA BIT for Case-B: check if the descendant has by now
                    # been (tentatively) long-linked. If yes, then ALL links 
                    # with CR below the long-link to this target 
                    # (if it exists) are off-limits
                    # N.B.: This works because the galaxies in snap j are 
                    # already updated with these (tentative) matches.
                    # N.B. II: second test is not really necessary, but avoids
                    # unnecessary searches where there is no such match 
                    # (likely the majority of cases)

                    # Safety check first:
                    if self.sender_mask[isend] == 2 and snaps[self.isnap].forward['Length'][isend] != 1:
                        print("Inconsistent lengths in filter_long_links!")
                        set_trace()
            
                    if self.sender_mask[isend] == 2 and snaps[self.isnap+1].forward['LengthTemp'][snaps[self.isnap].forward['SHI'][isend]] == nback-1:

                        # Find current galaxy (for sender and descendant!)
                        gal_curr = snaps[self.isnap].local['Galaxy'][isend]
                        
                        # Find the long-link to (potential, tentative)
                        # descendant in snapshot j:
                        # (N.B.: this does NOT have to be a 'goodlink'!)
                        ind_ll_to_descJ = np.nonzero(snaps[self.isnap+self.length].local['Galaxy'][self.links['receiver'][links_curr]] == gal_curr)[0]

                        if len(ind_ll_to_descJ) > 1:
                            print("Why is there more than one link to one galaxy???")
                            set_trace()

                        if len(ind_ll_to_descJ) == 1:
                            cr_descJ = self.links['coreRank'][links_curr[ind_ll_to_descJ[0]]]
                            if cr_descJ != ind_ll_to_descJ[0]:
                                print("Inconsistent ordering of links...")
                                set_trace()

                            # Now cross out galaxies with higher rank in 
                            # marklist - BUT marklist is only for goodlinks
                            
                            lowest_forbidden_goodlink = np.searchsorted(ind_goodlinks, cr_descJ, side = 'left') 

                            marklist[lowest_forbidden_goodlink:] = 1

                    # Almost done. Now find the best 'clear' (good)link:
                    ind_clear = np.nonzero(marklist == 0)[0]

                    if len(ind_clear) > 0:
                        candidate = ind_clear[0]
        
            # All if-clauses finished...
            # Now need to find all (allowed) galaxies within 2/3 of 
            # the coreNumPart of this candidate (if enabled, otherwise
            # just choose the candidate itself)

            if candidate >= 0:

                if args.allow_multiple_longlinks:
                    cnp_candidate = self.links['coreNumPart'][links_curr[ind_goodlinks[candidate]]]
                    ind_permitted = np.nonzero((self.links['coreNumPart'][links_curr[ind_goodlinks]] >= sim_frac * cnp_candidate) & (marklist == 0))[0]
                    
                    # Safety check:
                    if len(ind_permitted) == 0:
                        print("Why are there zero permitted galaxies?")
                        set_trace()
                    
                    permitted_mask[links_curr[ind_goodlinks[ind_permitted]]] = 1
                else:
                    permitted_mask[links_curr[ind_goodlinks[candidate]]] = 1


        # Done looping through senders.
        # Last step: convert from permitted mask (1 = good) to status
        # (1 = bad, because 0 = good)

        self.links['status'] = 1-permitted_mask



    def select_links(self, priority, quiet = False):

        """
        Core function to select 'connection' links from the network.
        """

        # Check which links are allowed (from long-link selection and/or
        # permissions on irrlicht rejection

        subind = np.nonzero(self.links['status'] == 0)[0]

        if not quiet:
            print(self._pre + "... selecting links...", flush = True)
            print(self._pre + "    ({:d}/{:d} eligible links = {:.2f}%)"
                .format(len(subind), self.nLinks, len(subind)/self.nLinks*100))

        match_ij = np.zeros(self.nHaloesA, dtype = int)-1
        match_ji = np.zeros(self.nHaloesB, dtype = int)-1
        connection_ij = np.zeros(self.nHaloesA, dtype = int)-1
        connection_ji = np.zeros(self.nHaloesB, dtype = int)-1
        
        if len(subind) == 0:
            return match_ji, connection_ji

        # The following is a bit abstractly written to allow ordering
        # by both rank or choice

        if priority == 'rank':
            q1 = self.links['coreRank'][subind]
            q2 = self.links['compensatedChoice'][subind]
            first_site = 'sender'
            second_site = 'receiver'
            firstmatch = match_ij
            secondmatch = match_ji
        else:
            q1 = self.links['compensatedChoice'][subind]
            q2 = self.links['coreRank'][subind]
            first_site = 'receiver'
            second_site = 'sender'
            firstmatch = match_ji
            secondmatch = match_ij

        for iq1 in range(np.max(q1)+1):

            ind_q1 = np.nonzero((q1 == iq1) & (firstmatch[self.links[first_site][subind]] == -1))[0]
            if len(ind_q1) == 0:
                continue

            for iq2 in range(np.max(q2[ind_q1])):
                ind_q2 = np.nonzero((q2[ind_q1] == iq2) & (secondmatch[self.links[second_site][subind[ind_q1]]] == -1))[0]
        
                # Only check whether second site is free now! Each first site is only called once in current q1 loop, but each second site may receive multiple links of current q1. So we need to check before each individually whether the second site is still available.
            
                if len(ind_q2) == 0:
                    continue

                match_ij[self.links['sender'][subind[ind_q1[ind_q2]]]] = self.links['receiver'][subind[ind_q1[ind_q2]]]
                match_ji[self.links['receiver'][subind[ind_q1[ind_q2]]]] = self.links['sender'][subind[ind_q1[ind_q2]]]
                connection_ji[self.links['receiver'][subind[ind_q1[ind_q2]]]] = subind[ind_q1[ind_q2]]
                connection_ij[self.links['sender'][subind[ind_q1[ind_q2]]]] = subind[ind_q1[ind_q2]]

        
        n_match = np.count_nonzero(match_ji >= 0)        
        
        if not quiet:
            print(self._pre + "   ---> could match {:d} subhaloes ({:.2f}% / {:.2f}%)" 
              .format(n_match, n_match/self.nHaloesA*100, n_match/self.nHaloesB*100))

        return match_ji, connection_ji



    def select_links_consistently(self, priority):

        """
        MODIFIED selection function aiming for better consistency
        across snapshots. Started 8-Feb-2018
        """

        stime_slc = time.time()

        # Check which links are allowed (from long-link selection and/or
        # permissions on irrlicht rejection

        subind = np.nonzero(self.links['status'] == 0)[0]

        print(self._pre + "... selecting links...")
        print(self._pre + "    ({:d}/{:d} eligible links = {:.2f}%)"
              .format(len(subind), self.nLinks, len(subind)/self.nLinks*100))

        match_ij = np.zeros(self.nHaloesA, dtype = int)-1
        match_ji = np.zeros(self.nHaloesB, dtype = int)-1
        connection_ij = np.zeros(self.nHaloesA, dtype = int)-1
        connection_ji = np.zeros(self.nHaloesB, dtype = int)-1
        
        if len(subind) == 0:
            return match_ji, connection_ji

        # In this version, ONLY RANK-PRIORITY is supported:
        if priority != 'rank':
            print("Sorry, select_links_consistently() only works with 'rank' priority...")
            set_trace()

        # Need to load long links (not if in last step, and note that
        # isnap refers to snap at the START of the interval!
        if self.length < 5 and self.isnap+self.length < (nsnap-1):
            longLinks = Links(self.isnap, self.length+1, norm = False, compensate = False, find_permitted = False, filter_long = False, quiet = True)


        # Now loop through (core)rank levels

        n_questionable = 0
        n_uncompwin = 0
        
        for icr in range(np.max(self.links['coreRank'][subind])+1):

            # Find all links to still unconnected receivers:
            ind_cr_link = np.nonzero((match_ji[self.links['receiver'][subind]] == -1) & (self.links['coreRank'][subind] == icr) & (match_ij[self.links['sender'][subind]] == -1))[0]
            if len(ind_cr_link) == 0:
                continue

            # Make receiver-offset-list for these links:
            recv_offset = np.zeros(self.nHaloesB+2, dtype = int) # One extra for last galaxy, one extra for coda
            
            histogram = np.histogram(self.links['receiver'][subind[ind_cr_link]], bins = self.nHaloesB+1, range = [0, self.nHaloesB+1])[0]
 
            recv_offset[1:] = np.cumsum(histogram)
            recv_argsort = np.argsort(self.links['receiver'][subind[ind_cr_link]])

            # Sort out simple cases: only one link to a receiver
            ind_recv_onelink = np.nonzero(recv_offset[1:]-recv_offset[:-1] == 1)[0]
            link_cr = self.links[subind[ind_cr_link]]
            link_onelink = link_cr[recv_argsort[recv_offset[ind_recv_onelink]]]

            match_ij[link_onelink['sender']] = link_onelink['receiver']
            match_ji[link_onelink['receiver']] = link_onelink['sender']
            connection_ji[link_onelink['receiver']] = subind[ind_cr_link[recv_argsort[recv_offset[ind_recv_onelink]]]]

            # Rest only for 2+ links per receiver
            ind_recv_multilink = np.nonzero(recv_offset[1:]-recv_offset[:-1] > 1)[0]
            
            for irecv in ind_recv_multilink:
                # irecv is the receiver we're dealing with currently

                subind_links = subind[ind_cr_link[recv_argsort[recv_offset[irecv]:recv_offset[irecv+1]]]]

                if len(subind_links) < 2:
                    print("Why are there < 2 links?")
                    set_trace()

                if self.isnap + self.length < nsnap-1:
                    probableDescendant = probMatch_jk[irecv]
                else:
                    probableDescendant = -1


                # If there is no probable descendant, we just use 
                # 'normal rules', i.e. pick the highest choice galaxy
                if probableDescendant < 0:
                    subind_sel = np.argmin(self.links['compensatedChoice'][subind_links])
                else:

                    # If there IS a (probable) descendant:
                    subind_sel_uncomp = np.argmin(self.links['choice'][subind_links])
                    subind_sel_comp = np.argmin(self.links['compensatedChoice'][subind_links])
                    if subind_sel_uncomp == subind_sel_comp:
                        subind_sel = subind_sel_comp
                        
                    else:
                        # Now the tricky case: compensated and uncompensated winner
                        # are different. Check long-links from these to PD:

                        n_questionable += 1

                        # Can't do anything if we're already looking at longest level
                        if self.length >= 5 or self.isnap+self.length >= nsnap-1:
                            subind_sel = subind_sel_comp
                        else:

                            ind_ll_uncomp = np.nonzero((longLinks.links['sender'] == self.links['sender'][subind_links[subind_sel_uncomp]]) & (longLinks.links['receiver'] == probableDescendant))[0]
                            ind_ll_comp = np.nonzero((longLinks.links['sender'] == self.links['sender'][subind_links[subind_sel_comp]]) & (longLinks.links['receiver'] == probableDescendant))[0] 

                            if len(ind_ll_uncomp) > 1 or len(ind_ll_comp) > 1:
                                print("Inconsistent result of np.nonzero...")
                                set_trace()
                
                            if len(ind_ll_uncomp) == 0:
                                subind_sel = subind_sel_comp
                            elif len(ind_ll_uncomp) == 1 and len(ind_ll_comp) == 0:
                                subind_sel = subind_sel_uncomp
                                n_uncompwin += 1
                            else:
                                # Both candidates have a long-link to PD
                                # Check if one has higher CR:
                    
                                cr_ll = longLinks.links['coreRank'][[ind_ll_uncomp[0], ind_ll_comp[0]]]
                    
                                if cr_ll[0] > cr_ll[1]:  # Comp CR LOWER (=better)
                                    subind_sel = subind_sel_comp
                                elif cr_ll[0] < cr_ll[1]: # Uncomp CR lower
                                    subind_sel = subind_sel_uncomp
                                    n_uncompwin += 1
                                else:
                                    subind_sel = subind_sel_comp
                    

                match_ij[self.links['sender'][subind_links[subind_sel]]] = self.links['receiver'][subind_links[subind_sel]]
                match_ji[self.links['receiver'][subind_links[subind_sel]]] = self.links['sender'][subind_links[subind_sel]] 
                connection_ji[self.links['receiver'][subind_links[subind_sel]]] = subind_links[subind_sel] 

     
        n_match = np.count_nonzero(match_ji >= 0)        
        print(self._pre + "   ---> could match {:d} subhaloes ({:.2f}% / {:.2f}%; {:.3f} sec.)" 
              .format(n_match, n_match/self.nHaloesA*100, n_match/self.nHaloesB*100, time.time()-stime_slc))

        if n_questionable == 0:
            print(self._pre + "   ---> {:d} unclear races!"
                  .format(n_questionable))
        else:
            print(self._pre + "   ---> out of {:d} unclear races, {:d} = {:.2f}% were won by uncompensated choice."
              .format(n_questionable, n_uncompwin, n_uncompwin/n_questionable*100))

            

        return match_ji, connection_ji


    def find_permitted_links(self):

        """
        Select links that are permitted by rules to limit connections
        to spectres.

        This is simplified from the earlier version, and only works
        within current snapshot.
        """

        if not self.quiet:
            print(self._pre + "... identifying permitted links...")

        stime = time.time()

        # Need to loop through the links (once), because we need to find the 
        # num / corenum / mass maxima at each A and B halo

        # We could get maxCoreNum_sender and maxMass_receiver from the 
        # offset lists, but not the other two...

        maxNum_sender = np.zeros(self.nHaloesA)
        maxNum_receiver = np.zeros(self.nHaloesB)

        maxMass_receiver = np.zeros(self.nHaloesB)
        maxCoreNum_sender = np.zeros(self.nHaloesA)

        ind_getsLink = np.nonzero(self.recvOffset[1:] > self.recvOffset[:-1])[0]
        ind_sendsLink = np.nonzero(self.sendOffset[1:] > self.sendOffset[:-1])[0]
        
        maxMass_receiver[ind_getsLink] = self.links['mass'][self.links['sortByRecv'][self.recvOffset[ind_getsLink]]]
        maxCoreNum_sender[ind_sendsLink] = self.links['coreNumPart'][self.sendOffset[ind_sendsLink]]

        maxNum_sender[ind_sendsLink] = np.array([np.max(self.links['numPart'][self.sendOffset[ii]:self.sendOffset[ii+1]]) for ii in ind_sendsLink], dtype = int)
        maxNum_receiver[ind_getsLink] = np.array([np.max(self.links['numPart'][self.links['sortByRecv'][self.recvOffset[ii]:self.recvOffset[ii+1]]]) for ii in ind_getsLink], dtype = int)

        """
        for ilink in range(self.nLinks):
            maxNum_sender[self.links['sender'][ilink]] = np.max((self.links['numPart'][ilink], maxNum_sender[self.links['sender'][ilink]]))
            maxNum_receiver[self.links['receiver'][ilink]] = np.max((self.links['numPart'][ilink], maxNum_receiver[self.links['receiver'][ilink]]))
            maxCoreNum_sender[self.links['sender'][ilink]] = np.max((self.links['coreNumPart'][ilink], maxCoreNum_sender[self.links['sender'][ilink]]))
            maxMass_receiver[self.links['receiver'][ilink]] = np.max((self.links['mass'][ilink], maxMass_receiver[self.links['receiver'][ilink]]))
        """
 
        # A link is forbidden if it has
        # (1) less than 2/3 coreNum of max link from same sender,
        #              AND
        # (2) less than 2/3 num of max link from same sender,
        #              AND EITHER
        # (3a) mass less than 2/3 the most massive link to same receiver,
        #                 OR
        # (3b) num less than 2/3 the most numerous link to same receiver
        # 
        # This means that clear descendant links are ALWAYS allowed,
        # even if they 'steal' the majority of mass from another galaxy.
        # Such a situation may arise e.g. in the run-up to a major merger.

        ind_forbidden = np.nonzero((self.links['coreNumPart'] < sim_frac*maxCoreNum_sender[self.links['sender']]) & 
                                   (self.links['numPart'] < sim_frac*maxNum_sender[self.links['sender']]) & 
                                   ((self.links['mass'] < sim_frac*maxMass_receiver[self.links['receiver']]) | (self.links['numPart'] < sim_frac*maxNum_receiver[self.links['receiver']])))[0]
        
        if args.limit_links:
            self.links['status'][ind_forbidden] = 2

        # New bit added 28-Apr-2018: Explicitly disable links containing
        # less than three core particles, unless they account for > 2/3
        # of all particles from the current sender
        #
        # This is to prevent artificial jumps to small galaxies near the edge
        # of a merged central that just happen to have been swamped with
        # (mis-assigned) particles from the central.

        if args.require_core:
            
            ind_suppressed = np.nonzero((self.links['coreNumPart'] < 3) & 
                                        (self.links['numPart'] < sim_frac * maxNum_sender[self.links['sender']]))[0]
            self.links['status'][ind_suppressed] = 2
        
        # If desired, exclude links that constitute 'revenge' takeover
        # (i.e. to a galaxy that had hidden a galaxy in the past)
        # Added 9-May-2018        

        if args.blockAllRevenge and self.blockAllRevenge and self.isnap > 1 and blocktable.nEntries > 0:

            if not self.quiet:
                print(self._pre + "   (... identifying blocked links...)")
                block_stime = time.time()

            # Convert block entries to subhalo in sender snapshot
            target_sh = snaps[self.isnap].SHI(blocktable.table['gal_target'])
            blocked_sh = snaps[self.isnap].SHI(blocktable.table['gal_blocked'])
            ind_valid_sh = np.nonzero((blocked_sh >= 0) & (target_sh >= 0))[0]
            if np.max(blocked_sh) >= len(self.sendOffset):
                print("Out of range entry in blocked_sh!")
                set_trace()
            
            # Additional check (added 18-05-2018): must (!) ensure that
            # blocked_sh actually sends one link, otherwise blocked_recv is 
            # taken from the NEXT subhalo!!
            subind_trueBlock = np.nonzero(self.sendOffset[blocked_sh[ind_valid_sh]+1] > self.sendOffset[blocked_sh[ind_valid_sh]])[0]
            ind_valid_sh = ind_valid_sh[subind_trueBlock]

            blocked_recv = self.links['receiver'][self.sendOffset[blocked_sh[ind_valid_sh]]]
            
            
            # Now identify links going from target_sh to blocked_recv:
            
            connType = [('sender', 'i4'), ('receiver', 'i4')]
            blockedConn = np.zeros(len(ind_valid_sh), dtype = connType)
            blockedConn['sender'] = target_sh[ind_valid_sh]
            blockedConn['receiver'] = blocked_recv

            linkConn = np.zeros(self.nLinks, dtype = connType)
            linkConn['sender'] = self.links['sender']
            linkConn['receiver'] = self.links['receiver']

            ind_blocked = np.nonzero(np.isin(linkConn, blockedConn))[0]
            self.links['status'][ind_blocked] = 2
            n_blocked = len(ind_blocked)
            
            """
            for ilink in range(len(self.links)):

                # Retrieve galaxy of current sender:
                curr_sender = self.links['sender'][ilink]
                curr_receiver = self.links['receiver'][ilink]
                gal_sender = snaps[self.isnap].local['Galaxy'][curr_sender]

                # Retrieve list of blocked galaxies
                gals_blocked = blocktable.query(gal_sender)
                
                # Now need to check if there is a block...
                links_to_recv = self.links['sortByRecv'][self.recvOffset[curr_receiver]:self.recvOffset[curr_receiver+1]]
                
                cr_links_to_recv = self.links['coreRank'][links_to_recv]
                sender_links_to_recv = self.links['sender'][links_to_recv]

                ind_cr0 = np.nonzero(cr_links_to_recv == 0)[0]
                if len(ind_cr0) == 0: continue

                for isender_cr0 in sender_links_to_recv[ind_cr0]:
                    isender_gal = snaps[self.isnap].local['Galaxy'][isender_cr0]
                    if isender_gal in gals_blocked:
                        self.links['status'][ilink] = 2
                        n_blocked += 1
                        break

            """

            if not self.quiet:
                print(self._pre + "      (---> excluded {:d}/{:d} (={:.2f}%) links ({:.3f} sec.)"
                          .format(n_blocked, self.nLinks, n_blocked/self.nLinks*100, time.time()-stime))

                

        if not args.lower_ranks:
            ind_lowrank = np.nonzero(self.links['coreRank'] > 0)[0]
            self.links['status'][ind_lowrank] = 2

        if not args.lower_choice:
            ind_lowchoice = np.nonzero(self.links['compensatedChoice'] > 0)[0]
            self.links['status'][ind_lowchoice] = 2

        if args.no_spectre_descendants and self.length > 1:
            spectreMask = np.zeros(self.nHaloesA, dtype = np.int8)
            spectreMask[snaps[self.isnap].test_flag("SPECTRE")] = 1
            ind_fromSpectre = np.nonzero(spectreMask[self.links['sender']] == 1)[0]
            self.links['status'][ind_fromSpectre] = 2


        n_excluded = np.count_nonzero(self.links['status'] == 2)

        if not self.quiet:
            print(self._pre + "   ---> excluded {:d}/{:d} (={:.2f}%) links ({:.3f} sec.)"
              .format(n_excluded, self.nLinks, 
                      n_excluded/self.nLinks*100, time.time()-stime))


    def find_links(self, sender, receiver):
        
        """
        Idenfity the links between the provided sender and receivers
        """

        maxsend_s = np.max(sender)+1
        maxsend_l = np.max(self.links['sender'])+1
        maxsend = np.max((maxsend_s, maxsend_l))
        
        # Set up array for provided connections
        connection = np.zeros(maxsend, dtype = int)-1
        connection[sender] = receiver
    
        ind_connect = np.nonzero(self.links['receiver'] == connection[self.links['sender']])[0]
        return ind_connect


class ExchangeList:

    """
    Class to hold information about *instantaneous* exchanges between 
    galaxies, as opposed to integrated info in ExchangeHistory
    """

    def __init__(self, links, snap_i, snap_j):

        utime = time.time()
      
        print("   ... constructing current exchange list...")
  
        self.snap_i = snap_i.isnap
        self.snap_j = snap_j.isnap

        self.xdtype = [('gal_i', 'i4'), ('gal_j', 'i4'), ('mass', 'd'), ('compMassFrac', 'd'), ('type', 'i1')]

        # i) exchange link: sender matched, and receiver either long-linked (then guaranteed not to sender!) or short-linked to something else

        ind_xlink_exchange = np.nonzero((snap_i.forward['Length'][links.links['sender']] > 0) & 
                                        ((snap_j.reverse['Length'][links.links['receiver']] > 1) | ((snap_j.reverse['Length'][links.links['receiver']] == 1) & (snap_j.reverse['SHI'][links.links['receiver']] != links.links['sender']))))[0]

        # ii) links from merged or to new galaxies

        ind_xlink_mergedNew = np.nonzero((snap_i.forward['Length'][links.links['sender']] == 0) | (snap_j.reverse['Length'][links.links['receiver']] == 0))[0]
        self.ind_xlink = np.concatenate((ind_xlink_exchange, ind_xlink_mergedNew))
        
        if len(self.ind_xlink) != len(np.unique(self.ind_xlink)):
            print("Duplicate exchange links?!")
            set_trace()
        
        self.nXlink = len(self.ind_xlink)
        
        print("   ---> found {:d} exchange links (= {:.2f}% of all)..." 
              .format(self.nXlink, self.nXlink/links.nLinks*100))

        print("      ---> ({:d} cross-links = {:.2f}%)" .format(len(ind_xlink_exchange), len(ind_xlink_exchange)/self.nXlink*100))
        print("      ---> ({:d} merger/formation-links = {:.2f}%)" .format(len(ind_xlink_mergedNew), len(ind_xlink_mergedNew)/self.nXlink*100))

        self.xlist = np.zeros(self.nXlink, dtype = self.xdtype)

        self.xlist['gal_i'] = snap_i.local['Galaxy'][links.links['sender'][self.ind_xlink]]
        self.xlist['gal_j'] = snap_j.local['Galaxy'][links.links['receiver'][self.ind_xlink]]
      
        if np.count_nonzero(self.xlist['gal_i'] == self.xlist['gal_j']) > 0:
            set_trace()
        
        self.xlist['mass'] = links.links['mass'][self.ind_xlink]
        self.xlist['compMassFrac'] = links.links['compensatedNormRF'][self.ind_xlink]/snap_j.reverse['NormRF'][links.links['receiver'][self.ind_xlink]]

        self.xlist['type'][:len(ind_xlink_exchange)] = 1
        self.xlist['type'][len(ind_xlink_exchange):] = 2

        # Additional step: sort xlist_curr by gal_i and gal_j and create an 
        # offset list, so it's easier to search the list later

        self._make_offset_list(main.maxgal)


        #print("<> Creating interaction list took {:.3f} sec." .format(time.time()-utime))


    def add_new(self, linkinds):
        
        """
        Add new elements later
        """
        
        n_new = len(linkinds)

        self.xlist = np.concatenate((self.xlist, np.zeros(n_new, dtype = self.xdtype)))
        self.xlist['gal_i'][self.nXlink:] = allLinks[self.snap_j].links['sender'][linkinds]
        self.xlist['gal_j'][self.nXlink:] = np.arange(main.maxgal+1, main.maxgal+n_new+1, dtype = int)
        self.xlist['mass'][self.nXlink:] = allLinks[self.snap_j].links['mass'][linkinds]
        self.xlist['type'][self.nXlink:] = 2

        self.nXlink += n_new
        self._make_offset_list(main.maxgal+n_new)


    def _make_offset_list(self, maxgal):

        if np.max(self.xlist['gal_i']) > maxgal:
            print("gal_i > maxgal in ExchangeList initialisation...?")
            set_trace()
            
        self.offset = np.zeros(maxgal+2, dtype = int) # One extra for last galaxy, one extra for coda

        self.xlist.sort(order = ('gal_i', 'gal_j'))
        
        histogram = np.histogram(self.xlist['gal_i'], bins = maxgal+1, range = [0, maxgal+1])[0]
 
        self.offset[1:] = np.cumsum(histogram)


  
class Galaxies:
    
    """
    Class to hold and compute global galaxy info.
    This is only computed at the end of the program.
    """

    def __init__(self, maxgal):
        
        self.shiTable = np.zeros((maxgal+1, nsnap), dtype = np.int32) - 100
        self.mergelist = np.zeros((maxgal+1, nsnap), dtype = np.int32) - 100
        self.firstlist = np.zeros(maxgal+1, dtype = np.int32) - 1
        self.lastlist = np.zeros(maxgal+1, dtype = np.int32) - 1
        self.numlist = np.zeros(maxgal+1, dtype = np.int32)
        
        maxgal_curr = -1

        for isnap in range(nsnap):
            
            snap = snaps[isnap]
            galaxies_curr = snap.local['Galaxy']

            # Update number of alive snapshots
            self.numlist[galaxies_curr] += 1
    
            # Identify galaxies that are new in this snapshot
            ind_new = snap.test_flag("NEW_GAL")
            if len(ind_new) > 0:

                """
                if np.max(snap.local['Galaxy'][ind_new]) <= maxgal_curr:
                    print("Unexpected index of new galaxy...?")
                    set_trace()
                """

                self.firstlist[galaxies_curr[ind_new]] = isnap
                self.mergelist[galaxies_curr[ind_new], isnap] = galaxies_curr[ind_new]
            
                # Update maxgal counter for next iteration
                maxgal_curr = np.max(galaxies_curr)
                
                #if maxgal_curr != maxgal_old + len(ind_new):
                #    print("Inconsistent number of new galaxies...")
                #    set_trace()
                
                #maxgal_old += len(ind_new)

            # Fill in (reverse) galaxy table for this snapshot
            self.shiTable[galaxies_curr, isnap] = np.arange(snap.nHaloes, dtype = np.int32)

            # Find which galaxies have permanently died after this snapshot
            ind_died = np.nonzero(snap.forward['Length'] == 0)[0]

            if len(ind_died) > 0:
                snap.add_flag(ind_died, "LASTSNAP")
                self.lastlist[galaxies_curr[ind_died]] = isnap

            print("Snapshot {:d}: {:d} galaxies total, of which {:d} new, {:d} last"
                  .format(isnap, snap.nHaloes, len(ind_new), len(ind_died)))

            # All the following is irrelevant for the last snapshot
            if isnap == nsnap-1:
                continue
                
            # Mark skipped galaxies in following snapshot(s)
            for ilen in range(2, np.min((6, nsnap-isnap))):
                ind_skipped = np.nonzero(snap.forward['Length'] == ilen)[0]
                
                if len(ind_skipped) > 0:
                    gal_skipped = snap.local['Galaxy'][ind_skipped]
                    self.shiTable[gal_skipped, isnap+1:isnap+ilen] = -9
                
            # Mark merged galaxies
            ind_merged = np.nonzero((snap.forward['Length'] == 0) & (snap.forward['SHI'] >= 0))[0]
            if len(ind_merged) != len(ind_died) - len(snap.test_flag("NOSENTLINK_SHORT", ind_died)):
                print("Inconsistent number of merged galaxies...")
                set_trace()

            if len(ind_merged) > 0:
                gal_merged = snap.local['Galaxy'][ind_merged]
                gal_targets = snaps[isnap+1].local['Galaxy'][snap.forward['SHI'][ind_merged]]
                self.shiTable[gal_merged, isnap+1] = -5
                self.shiTable[gal_merged, isnap+2:] = -15

            # Now update mergelist
            # Default is to DO NOTHING and keep same galaxy
            # as in previous snapshot!
            merger_update_table = np.arange(main.maxgal+1, dtype = int)
            
            if len(ind_merged) > 0:
                merger_update_table[gal_merged] = gal_targets

            # Can only update the table for (already) exist(ing/ed)
            # galaxies. Others will be added new in subsequent snaps.
            ind_exists = np.nonzero(self.mergelist[:, isnap] >= 0)[0]
            
            if np.min(merger_update_table[self.mergelist[ind_exists, isnap]]) < 0:
                print("Unexpected entry in merger_update_table...")
                set_trace()
            
            self.mergelist[ind_exists, isnap+1] = merger_update_table[self.mergelist[ind_exists, isnap]]


            # Mark faded galaxies:
            ind_faded = np.nonzero((snap.forward['Length'] == 0) & (snap.forward['SHI'] < 0))[0]

            if len(ind_faded) > 0:
                gal_faded = snap.local['Galaxy'][ind_faded]

                self.shiTable[gal_faded, isnap+1] = -10
                self.shiTable[gal_faded, isnap+2:] = -20
            
            print("        ---> {:d} galaxies faded, {:d} merged."
                  .format(len(ind_faded), len(ind_merged)))

        # Some sanity checks:
        if np.count_nonzero(self.firstlist < 0):
            print("A galaxy has no first snapshot?")
            set_trace()
        if np.count_nonzero(self.lastlist < 0):
            print("A galaxy has no last snapshot?")
            set_trace()
        if np.count_nonzero(self.numlist < 0):
            print("A galaxy appears nowhere")
            set_trace()
            

    def write(self, outloc):

        """
        Write to output file
        """

        yb.write_hdf5(self.shiTable, outloc, 'SubHaloIndex', comment = "Subhalo index of galaxy i (first index) in snapshot j (second index). Negative values mean that the galaxy does not exist in this snapshot (-100: not formed yet; -5: just merged; -9: skipped; -10: just faded; -15: merged in past; -20: faded in past")
        yb.write_hdf5(self.mergelist, outloc, 'MergeList', comment = "Galaxy containing galaxy i (first index) in snapshot j (second index). After a galaxy has merged, the entry of it and all other galaxies with entry i in the pre-merger snap are updated to the index of the galaxy it has merged with. All galaxies that are still alive in a given snapshot therefore have entry i, and the number of entries equal to i at a given snapshot is equal to the total number of progenitors of this galaxy. Negative entries mean that the galaxy has not formed yet.")
        yb.write_hdf5(self.firstlist, outloc, 'FirstSnap', comment = "Snapshot in which the galaxy was first identified.")
        yb.write_hdf5(self.lastlist, outloc, 'LastSnap', comment = "Last snapshot in which the galaxy was identified.")
        yb.write_hdf5(self.numlist, outloc, 'NumOfSnaps', comment = "Total number of snapshots in which the galaxy was identified.")


    
class Mergers:
    
    """
    Class to compute information about mergers
    """

    def __init__(self, isnap):

        # Don't have mergers before first snapshot...
        if isnap == 0:
            return

        galaxies_curr = snaps[isnap].local['Galaxy']
        nHaloesCurr = galaxies_curr.shape[0]

        # Set up and initialize output array
        merger_results = np.zeros(nHaloesCurr, dtype = [('MaxMergFrac', 'd'), ('NumSwallows', 'i'), ('IndMaxMerg', 'i')])
        merger_results['IndMaxMerg'][:] = -1
        
        xlist_offset_curr = xlist[isnap].offset
        xlist_curr = xlist[isnap].xlist
        
        # Go through MERGED subhaloes in previous snap
        ind_merged = np.nonzero((snaps[isnap-1].forward['Length'] == 0) & 
                                (snaps[isnap-1].forward['SHI'] >= 0))[0]

        print("")
        print("Merger analysis, snap {:d}: {:d} galaxies merged." 
              .format(isnap, len(ind_merged)))

        for imerg, sh_merg in enumerate(ind_merged):

            targ_sh = snaps[isnap-1].forward['SHI'][sh_merg]
            curr_mrat = snaps[isnap-1].forward['CompensatedNormRF'][sh_merg] / snaps[isnap].reverse['CompensatedNormRF'][targ_sh]
            
            merger_results[targ_sh]['NumSwallows'] += 1
            
            if curr_mrat > 0.1 and curr_mrat <= 1/3:
                snaps[isnap].add_flag(targ_sh, "MINOR_MERGER")
            if curr_mrat > 1/3 and curr_mrat <= 2/3:
                snaps[isnap].add_flag(targ_sh, "MAJOR_MERGER")
            if curr_mrat > 2/3:
                snaps[isnap].add_flag(targ_sh, "NEAREQUAL_MERGER")
            
            if curr_mrat > merger_results[targ_sh]['MaxMergFrac']:
                merger_results[targ_sh]['MaxMergFrac'] = curr_mrat
                merger_results[targ_sh]['IndMaxMerg'] = sh_merg
        

        # And final bit: identify new galaxies that experienced merger
        ind_new_with_mergers = np.nonzero((merger_results['NumSwallows'] > 0) & (snaps[isnap].check_flag("NEW_GAL")))[0]
        snaps[isnap].add_flag(ind_new_with_mergers, "NEW_WITH_MERGER")


        # ------------------------------
        # Write out results immediately
        # ------------------------------
        
        yb.write_hdf5(merger_results['MaxMergFrac'], outloc, 'Subhalo/Snapshot_' + str(isnap).zfill(3) + '/Merger/MaxRatio', comment = "(Compensated) mass ratio of maximum merger of this galaxy since last snapshot.")
        yb.write_hdf5(merger_results['IndMaxMerg'], outloc, 'Subhalo/Snapshot_' + str(isnap).zfill(3) + '/Merger/MaxMergerSubhalo', comment = "Subhalo index in last snapshot that corresponds to the maximum merger in preceding snapshot interval.")
        yb.write_hdf5(merger_results['NumSwallows'], outloc, 'Subhalo/Snapshot_' + str(isnap).zfill(3) + '/Merger/NumOfSwallows', comment = "Total number of subhaloes that have merged with this galaxy since the last snapshot.")


        # ------------------------------
        # Print some overview statistics
        # ------------------------------

        n_minor = len(snaps[isnap].test_flag('MINOR_MERGER'))
        n_major = len(snaps[isnap].test_flag('MAJOR_MERGER'))
        n_neareq = len(snaps[isnap].test_flag('NEAREQUAL_MERGER'))

        print("     ==> {:d} galaxies experienced minor merger (={:.2f}%)"
              .format(n_minor, n_minor / snaps[isnap].nHaloes * 100))
        print("     ==> {:d} galaxies experienced major merger (={:.2f}%)"
              .format(n_major, n_major / snaps[isnap].nHaloes * 100))
        print("     ==> {:d} galaxies experienced near-equal merger (={:.2f}%)"
              .format(n_neareq, n_neareq / snaps[isnap].nHaloes * 100))
        


class PSS:

    """
    Full table of subhaloes for each particle. EXPERIMENTAL.
    """

    def __init__(self, maxid = None, dense_ids = True):

        if maxid is None:
            snapdir0 = st.form_files(rundir, 0, 'snap')
            ndm = yb.read_hdf5_attribute(snapdir0, 'Header', 'NumPart_Total')[1]
            maxid = 2*(ndm+2)+1


        self.table = np.zeros((0, nsnap), dtype = np.int32)
        self.index = np.zeros(maxid, dtype = int)-1
        self.id = np.zeros(0, dtype = int)
        self.nIDs = 0

    def update(self, isnap):

        """
        Main function to load particle info from disk and append table
        """

        print("Updating particle table, snapshot {:d}..."
              .format(isnap))

        stime = time.time()

        print("   ... loading ID files...", end = "")
        subdir = st.form_files(rundir, isnap)
        ids = st.eagleread(subdir, 'IDs/ParticleID', astro = False, silent = True)
        offset = st.eagleread(subdir, 'Subhalo/SubOffset', astro = False, silent = True)
        length = st.eagleread(subdir, 'Subhalo/SubLength', astro = False, silent = True)

        print("done ({:.3f} sec.)"
              .format(time.time() - stime))

        itime = time.time()

        nsub = len(length)
        newlist = np.zeros(0, dtype = np.int32)
        curr_entry = self.table.shape[0]
        npart_old = curr_entry
        new_ctr = 0

        print("   ... now checking individual subhalo particles...")
        #print(self._pre, end = '', flush = True)
        dotgap = next_dot = nsub/50

        # Pre-allocate max possible newlist, so we don't have to 
        # concatenate to it a zillion times below...
        totalIDs = np.sum(length)
        newlist = np.zeros(totalIDs, dtype = np.int32)-1

        print("   ", end = "")

        for isub in range(nsub):
            
            if isub > next_dot:
                print(".", end = '', flush = True) 
                next_dot += dotgap
                
            ids_this = ids[offset[isub]:offset[isub]+length[isub]]

            # Particles that are already in the list: add SHI
            ind_old = np.nonzero(self.index[ids_this] >= 0)[0]
            
            if len(ind_old) > 0:
                self.table[self.index[ids_this[ind_old]], isnap] = isub

            # Particles that are bound for the first time:
            ind_new = np.nonzero(self.index[ids_this] < 0)[0]

            if len(ind_new) > 0:
                self.index[ids_this[ind_new]] = np.arange(curr_entry, curr_entry + len(ind_new), dtype = int)
                newlist[new_ctr:new_ctr+len(ind_new)] = isub
                #newlist = np.concatenate((newlist, np.zeros(len(ind_new), dtype = np.int32)+isub))

                curr_entry += len(ind_new)
                new_ctr += len(ind_new)

        print("")
        print("    done, updating list...")

        self.nIDs += new_ctr

        # Now actually need to expand table to include new particles
        self.table = np.concatenate((self.table, np.zeros((new_ctr, nsnap), dtype = np.int32)-1))

        # TEMP TEST:
        # newlist = np.arange(new_ctr, dtype = int)
        self.table[npart_old:, isnap] = newlist[:new_ctr]

        # And update ID list (pss-index --> ID)
        self.id = np.zeros(curr_entry, dtype = int)-1
        ind_found = np.nonzero(self.index >= 0)[0]
        if len(ind_found) != curr_entry:
            print("Inconsistent numbers in PSS list...")
            set_trace()

        self.id[self.index[ind_found]] = ind_found 

        print("   ---> PSS table updated to length={:d}"
              .format(self.nIDs))
        print("   ---> writing to particle table took {:.3f} sec." 
              .format(time.time() - itime))

        print("   ---> total time for update was {:.3f} sec." 
              .format(time.time() - stime)) 
        
        
    def load_mass(self, isnap):
    
        """
        Load particle masses for the IDs listed in the table
        """

        partdir = st.form_files(rundir, isnap, 'subpart')
        
        self.mass = np.zeros(self.nIDs)

        for iiptype, ptype in enumerate([0, 1, 4, 5]):

            print("   ...loading masses for particle type {:d}, snap={:d}..."
                  .format(ptype, isnap), end = "")
            mstime = time.time()

            npart = yb.read_hdf5_attribute(partdir, 'Header', 'NumPart_Total')
            if npart[ptype] == 0:
                continue

            ids = st.eagleread(partdir, 'PartType{:d}/ParticleIDs' .format(ptype), astro = False, silent = True)

            if ptype != 1:
                mass = st.eagleread(partdir, 'PartType{:d}/Mass' .format(ptype), astro = True, silent = True)[0]
            else:
                mass = np.zeros(len(ids))+st.m_dm(rundir)
        

            ind_inpss = np.nonzero(self.index[ids] >= 0)[0]

            if self.mass[self.index[ids[ind_inpss]]].max() > 0:
                print("Why has a mass already been loaded for a particle...?")
                set_trace()

            self.mass[self.index[ids[ind_inpss]]] = mass[ind_inpss]
            
            print(" done ({:.3f} sec.)" .format(time.time()-mstime))
    

class SnipIDs:

    """
    Extract appropriate tracing IDs for snipshots.
    """

    def __init__(self):

        if args.trace_prebirth:
            self.lastSeenList = galProps.firstlist
            #ind_last_only = np.nonzero(self.lastSeenList >= nsnap-1)[0]
            #if len(ind_last_only) > 0:
            #    self.lastSeenList[ind_last_only] = -1
        else:
            self.lastSeenList = np.zeros(galProps.firstlist.shape[0], dtype = np.int32)-1

        self.tracLengthList = np.zeros(galProps.firstlist.shape[0], dtype = np.int32)
        self.tracLinkIndList = galProps.shiTable[np.arange(galProps.shiTable.shape[0],dtype = int), galProps.firstlist]  #np.zeros(galProps.firstlist.shape[0], dtype = np.int32)
        self.skipLengthList = np.zeros(galProps.firstlist.shape[0], dtype = np.int32)

        self.tracerOffsets = []
        self.IDlists = []
        self.Masslists = []
        self.Typelists = []
        self.isReadFlag = np.zeros((nsnap, 6), dtype = np.int8)

        for isnap in range(nsnap):
            curr_list_id = []
            curr_list_mass = []
            curr_list_type = []
            curr_list_offset = []
            
            for ilev in range(6):
                curr_list_id.append([])
                curr_list_mass.append([])
                curr_list_type.append([])
                curr_list_offset.append([])

            self.IDlists.append(curr_list_id)
            self.Masslists.append(curr_list_mass)
            self.Typelists.append(curr_list_type)
            self.tracerOffsets.append(curr_list_offset)


    def extract(self, isnap):
        
        """
        Main function: write out tracing particles for all galaxies
        for current snapshot
        """

        print("")
        print("Extracting snipshot tracing IDs, snapshot {:d}..." .format(isnap))
        print("   ... setup...", end = "")
        stime = time.time()


        
        # =====================================================
        # Part I: update the list of where to take tracers from
        # =====================================================

        # self.lastSeenList and self.tracLengthList indicate from which
        # link (initial snapshot and length) we should take the tracers.
        # For a subset galaxies, this needs to be updated because they 
        # are alive and 'good' (= not stolen) in current snap
        
        ind_update = np.nonzero(self.lastSeenList+self.skipLengthList == isnap)[0]        
        shi_update = galProps.shiTable[ind_update, isnap]
   
        # Safety check: all updating galaxies must be alive right now!
        if shi_update.min() < 0:
            print("Why does an update galaxy not exist??")
            set_trace()

        # The easy part: update lastSeenList
        self.lastSeenList[ind_update] = isnap

        # The not-so-easy part: find out length and skip values
        # (these are NOT always the same due to stolen subhaloes...)

        # Default: length is length of tracing link...
        forwardLength_update = snaps[isnap].forward['Length'][shi_update]

        # ... and distinguish between 'traced' and 'local'
        ind_localonly = np.nonzero(forwardLength_update == 0)[0]
        ind_traced = np.nonzero(forwardLength_update > 0)[0]

        # For galaxies that are not traced forward, things are simple:
        self.tracLengthList[ind_update[ind_localonly]] = 0
        self.skipLengthList[ind_update[ind_localonly]] = 0
        self.tracLinkIndList[ind_update[ind_localonly]] = shi_update[ind_localonly]
        
        # For all others, the fun is increased tremendously if we 
        # are rejecting stolen snapshots...
        if args.skip_stolen:

            # Find next subhalo of these galaxies
            targSHI_traced = snaps[isnap].forward['SHI'][shi_update[ind_traced]]
            # Now need to find their stolen fractions in the target snapshot
            fStolen_targ = np.zeros(len(ind_traced))
            for ilen in range(1,6):
                if isnap+ilen >= nsnap:
                    break

                subind_ilen = np.nonzero(forwardLength_update[ind_traced] == ilen)[0]
                fStolen_targ[subind_ilen] = snaps[isnap+ilen].local['fracStolen'][targSHI_traced[subind_ilen]]

            # Check which are ok in their target snap:
            ind_traced_ok = np.nonzero(fStolen_targ <= stolen_skip_threshold)[0]
            # For the ones that are ok, we can finish here:
            self.tracLengthList[ind_update[ind_traced[ind_traced_ok]]] = forwardLength_update[ind_traced[ind_traced_ok]]
            self.skipLengthList[ind_update[ind_traced[ind_traced_ok]]] = forwardLength_update[ind_traced[ind_traced_ok]]
            self.tracLinkIndList[ind_update[ind_traced[ind_traced_ok]]] = snaps[isnap].forward['Link'][shi_update[ind_traced[ind_traced_ok]]]

            # Now the fun part: dealing with those galaxies that are NOT
            # ok in their target snap

            ind_traced_corrupted = np.nonzero(fStolen_targ > stolen_skip_threshold)[0]

            if len(ind_traced_corrupted) > 0:

                # Initialize length and link of corrupted to local only
                # (i.e. local tracers, no more updates)

                length_corrupted = np.zeros(len(ind_traced_corrupted),dtype = int)
                link_corrupted = shi_update[ind_traced[ind_traced_corrupted]]
                skip_corrupted = np.zeros(len(ind_traced_corrupted),dtype = int)
                
                # Need to process each stolen galaxy in turn...
                for iicorr, corr in enumerate(ind_traced_corrupted):
            
                    # Start from snapshot that we've just tested as bad
                    temp_snap = isnap+forwardLength_update[ind_traced[corr]]
                    temp_shi = targSHI_traced[corr]

                    # Need to loop through all future snapshots to find
                    # next 'good' one. NOT enough to just go up to five next,
                    # because there may not be a good one in those.

                    while(True):

                        # Start with escape plan if we've exhausted all snaps

                        if temp_snap >= nsnap:
                            break

                        # Find next subhalo of this galaxy:
                        next_length = snaps[temp_snap].forward['Length'][temp_shi]
                        next_shi = snaps[temp_snap].forward['SHI'][temp_shi]
                        next_snap = temp_snap + next_length

                        # If galaxy has died in this snap
                        # --> leave at default values (local only)
                        if next_length == 0:
                            break
                
                        # If next subhalo is good, then take it:
                        if snaps[next_snap].local['fracStolen'][next_shi] <= stolen_skip_threshold:
                            # Can use long-links if next_snap is not too 
                            # far away. In a somewhat hack-y way, we just load
                            # the long-links as required...
                            if next_snap-isnap <= 5 and next_snap < nsnap:
                                
                                # Still need to find the long-link...
                                sender_curr = yb.read_hdf5(spiderloc, 'Level' + str(next_snap-isnap) + '/Snapshot_' + str(isnap).zfill(3) + '/Sender')
                                receiver_curr = yb.read_hdf5(spiderloc, 'Level' + str(next_snap-isnap) + '/Snapshot_' + str(isnap).zfill(3) + '/Receiver')
                                trialLinks = np.nonzero((sender_curr == shi_update[ind_traced[corr]]) & (receiver_curr == next_shi))[0]

                                # If there is a link, use it and not default
                                if len(trialLinks) == 1:
                                    link_corrupted[iicorr] = trialLinks[0]
                                    length_corrupted[iicorr] = next_snap-isnap
                                    
                            # In any case, we set this (next) SHI as time
                            # of next update, and are done
                            skip_corrupted[iicorr] = next_snap-isnap
                            break
                        
                        # If the subhalo is also stolen-corrupted, go to next
                        else:
                            temp_snap = next_snap
                            temp_shi = next_shi
                            continue
 
                # Done with the loop now, can finally update full list:
     
                if np.max(length_corrupted) + isnap >= nsnap:
                    print("Inconsistent tracers!")
                    set_trace()

                self.tracLengthList[ind_update[ind_traced[ind_traced_corrupted]]] = length_corrupted
                self.skipLengthList[ind_update[ind_traced[ind_traced_corrupted]]] = skip_corrupted
                self.tracLinkIndList[ind_update[ind_traced[ind_traced_corrupted]]] = link_corrupted

        # Life is much simpler if we are not rejecting stolen snapshots:
        else:
            self.tracLengthList[ind_update[ind_traced]] = forwardLength_update[ind_traced]
            self.skipLengthList[ind_update[ind_traced]] = forwardLength_update[ind_traced]
            self.tracLinkIndList[ind_update[ind_traced]] = snaps[isnap].forward['Link'][shi_update[ind_traced]]


        print(" done ({:.3f} sec.)" .format(time.time()-stime))
        print("   --> {:d} subhaloes are excluded because most of their mass is stolen..."
              .format(len(ind_traced_corrupted)))


        print("   now looping through individual galaxies...")  

        # ===============================================================
        # Part I done. Now for the actual work - writing out the tracers!
        # ===============================================================

        # Initialise output vectors (guess number of tracers)
        tracerIDs = np.zeros((main.maxgal+1)*100, dtype = int)-1
        tracerMass = np.zeros((main.maxgal+1)*100)-1
        tracerPType = np.zeros((main.maxgal+1)*100, dtype = np.int8)-1
        tracerOffset = np.zeros(main.maxgal+2, dtype = int)

        # Initialise counters in case we need to expand the vectors later
        updateCounter = 0
        nCurrAllocSpace = (main.maxgal+1)*100

        dotgap = next_dot = (main.maxgal+1)/50
        estime = time.time()

        # Some (possibly temporary) time counters
        n_zero_tracers = 0
        n_notyet_existing = 0
        tc_preborn = 0
        tc_setup = 0
        tc_read = 0
        tc_expand = 0
        tc_add = 0

        print("   ", end = "")


        # Unfortunately, there is no real way around a loop here...
        for igal in range(main.maxgal+1):

            ta = time.time()

            if igal > next_dot:
                print(".", end = '', flush = True) 
                next_dot += dotgap
                
            lastSeenSnap = self.lastSeenList[igal]
            
            if lastSeenSnap < 0:
                tracerOffset[igal+1] = tracerOffset[igal]
                n_notyet_existing += 1
                tc_preborn += (time.time()-ta)
                continue;

            if lastSeenSnap >= nsnap:
                set_trace()
                tracerOffset[igal+1] = tracerOffset[igal]
                n_notyet_existing += 1
                tc_preborn += (time.time()-ta)
                continue;


            refLength = self.tracLengthList[igal]
            tracerInd = self.tracLinkIndList[igal]
            
            if refLength == 0:
                tracerDir = 'Local'
            else:
                tracerDir = 'Level' + str(refLength)
            
            if refLength > 5:
                print("WTH is refLength > 5???")
                set_trace()

            if refLength + lastSeenSnap >= nsnap:
                print("Inconsistent ref-length")
                set_trace()

            # Now we need to work out how many tracer particles there are...
            # For this, we Need to load actual data from disk...
            h5dir = tracerDir + '/Snapshot_' + str(lastSeenSnap).zfill(3) + '/SnipshotIDs/'
            
            if self.isReadFlag[lastSeenSnap, refLength] == 0:
                if refLength + lastSeenSnap >= nsnap:
                    print("abt to do sth stupid...")
                    set_trace()

                self.IDlists[lastSeenSnap][refLength] = yb.read_hdf5(spiderloc, h5dir + 'IDs')
                self.Masslists[lastSeenSnap][refLength] = yb.read_hdf5(spiderloc, h5dir + 'Mass')
                self.Typelists[lastSeenSnap][refLength] = yb.read_hdf5(spiderloc, h5dir + 'PartType')
                self.tracerOffsets[lastSeenSnap][refLength] = yb.read_hdf5(spiderloc, h5dir + 'Offset')
                
                self.isReadFlag[lastSeenSnap, refLength] = 1

                
            tracOffsetThis = self.tracerOffsets[lastSeenSnap][refLength][tracerInd]
            tracOffsetNext = self.tracerOffsets[lastSeenSnap][refLength][tracerInd+1]

            numTracers = tracOffsetNext-tracOffsetThis

            tb = time.time()
            tc_setup += (tb-ta)

            if (numTracers) <= 0:
                tracerOffset[igal+1] = tracerOffset[igal]
                n_zero_tracers += 1
                continue

            tracIDs = self.IDlists[lastSeenSnap][refLength][tracOffsetThis:tracOffsetNext]
            tracMass = self.Masslists[lastSeenSnap][refLength][tracOffsetThis:tracOffsetNext]
            tracPType = self.Typelists[lastSeenSnap][refLength][tracOffsetThis:tracOffsetNext]
                
                
            tc = time.time()
            tc_read += (tc-tb)

            # Need to expand the holding lists in case we ran out of space
            if tracerOffset[igal]+numTracers > nCurrAllocSpace:


                tracerIDs = np.concatenate((tracerIDs, np.zeros((main.maxgal+1)*100, dtype = int)-1))
                tracerMass = np.concatenate((tracerMass, np.zeros((main.maxgal+1)*100)-1))
                tracerPType = np.concatenate((tracerPType, np.zeros((main.maxgal+1)*100, dtype = np.int8)-1))
                nCurrAllocSpace += ((main.maxgal+1)*100)

                updateCounter += 1
                print("")
                print("    --> needed to expand ID list (curr size = {:d} = {:.1f} times Ngal)" 
                      .format(tracerIDs.shape[0], tracerIDs.shape[0]/(main.maxgal+1)))
                
            td = time.time()
            tc_expand += (td-tc)
            
            tracerIDs[tracerOffset[igal]:tracerOffset[igal]+numTracers] = tracIDs
            tracerMass[tracerOffset[igal]:tracerOffset[igal]+numTracers] = tracMass
            tracerPType[tracerOffset[igal]:tracerOffset[igal]+numTracers] = tracPType
            tracerOffset[igal+1] = tracerOffset[igal] + numTracers

            te = time.time()
            tc_add += (te-td)

            #if igal >= (maxgal+1)/20:
            #    break
      
        # Done looping through galaxies. Just write output

        print("")
        print("   ---> extracted {:d} particles in {:.3f} sec."
              .format(tracerOffset[-1], time.time()-estime))
        
        ttot = time.time()-estime
        #print("        (pre-born = {:.3f} sec. = {:.2f}%)"
        #      .format(tc_preborn, tc_preborn/ttot*100))
        print("        (setup =    {:.3f} sec. = {:.2f}%)"
              .format(tc_setup, tc_setup/ttot*100))
        print("        (read =     {:.3f} sec. = {:.2f}%)"
              .format(tc_read, tc_read/ttot*100))
        #print("        (expand =   {:.3f} sec. = {:.2f}%)"
        #      .format(tc_expand, tc_expand/ttot*100))
        #print("        (add =      {:.3f} sec. = {:.2f}%)"
        #      .format(tc_add, tc_add/ttot*100))

        print("   ---> average = {:.2f} ({:.2f}) particles per (existing) galaxy"
              .format(tracerOffset[-1]/(main.maxgal+1), tracerOffset[-1]/(main.maxgal+1-n_notyet_existing)))

        print("   ---> {:d} galaxies (={:.2f}%) have zero tracers"
              .format(n_zero_tracers, n_zero_tracers/(main.maxgal+1-n_notyet_existing)*100))

        


        print("   ... writing output...", end = "")
        
        tracerIDs = tracerIDs[:tracerOffset[-1]]
        tracerMass = tracerMass[:tracerOffset[-1]]
        tracerPType = tracerPType[:tracerOffset[-1]]

        h5dir = 'Tracers/Snapshot_' + str(isnap).zfill(3) + '/'

        yb.write_hdf5(tracerOffset, outloc, h5dir + 'Offset', comment = "Galaxy offset list for tracers. The tracers for galaxy i are located between index Offset[i] and Offset[i+1] in the IDs/Mass/PartType datasets.")

        yb.write_hdf5(tracerIDs.astype(np.uint64), outloc, h5dir + 'IDs', comment = "Tracer particle IDs for the interval between snapshot {:d} and {:d}. The partition table into individual galaxies is provided in the dataset 'Offset'" .format(isnap, isnap+1))
        yb.write_hdf5(tracerMass, outloc, h5dir + 'Mass', comment = "Tracer particle masses (at snapshot {:d}) for the interval between snapshot {:d} and {:d}. The partition table into individual galaxies is provided in the dataset 'Offset'" .format(isnap, isnap, isnap+1))

        yb.write_hdf5(tracerPType, outloc, h5dir + 'PartType', comment = "Tracer particle types (at snapshot {:d}) for the interval between snapshot {:d} and {:d}. The partition table into individual galaxies is provided in the dataset 'Offset'" .format(isnap, isnap, isnap+1))

        # Final bit (added 13-Feb-18): remove no-longer-needed tracer info
        # (to reduce memory footprint)

        i_rem = isnap-5
        if i_rem >= 0:
            for ilen_remove in range(1, 6):
                self.IDlists[i_rem][ilen_remove] = []
                self.Masslists[i_rem][ilen_remove] = []
                self.Typelists[i_rem][ilen_remove] = []
                self.tracerOffsets[i_rem][ilen_remove] = []
                self.isReadFlag[i_rem][ilen_remove] = 0
                        
        print(" done.")
        print(" <> Tracer extraction for snapshot {:d} took {:.3f} sec."
              .format(isnap, time.time()-stime))


class BlockTable: 

    """
    Set up and update a blocking table to keep track of forbidden links for 
    galaxies that had to be long-linked.

    Added 9-May-2018
    """

    def __init__(self):
        
        self.btype = [('gal_target', 'i4'), ('gal_blocked', 'i4')]
        self.table = np.zeros(0, dtype = self.btype)
        self.nOffset = 0
        self.nEntries = 0

    def add(self, target, blocked):

        """
        Add pairs of target and blocked galaxies, but do not re-order table yet.
        """ 

        n_target = len(target)
        n_blocked = len(blocked)

        if n_target != n_blocked:
            print("Inconsistent numbers in addition to blocktable!")
            set_trace()

        new_table = np.zeros(n_target, dtype = self.btype)
        new_table['gal_target'] = target
        new_table['gal_blocked'] = blocked

        self.table = np.concatenate((self.table, new_table))


    def reorder(self, maxgal, update_offset = True):

        """
        Re-order the table and update offset list, if desired.
        """

        if self.table.shape[0] > 0:
            self.table.sort(order = ['gal_target', 'gal_blocked'])
        
            self._remove_duplicates()

            if update_offset:
                self._build_offset_list(maxgal)

        self.nEntries = self.table.shape[0]
        
    def _remove_duplicates(self):

        """
        Internal function to remove duplicate entries from the list
        """

        len_original = self.table.shape[0]
        keep_list = np.zeros(len_original, dtype = np.int8)+1

        # Check whether entries in target AND blocked are same as one before
        ind_repeat = np.nonzero((self.table['gal_target'][1:] == self.table['gal_target'][:-1]) & (self.table['gal_blocked'][1:] == self.table['gal_blocked'][:-1]))[0]

        keep_list[ind_repeat+1] = 0

        ind_toKeep = np.nonzero(keep_list == 1)[0]
        self.table = self.table[ind_toKeep]


    def _build_offset_list(self, maxgal): 

        """
        Build an offset list to index the gal_target part of self.table
        """

        max_targ = np.max(self.table['gal_target'])
        
        if max_targ > maxgal+1:
            print("ERROR: BlockTable contains galaxy with higher ID than ngal+1?")
            set_trace()

        numbins = maxgal+1  # +1 because if maxgal = 10, there are 11...
        self.offset = np.zeros(numbins+1, dtype = int) # ... and another one extra for coda
        
        # ... and here +1 is necessary to get proper behaviour out of histogram
        # N.B.: final [0] only selects actual histogram, not bin edges
        histogram = np.histogram(self.table['gal_target'], bins = max_targ+1, range = [0, max_targ+1])[0]
        self.offset[1:1+len(histogram)] = np.cumsum(histogram)
        
        # If there are more offset_list entries than histogram entries, fill all those 
        # with the coda value
        self.offset[1+len(histogram):] = self.offset[len(histogram)]

        self.nOffset = maxgal+1


    def query(self, gal):

        """
        Query the table for all galaxies blocked for input target galaxy.
        """

        if gal >= self.nOffset:
            return np.zeros(0, dtype = np.int32)

        return self.table['gal_blocked'][self.offset[gal]:self.offset[gal+1]]
        
        

        
        

        
# ==========================
# ACTUAL PROGRAM STARTS HERE
# ==========================

pstime = time.time()
main = Status()

# Initialize PSS for exact history compensation
if args.exact_history:
    pss = PSS()

# Initialize blocktable:
if args.blockAllRevenge:
    blocktable = BlockTable()

for jsnap in range(nsnap):

    print("")
    print("")
    print("---------------------------")
    print("Processing snapshot {:d}" .format(jsnap))
    print("---------------------------")
    print("")
    print("")
    
    stime = time.time()

    main.update_snap(jsnap)

    # Need to update the PSS particle table even for first snap
    if args.exact_history:
        pss.update(jsnap)

        # Don't need to load masses for isnap==1, cause there's not 
        # history compensation yet.
        if jsnap > 1:
            pss.load_mass(jsnap)
            print("")
        
    # The first snapshot is VERY easy: each subhalo starts its own galaxy
    if jsnap == 0:

        senderOffset_ij = yb.read_hdf5(spiderloc, 'Level1/Snapshot_' + str(jsnap).zfill(3) + '/SenderOffset')
        nHaloesA = len(senderOffset_ij)-1

        # Initialise current snapshot info, and append it to the 
        # (currently still empty) list:
        snaps.append(Snapshot(0, nHaloesA))
        snaps[0].initialize_new_galaxies()

        xlist.append(np.zeros(0, dtype = int))
        allLinks.append(np.zeros(0, dtype = int))

        # Initialise the total exchange list
        exchange = ExchangeHistory()
        partialExchangeHistory.append(copy(exchange))

        # New bit added 11-May-18: full exchange history to speed things up
        if args.quickHistory:
            fullExchange = ExchangeHistory()

        print("")
        print("<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>")
        print("<<<>>> Snapshot {:d} took {:.3f} sec <<<>>>" .format(main.jsnap, time.time()-stime))
        print("<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>")
        print("")    

        continue

    # -----------------------------------------------------------------

    # ---------------------------------------------------
    # Now we're dealing with 'real' snapshots (1 onwards)
    # ---------------------------------------------------

    # ----------------------------------------------------
    # Nomenclature:
    # - jsnap is the TARGET snap of currently considered links
    # - current snapshot (jsnap) is 'j'
    # - previous snapshot (jsnap-1) is 'i' / 'isnap'
    # - subscript 'ij' means 'from i to j' (and vice versa)
    # -----------------------------------------------------

    # ----------------------------------------------------------------

    
    # NB: we need to set up a temporary list to hold long links
    # This is because forward-props can only be updated AFTER ALL
    # long-links are done, in case of long-over-long overwrite

    allLongLinks = []
    allLongLinks.append([]) # so we can index using nback
    
    
    # ======================== Step 1 =================================
    # -----------------------------------------------------------------
    # ------ Baseline tracing (i.e. Level1 to previous snapshot) ------
    # -----------------------------------------------------------------
    # =================================================================


    print("Snapshot {:d} -- analysing level-1 links ({:d} --> {:d})..." 
          .format(main.jsnap, main.isnap, main.jsnap))
    
    # Read and post-process short-links, including finding permitted ones
    links_ij = Links(main.isnap, 1, pre = "   ")
    allLinks.append(links_ij)
    allLongLinks.append(links_ij)

    # Initialise current snapshot info
    snaps.append(Snapshot(jsnap, links_ij.nHaloesB)) 
    curr_snap = snaps[main.jsnap]  # Shortcut, for compatibility

    # Add CR0 target info already:
    ind_sendsLink = np.nonzero(links_ij.sendOffset[1:] > links_ij.sendOffset[:-1])[0]
    snaps[main.isnap].forward['SHI_CR0'][ind_sendsLink] = links_ij.links['receiver'][links_ij.sendOffset[ind_sendsLink]] 

    # Also (new) load next short link set, for improving tracing consistency
    if jsnap < nsnap-1:

        print("   ... loading and pre-selecting next L1 links...", end = "")
        jkstime = time.time()

        links_jk = Links(jsnap, 1, pre = "   ", norm = False, compensate = False, find_permitted = True, filter_long = False, quiet = True, blockAllRevenge = False)
        probMatch_kj, probConn_jk = links_jk.select_links(selectionPriority, quiet = True)
        probMatch_jk = np.zeros(links_jk.nHaloesA, dtype = int)-1
        ind_match_kj = np.nonzero(probMatch_kj >= 0)[0]
        probMatch_jk[probMatch_kj[ind_match_kj]] = ind_match_kj
        
        print(" done ({:.3f} sec.)" .format(time.time()-jkstime))
    
    snaps[main.jsnap].reverse['NumShortLinks'] = links_ij.recvOffset[1:]-links_ij.recvOffset[:-1]
    snaps[main.isnap].forward['NumShortLinks'] = links_ij.sendOffset[1:]-links_ij.sendOffset[:-1]

    if flag_computeFree:
        snaps[main.jsnap].reverse['FreeMassFracPrev'] = links_ij.freeRecv['Mass']
        snaps[main.jsnap].reverse['FreeNumFracPrev'] = links_ij.freeRecv['Num']
        snaps[main.isnap].forward['FreeFracNext'] = links_ij.freeSend['All']
        snaps[main.isnap].forward['FreeCFracNext'] = links_ij.freeSend['Core']

    # Find preliminary connections based on short-link network.
    # N.B.: the returned connections are already adjusted to 
    # be indices into the *full* links, not the permitted subset

    if args.consistent_selection:
        match_ji, connection_ji = links_ij.select_links_consistently(selectionPriority)
    else:
        match_ji, connection_ji = links_ij.select_links(selectionPriority)
    
    ind_matched_ji = np.nonzero(match_ji >= 0)[0]
    
    snaps[main.jsnap].update_reverse(links_ij, update = ind_matched_ji, ind_link = connection_ji[ind_matched_ji], connection = True)

    snaps[main.isnap].update_forward(links_ij, update = match_ji[ind_matched_ji], ind_link = connection_ji[ind_matched_ji], connection = True)

    # Set forward flags NOW, because these are meant to represent the state
    # of affairs AFTER THE SHORT-LINK PHASE
    snaps[main.isnap].set_forward_flags()

    print("<> Level-1 link selection took {:.3f} sec." 
          .format(time.time()-stime))
    print("")

    utime = time.time()


    # ======================== Step 2 =====================================
    # ---------------------------------------------------------------------
    # --- Connect "loose ends" using long-links ---------------------------
    # ---------------------------------------------------------------------
    # =====================================================================

    if args.long_links and main.jsnap >= 2:
        
        print("Snapshot {:d} -- connecting loose ends via long links..." .format(main.jsnap))

        llstime = time.time()

        # NB: 'iorig' refers to the snapshot where the galaxy was LAST 
        # identified. So we start from two back.
 
        for main.iorig in range(main.jsnap-2, -1, -1):
            iorig_stime = time.time()

            main.nback = main.jsnap - main.iorig  # Link length being tested
            if main.nback > 5:
                break  # because we only store long-links up to Level5

            print("")
            print("... level {:d} ({:d} --> {:d})..." 
                  .format(main.nback, main.iorig, main.jsnap))
            
            # Load links and find eligible ones
            links_oj = Links(main.iorig, main.nback, pre = "   ")
            allLongLinks.append(links_oj)

            """
            # Clear old Temp entries from haloes in orig snap!
            snaps[main.iorig].forward['SHITemp'] = -1
            snaps[main.iorig].forward['LengthTemp'] = 0
            """

            # Select as usual
            if args.consistent_selection:
                match_jo, connection_jo = links_oj.select_links_consistently(selectionPriority)
            else:
                match_jo, connection_jo = links_oj.select_links(selectionPriority)
            
            # Find subhaloes that could be matched in this way
            # (on both sides)

            ind_matched_jo = np.nonzero(match_jo >= 0)[0]
            
            # Bit added 14-May-18 to deal with 'bridged' links instead of 
            # creating orphans:
            connMode = np.zeros(len(match_jo), dtype = np.int8)
            connMode[ind_matched_jo] = 1

            if len(ind_matched_jo) > 0:

                # Two things need doing BEFORE we apply these new
                # matches to the subhalo tables:
                
                # First, deal with cases where a (current-level) long-link
                # has "overridden" a previous (long/short) link to the same
                # receiver
                snaps[main.jsnap].override(match_jo)

                # 1b) [14-May-2018]: check if new LL just 'reactivates'  
                #       existing SL (only for main.nback == 2)
                #       For these, set connMode -> 2

                if main.nback == 2 and args.bridgeInOrphans:

                    # Start (initially empty) list of SLs to be reactivated
                    ind_reactivateSL = np.zeros(0, dtype = int)
                    ind_matched_caseB = np.nonzero(links_oj.sender_mask[links_oj.links['sender'][connection_jo[ind_matched_jo]]] == 2)[0]
                    if len(ind_matched_caseB) > 0:
                        
                        # Find 'pole' SH = the one in the middle that 
                        # would normally become an orphan:

                        llSender = links_oj.links['sender'][connection_jo[ind_matched_jo[ind_matched_caseB]]]
                        llReceiver = links_oj.links['receiver'][connection_jo[ind_matched_jo[ind_matched_caseB]]]

                        if np.count_nonzero(llReceiver != ind_matched_jo[ind_matched_caseB]) > 0:
                            print("Inconsistent receivers in bridging snaps...?")
                            set_trace()


                        if np.count_nonzero(snaps[main.jsnap-2].forward['Length'][llSender] != 1) > 0: 
                            print("Inconsistent length in bridging snaps...?")
                            set_trace()
                            
                        sh_pole = snaps[main.jsnap-2].forward['SHI'][llSender] 
                        
                        for iipole, ish_pole in enumerate(sh_pole):
                            slinks_this = links_ij.sendOffset[ish_pole]+np.arange(links_ij.sendOffset[ish_pole+1]-links_ij.sendOffset[ish_pole], dtype = int)
                            subind_toReactivate = np.nonzero(links_ij.links['receiver'][slinks_this] == llReceiver[iipole])[0]

                            if len(subind_toReactivate) > 1:
                                print("Multiple links between SH pair?!?")
                                set_trace()

                            if len(subind_toReactivate) == 1:
                                connMode[ind_matched_jo[ind_matched_caseB[iipole]]] = 2
                                ind_reactivateSL = np.concatenate((ind_reactivateSL, slinks_this[subind_toReactivate]))
                    
                    if len(ind_matched_caseB) > 0:
                        print("      --> bridged {:d} galaxies instead of making them orphans (={:.2f}%)"  .format(len(ind_reactivateSL), len(ind_reactivateSL)/len(ind_matched_caseB)*100))
                    else:
                        print("      --> no case-B long-links.")


                # Second, deal with cases where a (current-level) long-link
                # was a "Case-B" link from a subhalo that already had a 
                # (short-linked) descendant. In these cases, the descendant
                # is now an "orphan"
                snaps[main.iorig+1].make_orphans(match_jo, connMode)

                # NOW update REVERSE galaxy table to reflect new connections

                subind_matchedViaLL = np.nonzero(connMode[ind_matched_jo] == 1)[0]
                snaps[main.jsnap].update_reverse(links_oj, update = ind_matched_jo[subind_matchedViaLL], ind_link = connection_jo[ind_matched_jo[subind_matchedViaLL]], connection = True)

                # And finally, reactivate the appropriate short links:
                if main.nback == 2:
                    ind_recvBridge = np.nonzero(connMode == 2)[0]
                    if len(ind_recvBridge) > 0:
                        snaps[main.jsnap].update_reverse(links_ij, update = ind_recvBridge, ind_link = ind_reactivateSL, connection = True) 
                        snaps[main.isnap].update_forward(links_ij, update = links_ij.links['sender'][ind_reactivateSL], ind_link = ind_reactivateSL, connection = True)

                """
                # Also write forward Len/SHI into a temporary array:
                # We do NOT want to overwrite the actual variables yet,
                # because it may be that the long link is overridden later
                snaps[main.iorig].forward['LengthTemp'][match_jo[ind_matched_jo]] = nback
                snaps[main.iorig].forward['SHITemp'][match_jo[ind_matched_jo]] = ind_matched_jo
                """

                # Last step: update long link threshold, if desired:
                
                if args.protectWithLongLinks:
                    snaps[main.jsnap].update_longLink_threshold(links_oj)


            print("  ---> Level-{:d} long-link selection took {:.3f} sec." 
                  .format(main.nback, time.time()-iorig_stime))


        print("")
        print("===> Matched total of {:d} galaxies by long-links ({:.3f} sec.)."
              .format(np.count_nonzero(curr_snap.reverse['Length'] > 1), time.time()-llstime))

        print("")


    # ======================== Step 3 =====================================
    # ---------------------------------------------------------------------
    # -- Set flags, find new galaxies, write info for new and merged ------
    # ---------------------------------------------------------------------
    # =====================================================================

    print("Snapshot {:d} -- bookkeeping..." .format(main.jsnap))
    bstime = time.time()

    # -- NOW (!!!!) we update the galaxy information, and forward-props
    # for long-linked galaxies ----

    # First, update blocktable (if desired):
    if args.blockAllRevenge:

        stime_blockAdd = time.time()
        n_added_to_blocktable = 0

        # Loop over LONG link origin snapshots:
        for snap_orig in range(main.jsnap-5, main.jsnap-1):
            
            if snap_orig < 0:
                continue

            snap_nback = main.jsnap-snap_orig
            sh_this_j = np.nonzero(snaps[main.jsnap].reverse['Length'] == snap_nback)[0]
            sh_this_o = snaps[main.jsnap].reverse['SHI'][sh_this_j]
            
            # Now we also need the CR0 subhalo for this:
            cr0_match_this_o_curr = snaps[snap_orig].forward['SHI_CR0'][sh_this_o]

            if np.count_nonzero(snaps[snap_orig].forward['Length'][sh_this_o] > 1) > 1:
                print("Halo sending long-link is already long-linked...?!?")
                set_trace()


            # And get the corresponding galaxy number
            ind_o_merged = np.nonzero(cr0_match_this_o_curr >= 0)[0]
            if len(ind_o_merged) > 0:
                gal_blocked_this = snaps[snap_orig+1].local['Galaxy'][cr0_match_this_o_curr[ind_o_merged]]
                blocktable.add(snaps[snap_orig].local['Galaxy'][sh_this_o[ind_o_merged]], gal_blocked_this)
                n_added_to_blocktable += len(gal_blocked_this)

        # And finally, re-order the blocktable
        blocktable.reorder(update_offset = True, maxgal = main.maxgal)
        print("   ---> added {:d} new entries to blocktable (now containing {:d}, took {:.3f} sec.)..." .format(n_added_to_blocktable, blocktable.nEntries, time.time()-stime_blockAdd))
                            


    # Go through affected snapshots in turn 
    for main.iorig in range(main.jsnap-5, main.jsnap):

        if main.iorig < 0:
            continue
 
        main.nback = main.jsnap-main.iorig
        sh_this_j = np.nonzero(snaps[main.jsnap].reverse['Length'] == main.nback)[0]
        sh_this_o = snaps[main.jsnap].reverse['SHI'][sh_this_j]

        # We need to do two things at each snapshot: 
        # (i) update forward link properties
        # (ii) update galaxy IDs where necessary

        # Forward link update (only for long-links)
        if main.nback >= 2 and len(sh_this_j) > 0:

            snaps[main.iorig].update_forward(allLongLinks[main.nback], update = sh_this_o, ind_link = snaps[main.jsnap].reverse['Link'][sh_this_j], connection = True)

        # Deal with (confirmed) orphans in this snapshot, including giving 
        # then new IDs
        if main.nback < 5:

            sh_new_orphan = np.nonzero(snaps[main.iorig].local["TempNewGalFlag"] == 1)[0]
            
            if len(sh_new_orphan) > 0:
                snaps[main.iorig].orphan_consequences_old(sh_new_orphan, main.iorig)
                snaps[main.iorig].initialize_new_galaxies() 
                snaps[main.iorig].orphan_consequences_new(sh_new_orphan, main.iorig)
                print("   ---> created {:d} new orphan galaxies"
                      .format(len(sh_new_orphan)))



        # Now update galaxies in current snapshot that were linked
        # to currently targetted snap:
        # Any cases where this match was orphanized are already taken care
        # of, and they have their proper IDs

        if len(sh_this_j) > 0:
            snaps[main.jsnap].update_galaxyID(sh_this_j, sh_this_o, main.iorig)
            

    # Now assign new galaxy IDs to (still-)unmatched galaxies in j:
    snaps[main.jsnap].initialize_new_galaxies(strict_check = True)
    
    # And set the reverse flags for subhaloes in current snapshot
    snaps[main.jsnap].set_reverse_flags(match_ji)

    # Check galaxy consistency between snapshots...
    for nback in range(1, 5):
        if main.jsnap-nback < 0:
            break

        snaps[main.jsnap].check_backwards_consistency(allLongLinks[nback])


    # Identify spawned galaxies and write out their choice-0 info
    ind_new = np.nonzero(curr_snap.reverse['Length'] == 0)[0]
    ind_spawned = np.nonzero((curr_snap.reverse['Length'] == 0) & 
                             (curr_snap.reverse['NumShortLinks'] > 0))[0]

    n_matched = np.count_nonzero(curr_snap.reverse['Length'] > 0)
    n_new = len(ind_new)
    
    print("   ---> could match {:d} galaxies in snap {:d} (= {:.2f}%)"
          .format(n_matched, main.jsnap, n_matched/curr_snap.nHaloes*100))
    print("   ---> {:d} new galaxies (= {:.2f}%)"
          .format(n_new, n_new/curr_snap.nHaloes*100))
    print("      ---> (of which {:d} (= {:.2f}%) are spawned)"
          .format(len(ind_spawned), len(ind_spawned)/len(ind_new)*100))
    print("   ---> total number of galaxies so far is {:d}"
          .format(main.maxgal+n_new+1))


    linkind_new = links_ij.links['sortByRecv'][links_ij.recvOffset[ind_spawned]] 
    
    # Now update entries with choice-0 link. This also sets
    # the link index itself.
    curr_snap.update_reverse(links_ij, ind_spawned, ind_link = linkind_new, connection = False)

    # Uncompensated choice should always be zero, check this is the case...
    if curr_snap.reverse['UncompensatedChoice'][ind_spawned].max() > 0:
        print("Unexpected choice encoutered in new galaxies...")
        set_trace()
        
    # Check if any of these are spectres, and if yes, set warning flag
    subind_spectre = np.nonzero((curr_snap.reverse['FreeMassFracPrev'][ind_spawned] < 0.5) | (curr_snap.reverse['FreeNumFracPrev'][ind_spawned] < 0.5))[0]
    curr_snap.add_flag(ind_spawned[subind_spectre], "SPECTRE")
    print("   ---> identified {:d} spectre galaxies (={:.2f}% of new)" 
          .format(len(subind_spectre), len(subind_spectre)/len(ind_new)*100))

    # Also make a record how many galaxies a sub has (predominantly) spawned:
    # Note that the reverse links for spawned galaxies have already updated

    for spawn_source in curr_snap.reverse['SHI'][ind_spawned]:
        if spawn_source < 0:
            print("Unexpected value of spawn_source!")
            set_trace()

        snaps[main.isnap].forward['NumSpawned'][spawn_source] += 1

    ind_hasspawned = np.nonzero(snaps[main.isnap].forward['NumSpawned'] > 0)[0]
    print("   ---> {:d} galaxies have spawned new galaxies, max={:d}..." 
          .format(len(ind_hasspawned), snaps[main.isnap].forward['NumSpawned'].max()))

    subind_spawned_alive = np.nonzero(snaps[main.isnap].forward['Length'][ind_hasspawned] > 0)[0]
    ind_spawned_curr = snaps[main.isnap].forward['SHI'][ind_hasspawned[subind_spawned_alive]]
    curr_snap.add_flag(ind_spawned_curr, "PROG_SPAWNED")

    # N.B.: "Spawned" also covers case of merger onto a new galaxy
    subind_spawned_merged = np.nonzero(snaps[main.isnap].forward['Length'][ind_hasspawned] == 0)[0]
    snaps[main.isnap].add_flag(ind_hasspawned[subind_spawned_merged], "MERGED_ONTO_NEW")

    # And find cases that are spawned from something that does not exist anymore
    subind_new_from_merged = np.nonzero(snaps[main.isnap].forward['Length'][curr_snap.reverse['SHI'][ind_spawned]] == 0)[0]
    curr_snap.add_flag(ind_spawned[subind_new_from_merged], "SPAWNED_FROM_MERGED")
 

    # ---------------
    # MERGED galaxies
    # ---------------

    # Note, this also includes all galaxies that will be long-linked in future
    
    ind_faded = np.nonzero(snaps[main.isnap].forward['Length'] == 0)[0]
    subind_merged = np.nonzero(snaps[main.isnap].forward['NumShortLinks'][ind_faded] > 0)[0]
    ind_merged = ind_faded[subind_merged]

    print("   ---> {:d} galaxies from last snap have disappeared (= {:.2f}%)"
          .format(len(ind_faded), len(ind_faded)/snaps[main.isnap].nHaloes*100))
    print("      ---> (of which {:d} (={:.2f}%) have merged)"
          .format(len(subind_merged), len(subind_merged)/len(ind_faded)*100))

    linkind_merged = links_ij.sendOffset[ind_merged]
    snaps[main.isnap].update_forward(links_ij, ind_merged, ind_link = linkind_merged, connection = False)

    # core rank should always be zero, check this is the case...
    if snaps[main.isnap].forward['CoreRank'][ind_merged].max() > 0:
        print("Unexpected rank encoutered in merged galaxies...")
        set_trace()
    

    # ============================= Step 4 ==============================
    # -------------------------------------------------------------------
    # ---- Update "interaction lists" -----------------------------------
    # ---- i.e. which surviving galaxies exchange particles -------------
    # -------------------------------------------------------------------
    # ===================================================================


    if flag_interaction_list:

        print("")
        print("Snapshot {:d} -- update interaction list..." .format(main.jsnap))
        
        xstarttime = time.time()

        # Create new exchange list for last snapshot interval
        xlist_curr = ExchangeList(links_ij, snaps[main.isnap], curr_snap)
        xlist.append(xlist_curr)

        if np.count_nonzero(xlist_curr.xlist['gal_i'] == xlist_curr.xlist['gal_j']) > 0:
            print("Diagonal elements in xlist??!!")
            set_trace()

        print("   <> done building exchange list ({:.3f} sec.)"
              .format(time.time() - xstarttime))

        # Update total history exchange list

        xutime = time.time()

        # This part is now CHANGED:
        if main.jsnap-args.lim_recent >= 0:
            exchange.update(xlist[main.jsnap-args.lim_recent+1])
            partialExchangeHistory.append(copy(exchange))

        # New bit added 11-May-2018: update separate total exchange list
        if args.quickHistory:
            fullExchange.update(xlist[main.jsnap])

        #print("   <> done updating history list ({:.3f} sec.)"
        #      .format(time.time() - xutime))
                


    print("")
    print("<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>")
    print("<<<>>> Snapshot {:d} took {:.3f} sec." .format(main.jsnap, time.time()-stime))
    print("<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>")
    print("")    


# ===========================================
# ----- End of loop through snapshots -------
# --- Remainder is end-of-program analysis --
# ===========================================



print("")
print("======================================================")
print("")
print("Finished matching links between snapshots")
print("Total number of galaxies is {:d}." .format(main.maxgal+1)) 
print("Computing galaxy and merger tables...")
print("")
print("======================================================")
print("")


# Need to update final few snapshots to exchange list...

print("Add last exchanges to full history list...")
for iorig in range(nsnap-args.lim_recent+1, nsnap):
    if iorig > 0:
        exchange.update(xlist[iorig])
        partialExchangeHistory.append(copy(exchange))
print("done!")

fstime = time.time()

# Compute global galaxy properties
galProps = Galaxies(main.maxgal)

if os.path.isfile(outloc):
    os.rename(outloc, outloc+'.old')

snaps[0].write_flag_list(outloc)

if flag_computeStolenFrac:

    print("")
    print(" --- Compute stolen fraction of subhaloes... --- ")
    print("")

    # 1.) Find all galaxy pairs that exchange particles both ways
    #     (output lists all paired galaxies directly after each other)
    xpairs_full = exchange.find_paired_exchanges()
    numPairs = int(len(xpairs_full)/2)

    print("   ---> found {:d} mutual exchange pairs..." 
          .format(numPairs))

    ind_up = np.arange(numPairs*2, step=2, dtype = int)
    ind_down = ind_up+1

    # 2.) For each pair, determine in each snapshot the stolen mass
    #     (can discard first and last)

    for isnap in range(1, nsnap-1):

        print("      ... snap {:d}..." .format(isnap))
        
        # Retrieve exchange mass for listed pairs
        xpairs_partial = copy(xpairs_full)
        xpairs_partial['mass'][:] = 0
        partialExchangeHistory[isnap].query_pairs(xpairs_partial)
        
        up_prev = xpairs_partial[ind_up]['mass'] - xpairs_partial[ind_down]['mass']
        up_fut = (xpairs_full[ind_up]['mass'] - xpairs_partial[ind_up]['mass']) - (xpairs_full[ind_down]['mass'] - xpairs_partial[ind_down]['mass'])
        
        # Now need to decide which is currently stealing...
        ind_i_stealing = np.nonzero(up_prev < 0)[0]
        ind_j_stealing = np.nonzero(up_prev > 0)[0]
        
        transfer_partial = np.abs(up_prev)
        return_future = np.abs(up_fut)

        stolen_pairwise = np.minimum(transfer_partial, return_future)

        # Combine this into a total stolen mass for each galaxy
        gal_stolen = np.zeros(main.maxgal)

        for iistolen in range(numPairs):
            if up_prev[iistolen] > 0: # j stealing
                gal_stolen[xpairs_partial['gal_j'][ind_up[iistolen]]] += stolen_pairwise[iistolen]
            elif up_prev[iistolen] < 0: # i stealing
                gal_stolen[xpairs_partial['gal_i'][ind_up[iistolen]]] += stolen_pairwise[iistolen]
        
        # Now make a list for all subhaloes at this snap
        sh_stolen = gal_stolen[snaps[isnap].local['Galaxy']]
        snaps[isnap].local['fracStolen'] = sh_stolen / allLinks[isnap].recvMTot

        # Some sanity checks at the end:
        if np.min(snaps[isnap].local['fracStolen']) < -0.000001:
            print("Why is the stolen fraction negative?")
            set_trace()

        if np.max(snaps[isnap].local['fracStolen']) > 1.000001:

            n_g1 = np.count_nonzero(snaps[isnap].local['fracStolen'] > 1.000001)

            print(" ==============================")
            print(" === WARNING: fStolen > 1 =====")
            print(" ===== (max = {:.3f}) ==========" .format(np.max(snaps[isnap].local['fracStolen'])))
            print(" ===== (N_>1 = {:d}) ==========" .format(n_g1))
            print(" ===== (f_>1 = {:.3f} =========" .format(n_g1/snaps[isnap].nHaloes))
            print(" ==============================")


        ind_severely_stolen = np.nonzero(snaps[isnap].local['fracStolen'] > stolen_skip_threshold)[0]
        n_severely_stolen = len(ind_severely_stolen)

        print("        ---> {:d} galaxies (={:.2f}%) are significantly stolen"
              .format(n_severely_stolen, n_severely_stolen/snaps[isnap].nHaloes*100))

        snaps[isnap].add_flag(ind_severely_stolen, "STOLEN")



print("")
print("----------------------------------------------")
print("Writing snapshot data and computing mergers...")
print("----------------------------------------------")
print("")
    
# Write snapshot/subhalo output:
for isnap in range(nsnap):
    snaps[isnap].write(outloc)

    # Compute and write info about mergers
    Mergers(isnap)

# Write galaxy output
galProps.write(outloc)

catID = calendar.timegm(time.gmtime())
catalogueID = yb.read_hdf5_attribute(spiderloc, "Header", "CatalogueID")[0]



yb.write_hdf5_attribute(outloc, "Header", "CatalogueID_SpiderWeb", catalogueID)
yb.write_hdf5_attribute(outloc, "Header", "CatalogueID_PathFinder", catID)

yb.write_hdf5_attribute(outloc, "Header", "Flag_LongLinks", args.long_links)
yb.write_hdf5_attribute(outloc, "Header", "Flag_LowerRanks", args.lower_ranks)
yb.write_hdf5_attribute(outloc, "Header", "Flag_LowerChoice", args.lower_choice)
yb.write_hdf5_attribute(outloc, "Header", "Flag_LimitLinks", args.limit_links)
yb.write_hdf5_attribute(outloc, "Header", "Flag_CompensateHistory", args.compensate_history)
yb.write_hdf5_attribute(outloc, "Header", "Flag_AllowOrphans", args.allow_orphans)
yb.write_hdf5_attribute(outloc, "Header", "Flag_ExtractTracers", args.extract_tracers)
yb.write_hdf5_attribute(outloc, "Header", "Flag_SuppressSpectreDescendants", args.no_spectre_descendants)
yb.write_hdf5_attribute(outloc, "Header", "Flag_TracePreBirth", args.trace_prebirth)
yb.write_hdf5_attribute(outloc, "Header", "Flag_SkipStolen", args.skip_stolen)
yb.write_hdf5_attribute(outloc, "Header", "Flag_ExactHistory", args.exact_history)
yb.write_hdf5_attribute(outloc, "Header", "Flag_ExactHistoryLimited", args.exact_limited)
yb.write_hdf5_attribute(outloc, "Header", "Flag_PreventRevenge", args.prevent_revenge)
yb.write_hdf5_attribute(outloc, "Header", "Flag_ConsistentSelection", args.consistent_selection)
yb.write_hdf5_attribute(outloc, "Header", "Flag_DualHistory", args.dual_history)
yb.write_hdf5_attribute(outloc, "Header", "LimitForRecent", args.lim_recent)
yb.write_hdf5_attribute(outloc, "Header", "Flag_AllowMultipleLonglinks", args.allow_multiple_longlinks)
yb.write_hdf5_attribute(outloc, "Header", "Flag_PreventMinorRevenge", args.exclude_all_linktargs)

yb.write_hdf5_attribute(outloc, "Header", "SimilarityFraction", sim_frac)
yb.write_hdf5_attribute(outloc, "Header", "StolenThreshold", stolen_skip_threshold)
yb.write_hdf5_attribute(outloc, "Header", "SelectionPriority", np.string_(selectionPriority))
yb.write_hdf5_attribute(outloc, "Header", "SpiderWeb_File", np.string_(spiderloc))
yb.write_hdf5_attribute(outloc, "Header", "RunDir", np.string_(rundir))
yb.write_hdf5_attribute(outloc, "Header", "NSnap", nsnap)
yb.write_hdf5_attribute(outloc, "Header", "NumGalaxies", main.maxgal+1)

yb.write_hdf5_attribute(outloc, "Header", "Spider_CoreFraction", yb.read_hdf5_attribute(spiderloc, "Header", "CoreFraction")[0])
yb.write_hdf5_attribute(outloc, "Header", "Spider_MinCoreNumber", yb.read_hdf5_attribute(spiderloc, "Header", "MinCoreNumber")[0])
yb.write_hdf5_attribute(outloc, "Header", "Spider_MaxCoreNumber", yb.read_hdf5_attribute(spiderloc, "Header", "MaxCoreNumber")[0])
yb.write_hdf5_attribute(outloc, "Header", "Spider_CoreLinkLimit", yb.read_hdf5_attribute(spiderloc, "Header", "CoreLinkLimit")[0])
yb.write_hdf5_attribute(outloc, "Header", "Spider_BaryonsInTotal", yb.read_hdf5_attribute(spiderloc, "Header", "Flag_BaryonsInTotal")[0])
yb.write_hdf5_attribute(outloc, "Header", "Spider_BaryonsInCore", yb.read_hdf5_attribute(spiderloc, "Header", "Flag_BaryonsInCore")[0])
yb.write_hdf5_attribute(outloc, "Header", "Spider_MinimumSnipTracerIDs", yb.read_hdf5_attribute(spiderloc, "Header", "MinimumSnipTracerIDs")[0])
yb.write_hdf5_attribute(outloc, "Header", "Spider_FractionSnipTracerIDs_Local", yb.read_hdf5_attribute(spiderloc, "Header", "FractionSnipTracerIDs_Local")[0])
yb.write_hdf5_attribute(outloc, "Header", "Spider_FractionSnipTracerIDs_Link", yb.read_hdf5_attribute(spiderloc, "Header", "FractionSnipTracerIDs_Link")[0])
yb.write_hdf5_attribute(outloc, "Header", "SimLabel", np.string_(yb.read_hdf5_attribute(spiderloc, "Header", "SimLabel")))



## Write out exchange list

if flag_interaction_list:

    tot_xnum = 0

    for isnap in range(1, nsnap):
        tot_xnum += xlist[isnap].xlist.shape[0]

    xlist_full = np.zeros((tot_xnum, 4), dtype = int)
    xlist_full_mass = np.zeros(tot_xnum)

    curr_offset = 0
    for isnap in range(1, nsnap):
        xlist_curr = xlist[isnap].xlist
        if len(xlist_curr) == 0:
            continue

        nx_curr = xlist_curr.shape[0]
        xlist_full[curr_offset:curr_offset+nx_curr, 3] = isnap
        xlist_full_mass[curr_offset:curr_offset+nx_curr] = xlist_curr['mass']
        xlist_full[curr_offset:curr_offset+nx_curr, 0] = xlist_curr['gal_i']
        xlist_full[curr_offset:curr_offset+nx_curr, 1] = xlist_curr['gal_j']
        xlist_full[curr_offset:curr_offset+nx_curr, 2] = xlist_curr['type']
        curr_offset += nx_curr

    yb.write_hdf5(xlist_full, outloc, 'ExchangeIndices', comment = "Galaxies exchanging mass between each other. Index 0: galaxy sending mass; index 1: galaxy receiving mass; index 2: exchange type [0 - between two existing galaxies; 1 - exchange involving new and/or merged galaxy]; index 3: snapshot at the end of the exchange interval (i.e., after the mass was exchanged). The transferred mass is stored in ExchangeMasses.")
    yb.write_hdf5(xlist_full_mass, outloc, 'ExchangeMasses', comment = "Masses exchanged between different galaxies. When and between which galaxies is detailed in ExchangeIndices.")

print("")
print("=============================================================")
print("Computing and writing global information took {:.3f} sec."
      .format(time.time() - fstime))
print("=============================================================")
print("")


if args.extract_tracers:

    print("-------------------------------")
    print("Extracting IDs for snipshots...")
    print("-------------------------------")

    snipIDs = SnipIDs()

    for isnap in range(nsnap):
        snipIDs.extract(isnap)


print("")
print("")

cplstring = "Completed Spiderweb-Pathfinder in {:.2f} sec." .format(time.time() - prog_stime)

print("="*len(cplstring))
print(cplstring)
print("="*len(cplstring))
print("")

tstring = "End timestamp: " + time.strftime("%A, %d %B %Y at %H:%M:%S (NL)", time.localtime())

print('-'*len(tstring))
print(tstring)
print('-'*len(tstring))
print("")



