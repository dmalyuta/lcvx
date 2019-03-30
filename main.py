"""
Main file which runs the numerical example.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import docking
import plots

def main():
    docking.solve_docking()
    plots.plot_automatica19()

if __name__=='__main__':
    main()