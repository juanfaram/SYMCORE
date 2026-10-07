#!/usr/bin/env python3
"""Frozen factorial audit of the three surviving geometry dimensions."""
SURVIVORS=(9,10,14)
SUBSETS=((9,),(10,),(14,),(9,10),(9,14),(10,14),(9,10,14))
STATUS="development_on_opened_hosts"
# No transfer claim unless a fixed subset beats E,h in >=2/3 held-out hosts on ValueCapture
# and does not increase mean regret in those same hosts.
TRANSFER_HOSTS_GTE=2
