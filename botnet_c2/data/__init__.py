"""botnet_c2 — graph-topology-based C2 node detection on CTU-13 botnet captures.

Graph topology alone — no ports, no payload, no protocol — can identify C2 nodes
in network traffic, and that signal generalizes across botnet families the model
has never seen.

Dataset: CTU-13 (Czech Technical University, 2011)
  13 captures spanning 7 botnet families (Neris, Rbot, Virut, Donbot, Murlo, Nsis, Sogou).
"""

__version__ = "0.1.0"
