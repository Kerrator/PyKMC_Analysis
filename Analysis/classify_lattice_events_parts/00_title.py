# %% [markdown]
# # Lattice Event Classification — Cross-Simulation Analysis
#
# Loads the reference tables of every run of one composition (default `100Ni`,
# NiAlH potential; set in the Parameters block of the next cell), classifies
# events under 1.2 eV into lattice move types (1NN hop, 2NN hop, exchange,
# concerted), and extracts parameters for an on-lattice KMC model.
#
# The data root comes from `PYKMC_ANALYSIS_DATA_ROOT` (runs laid out as
# `{composition}/{N}vac/T{temp}/{variant}/`).
#
# Uses functions from `lattice_map_events.py` for event classification and
# lattice projection.
