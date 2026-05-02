# Hybrid CNN-BiLSTM with Hypergraph Learning

This repository implements an inductive deep learning model for tropical cyclone 
intensity forecasting. Unlike conventional approaches that process each storm in 
isolation, the model constructs a population-aware hypergraph incidence matrix over the entire 
storm dataset, encoding higher-order group relationships between storms through 
learnable prototype-based hyperedges. Satellite imagery and meteorological metadata 
are processed through separate embedding streams, with storm archetypes discovered 
automatically via gradient-refined prototype vectors. The resulting hypergraph context 
is precomputed and cached, making per-batch inference highly efficient. The model is 
fully inductive — it generalises to unseen storms at inference time without any 
retraining. Experiments are conducted on the TCIR dataset , a public 
benchmark comprising multi-channel satellite imagery and intensity records across 
Atlantic, East Pacific, and West Pacific basins spanning 2003–2016.
