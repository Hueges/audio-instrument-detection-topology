---
title: Musical Classifier
emoji: 🏆
colorFrom: gray
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Classification of musical instrument based on topological properties

**Short introduction**: 
Our entry set is collection of tones played on different types of instruments, which we load as time series.
Then we transform them into point clouds from which we construct Vietoris–Rips complex that exibit distinct topological properties for different instruments.  
Computing persistent homology on Vietoris–Rips complexes we can classify tone to the instrument it was played on.  
Visualization.py is just for detailed representation of topology of these tones. For example we can see in added pictures what 
shape trumpet and piano take.  
Podaci.py cleanses and prepares data.  
Train.py splits data, and makes a model.
