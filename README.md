# Queuing at Boston Stadium

Mus Lamia and Lucien Wallace

This repository is the official implementation of _Is the MBTA Match Ready? Assessing Post-Game Service Plans from Foxboro to South Station Using Queuing Theory_

## Requirements

To install requirements:

```setup
pip install scipy
pip install numpy 
pip install matplotlib
```

## Using the py files

To recreate the simpler cumulative plots, run foxboro_original_scenarios.py. foxborotosouthstation.py creates very similar plots but merged together in one image and without certain information callouts.

To experiment with the N(t) model, run foxboro_MG1_simulation and adjust the parameters in the parameters section.

## Interpreting the cumulative plots and queue depth charts

The simpler deterministic plots all have the form "foxboro_original_scenario_[#].png". foxboro_cumulative_diagrams.png is a combined png of all three.

The N(t) cumulative plots and queue depth charts referenced in the report are in pairs with the form:
  "foxboro_cumulative_[identifier].png"
  "foxboro_queue_depth_[identifier].png"

The baseline N(t) plots have no identifier and are thus:
  "foxboro_cumulative.png"
  "foxboro_queue_depth.png"

The rest of the plots were used to test different parameters as defined below:
  
  N1: high N value
  
  N2: low N value
  
  t1: low t value
  
  t2: high t value
  
  h1: low minimum headway
  
  h2: high minimum headway
  
  b: optimized parameters

## Interpreting the cumulative plots and queue depth charts


