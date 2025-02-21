# SpOC: Delivery Schedule

### Advanced Optimization Techniques for a Delivery Scheduling Problem by Using Genetic and Evolutionary Algorithms

Brief description of the Thesis code Implementation:

## Prerequisites

Ensure you have the following installed:
- Python 3.8 and depends on the following libraries:
- pip (Python package installer) Numpy, Random, Json, os, matplotlib.pyplot 
- matplotlib >= 3.4.3
- numpy >= 1.12.0


## Setup

**Clone the repository**:

   ```bash
   git clone https://github.com/shimu5/SpOCDeliveryScheduleProblem.git
   ```
## SpOC Challenge Provided Evaluated Code: 
   ```bash
   spoc_delivery_scheduling.py
   ```
## SpOC Data: 
  ```bash
\data\spoc\scheduling\candidates.txt 
  ```
  
## Proposed Algorithms: 
 
   ### Roulette Wheel selection with Boundary Mutation
  
   ```bash
python ga_roulette_boundary.py
```

   ### Roulette Wheel selection with Swap Mutation

  ```bash
   python ga_roulette_swap.py
   ```

   ### Elitism with Boundary Mutation
   
   ```bash
   python ga_elitism_boundary.py
```

   ### Elit with Swap Mutation

   ```bash
python ga_elitism_swap.py
   ```

   ## Local Search(LS) with Greedy LS applied on Initialize Population on dir Greedy: (Algorithm is incomplete) 
 ```bash
python  ./Greedy/Greedy_RW_Bound.py
```
