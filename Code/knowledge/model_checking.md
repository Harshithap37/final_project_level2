# Model Checking Overview

Model checking is an automatic technique to verify finite-state systems. 

## Algorithms
- **BMC (Bounded Model Checking)**: unrolls the system for k steps, checks satisfiability.
- **k-induction**: proof by induction on number of steps.
- **LTL (Linear Temporal Logic)**: used to specify temporal properties like safety, liveness.

## Tools
- NuSMV
- SPIN
- UPPAAL

## Pros
- Fully automatic
- Counterexample traces help debugging

## Cons
- State explosion problem
- Requires finite-state abstraction
