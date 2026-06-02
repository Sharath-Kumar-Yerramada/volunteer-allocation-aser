# Optimizing Volunteer Deployment in Rural India

## Overview
A linear programming approach to allocating teaching volunteer hours 
across Indian districts to maximize foundational learning outcomes.
Uses district-level data from the ASER (Annual Status of Education 
Report) 2024. Course project for Advanced Statistical Methods I, 
BSDS-II, Indian Statistical Institute Bangalore (May 2026).
Team: Anubhav Ray, Antareep Dutta Choudhury, Sharath Kumar Yerramada.

## The Problem
Educational volunteers are scarce. This project frames their deployment 
as a constrained optimization problem: how to distribute a fixed budget 
of volunteer hours across districts within each state to maximize 
expected learning impact, subject to fairness constraints.

## What's in this repo
- `ASER_Volunteer_Optimization_Report.pdf` — full project report
- `district2.py` — Python script for data cleaning, LP formulation, 
  and solving (coming soon)

## Method
- **Data**: ASER 2024 district-level reading and numeracy proficiency
- **Need score**: Composite inverse-proficiency score per district
- **Model**: State-wise Linear Programming (LP) using SciPy HiGHS solver
- **Constraints**: State budget, minimum 1 hr and maximum 5 hrs per district

## Key Results
- Optimized allocations across 537 districts in 26 states
- 395/537 districts at minimum allocation (1 hr); 123/537 at maximum (5 hrs)
- Strong "bang-bang" characteristic typical of LP solutions
- Districts like Koraput (OD), Malkangiri (OD), and West Garo Hills (ML)
  identified as highest-need

## Data Source
ASER Centre 2024 — [asercentre.org](https://asercentre.org)  
CSV files: [github.com/AnubhavRay25/ASER](https://github.com/AnubhavRay25/ASER)

## Tools
Python (pandas, SciPy, matplotlib)
