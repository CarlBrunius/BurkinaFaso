# BurkinaFaso

This repository contains functions, script as well as in- and outdata for predictive random forest modelling associated with the publication:

Buck M, Nilsson LKJ, Brunius C, Dabire RK, Hopkins R and Terenius O. Bacterial associations reveal spatial population dynamics in Anopheles gambiae mosquitoes.

file | description
:--- | :----------
RF-Func.r | Functions to perform rdCV-RF with unbiased variable selection
RF-BurkinaFaso.r | Script to perform predictive RF-classification of mosquitoes by OTU and to produce RF and permutation figures (or reproduce those in the manuscript using data from 'modelData.RData')
metaOtus.RData | OTUs and metadata
modelData.RData | Data from actual modelling in manuscript

Author: Carl Brunius <carl.brunius@slu.se>
