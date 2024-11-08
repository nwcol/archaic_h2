I would like to fit a tree-like model of Neandertal demography with or without
migration bewteen the sampled demes. Due to probable identifiability issues,
it seems that the best way to do this is to fix a grid of parameter values for
(1) ancestral Ne, NA and (2) the transition time from ancestral population size
to the Neandertal branch Ne. In my mind this transition represents either the
ancient `OOA` event by which Neandertals separated from other human lineages in 
Africa, or the seperation from Denisovans.


Grid of fixed values:
    NA 12k 14k 16k 18k 20k
    TA 500ka 600ka 700ka 800ka => 20 models

Fit parameters:
NN 
TN 
NA [Altai]
TA
NCV [ancestor of Chag and Vindija]
TCV 
NC 
NV 
mAVC [between Altai and CV]
mCV

I include migration because it seems essential in keeping H2 between the 
Neandertal samples low and close to the empirical curve.
I may do another grid search without migration.