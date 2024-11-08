I have two major goals today: (1) fitting archaic/modern models over a grid
of NA parameters and then trying to link them by estimating `marginal`
divergence times between pairs of modern/archaic lineages, and (2) taking up
the tree inferred on 24-11-04, improving its fit, adding pulses/migrations to
it to test improvements to the fit, and adding more demes (OOA). It would also
be nice to try to retrieve a more recent/realistic archaic/modern divergence 
time.

Modern model; 13 parameters  
    1 transition time from NA to NMH
    4 split times
    7 Nes
    1 migration

Archaic model; 14 parameters
    1 transition time from NA to NND
    3 split times       
    6 Nes
    4 migrations


full_fit/ 
Here I started writing down the entire comprehensive tree-like null model. This 
model almost certainly has too many parameters for all of them to be fit at once
but it may be possible to run partial fits on it-


refit/ 
Here I take and heavily permute the model from 24-11-04 which I called
`from_900ka`- it used archaic/modern trees fit separately with this ancestral
transition time and NA := 16000. I free the ancestral size and fit a large
number of parameters at once in full_fit/, perturbing heavily. 
Will this approach be effective? 


full_models/
Here I write out the topologies and parameters for the two tree-like models
that we wish to infer, and a model called `basis` which is just elements common
to the two.
    
full_models/basis/ 
Questions I want to answer here:
    (1) does adding a MH->Vindija pulse help solve the human-D/human-N 
        divergence problem? 
    (2) how many size changes do Vindija, Denisova need in a model where they 
        are the only sampled archaic demes?
