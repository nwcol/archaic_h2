subsets/


archaic_tree/
Here I'm continuing to fit the archaic tree model. I make a few changes against
the model which was fit on the 8th of November;
    (1) the introgressing modern human branch MHI is removed and replaced with
        a pulse that emanates directly from MH. 
    (2) I consider only one such pulse rather than two. Also the pulse time is
        fixed at 250ka
    (3) NA is freed. Identifiability problems will be rife but this may actually
        allow me to identify them
    (4) for now, I revert the Yoruba deme (the representitive of modern humans)
        to have a single Ne from the ND/MH divergence to the present. 
        a test in subsets/Y showed that adding a third epoch does improve 
        likelihood, but the interstitial Ne blows up to 100k and thus seems
        poorly constrained.

The plan is to run 200x replicates on the cluster with the fmin algorithm and
then refit the highest-ll reps. 



modern_tree/
I considered adding a size change in MH but decided against it for the moment.
I may add this later.