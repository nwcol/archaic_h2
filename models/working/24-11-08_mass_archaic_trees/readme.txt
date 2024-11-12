Here I ran two different grids for the archaic tree null model (in NA and TA) 
on the cluster with the Nelder-Mead fmin algorithm. summary of results;

NA grid

NA14k: attained ll = -3081 without convergence (rep 8). very ancient TA ~ 1.1Mya
    large NND (20k), very large NY (70k) in SOME replicates
    NND inversely proportional to TND...

NA15k: attained ll = -3055, didn't converge (rep 5). In best fits,
    TA ~ 1.0Mya
    TND ~ 600kya
    NND ~ 20k (TND, NND pose IDability issues)
    NY ~ 70k to 90k

NA16k: attained ll = -3037, didn't converge (rep 25). Fits qualitatively not
    very good.
    It seems likely from comparison of reps 25, 15 that Ne of the introgressing
    human lineages (which are parameterized as one) and Neandertal Ne, m, T 
    parameters are highly conflated. 
    TA ~ 950kya
    TND ~ 950kya
    NND ~ 300 (rep 25), or
    TA ~ 970kya
    TND ~ 700kya,
    NND ~ 13k (rep 14)

    pulse_experiment/
        Here I took a replicate with high NMHI (25) and one with low (14)
        and compared expectations when the pulses were removed. The high-NMHI 
        rep is worse when the pulses are removed, as one might expect.

NA17k: rep 12 had ll = -2981 w/o convergence.
    rep     28          12          27          13
    ll      -3015       -2981       -3063       -3012
    TA      900kya      910kya      910kya      921kya
    TND     885kya      897kya      780kya      850kya
    ND      1k          600         6k          3k
    NMHI    12k         13k         11k         19k
    NY      60k         52k         100k (!)    60k

    The shape of the Yoruba curve convinces me that this ancestral Ne is too
    high. 

NA18k: highest lls in reps 1, 21, with very small ND epoch and low NND

NA19k: attained ll = -3200 and converged (rep 12). Best fits have TA ~ 800kya with 
    TND following very closely thereafter and small NND ~ 100 (lower bound).
    Some slightly worse fits allow a longer ND epoch and a reasonable size,
    ~ 2000 (see rep 24). Some converged fits have even longer epochs with 
    largew NND ~ 10k (see rep 21). 
    TA ~ TND ~ 800kya
    NND ~ 100
    NY ~ 50k to 100k (!)
    Fits to the Yoruba curve are remarkably poor- this ancestral Ne gives us
    a lot of polymorphism to work with, so a somewhat smaller NMH is inferred
    and the Yoruba population lacks the history of expansion apparently required
    to produce its shallow H2 curve. Neandertal fits are pretty good.
    Denisovan quite poor.


600ka, 700ka: fits here were generally poor, all ll < -4000 

800ka: rep 16 converged to ll = -3335 with reasonable parameters. Fits
    qualitatively rather poor.

900ka: ll approaching -3000 but unreasonable fits (very small ND epoch, low NND)


Conclusions.
    (1) fixing TA seems a poor approach.
    (2) NA is likely in the range 15-17k~ increasing Ne will probably tend to   
        push the time of human/archaic divergence down
    (3) most of the replicates reported here had not converged. you should run
        them to convergence.