I am continuing to try to fit trees....

Here I try to verify whether large ancestral Ne ~= 22,000 helps prevent 
pathological parameter behavior (blowup of Yoruba and MH Ne) and improve the fit

Attempts:
from_24-10-31
    This model gives pretty poor fits. The over-estimation of low range H2
    makes me wonder whether a large ancestral Ne is bad for fits and the 
    signal I'm seeing is actually one of a more recent size expansion affecting
    modern human lineages.

expansions
    No fits here, I just wrote some models to get an idea of the histories
    which could produce the Yoruba curve, which has
        H2 r -> 0 ~= 1.5e-6. H2 r -> 0.5 ~1e-6
    seems that ancient (~800 or 900ka) expansion from a somewhat smaller Ne 
    to a constant 30k has some promise

Yoruba_expansion
    Here I fix NY := 3e4 and fit the ancestral Ne, ranging accross fixed TA.
    When NY := 40,000, we infer small NA and the fit is worse. 
    With NY := 20,000, NA is broadly higher than NY- not the behavior we are 
        looking for here.
    NY := 30,000 gives good fits with rather ancient size changes (order 1.5 or 1.7Ma).
        
San_expansion
    This isn't going to be as fine-grained as above; I fix transition times and
    fit the Ne on both sides.
    This looks promising! Will repeat this analysis for Yoruba-- It seems likely
    that a good model could be built with these as anchors by matching up NA/TA.
    likelihood keeps dropping as TA increases...

    Fit NA:
    TA(ka)  Yoruba     San 
    500     16526       16144
    700     15725       15271
    900     14871       14338
    1100    13933       13312
    1300    12885       12163
    1500    11701       10862
    1700    10351       9371
    1900    8796        7697

    
