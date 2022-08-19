import jax
import jax.numpy as jnp
import numpy as np


def generate_recog_data(
    T=2000,
    d=50,
    R=1,
    P=0.5,
    interleave=True,
    multiRep=False,
    xDataVals="+-",
    softLabels=False,
):
    """Generates "image recognition dataset" sequence of (x,y) tuples.
    x[t] is a d-dimensional random binary vector,
    y[t] is 1 if x[t] has appeared in the sequence x[0] ... x[t-1], and 0 otherwise

    if interleave==False, (e.g. R=3) ab.ab.. is not allowed, must have a..ab..b.c..c (dots are new samples)
    if multiRep==False a pattern will only be (intentionally) repeated once in the trial

    T: length of trial
    d: length of x
    R: repeat interval
    P: probability of repeat
    """
    if np.isscalar(R):
        Rlist = [R]
    else:
        Rlist = R

    data = []
    repeatFlag = False
    r = 0  # countdown to repeat
    for t in range(T):
        # decide if repeating
        R = Rlist[np.random.randint(0, len(Rlist))]
        if interleave:
            repeatFlag = np.random.rand() < P
        else:
            if r > 0:
                repeatFlag = False
                r -= 1
            else:
                repeatFlag = np.random.rand() < P
                if repeatFlag:
                    r = R

        # generate datapoint
        if t >= R and repeatFlag and (multiRep or data[t - R][1].round() == 0):
            x = data[t - R][0]
            y = 1
        else:
            if xDataVals == "+-":  # TODO should really do this outside the loop...
                x = 2 * np.round(np.random.rand(d)) - 1
            elif xDataVals.lower() == "normal":
                x = np.sqrt(d) * np.random.randn(d)
            elif xDataVals == "01":
                x = np.round(np.random.rand(d))
            else:
                raise ValueError('Invalid value for "xDataVals" arg')
            y = 0

        if softLabels:
            y *= 1 - 2 * softLabels
            y += softLabels
        data.append((x, np.array([y])))

    return data
