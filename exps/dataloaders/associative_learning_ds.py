import numpy as np

f = 0.1
S = 200
U = 2
R = 1
B = 30
dt = 0.5
dus = int(2 / dt)  # time between CS/US
cslen = int(2 / dt)  # CS presentation length
uslen = int(2 / dt)  # US presentation length
T = int(100 / dt)  # trial length
prob_valid = 0.5


def genrandtrials():
    trialinfo = [dict() for bi in range(B)]
    s = np.zeros([T, S, B], dtype=np.float32)
    u = np.zeros([T, U, B], dtype=np.float32)
    rtarg = np.zeros([T, R, B], dtype=np.float32)

    for bi in range(B):
        valence = np.random.choice([-1, 1])
        isvalid = np.random.rand() < prob_valid
        if isvalid:
            s[:, :, bi], u[:, :, bi], rtarg[:, :, bi], trialinfo[bi] = firstordertrial(
                valence, True, False
            )
        else:
            s[:, :, bi], u[:, :, bi], rtarg[:, :, bi], trialinfo[bi] = firstordertrial(
                valence, np.random.randint(2), np.random.randint(2)
            )

    return s, u, rtarg, trialinfo


def firstordertrial(valence, doUS, doC, teststart=65):
    s = np.zeros([T, S])
    u = np.zeros([T, U])
    rtarg = np.zeros([T, R])

    ainds = np.random.choice(S, int(f * S), replace=False)
    cinds = np.random.choice(S, int(f * S), replace=False)
    stimA = np.zeros(S)
    stimA[ainds] = 1
    stimC = np.zeros(S)
    stimC[cinds] = 1

    trialinfo = dict()

    tA = int(np.random.randint(5, 15) / dt)
    s[tA : (tA + cslen), :] = stimA
    tUS = tA + dus

    ttest = int(np.random.randint(teststart, teststart + 10) / dt)
    if doC:
        s[ttest : (ttest + cslen), :] = stimC
    else:
        s[ttest : (ttest + cslen), :] = stimA

    if doUS:
        if valence > 0:
            u[tUS : (tUS + uslen), 0] = 1.0
        else:
            u[tUS : (tUS + uslen), 1] = 1.0

    if doUS and not doC:
        rtarg[ttest : (ttest + cslen), 0] = valence

    trialinfo["type"] = "firstorder"
    trialinfo["tA"] = tA
    trialinfo["tUS"] = tUS
    trialinfo["valence"] = valence
    trialinfo["ttest"] = ttest
    trialinfo["stimA"] = stimA
    trialinfo["stimC"] = stimC
    trialinfo["doUS"] = doUS
    trialinfo["doC"] = doC

    return s, u, rtarg, trialinfo
