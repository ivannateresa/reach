yocoLogInfo, "Split night to isolate SCI-CAL sequences";
cc = [59672.312627314815];
oiFitsSplitNight, oiWave, oiVis2, oiVis, oiT3, tsplit=cc;yocoLogInfo, "Ignore bad baseline G1-J2";
startend = [59672.3683864400009, 59672.3848757899977];
station = "*G1-J2*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

