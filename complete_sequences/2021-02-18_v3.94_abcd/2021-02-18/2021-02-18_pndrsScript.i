yocoLogInfo, "Split night to isolate SCI-CAL sequences";
cc = [59263.16006944444, 59263.164618055554];
oiFitsSplitNight, oiWave, oiVis2, oiVis, oiT3, tsplit=cc;yocoLogInfo, "Ignore bad baseline G1-J2";
startend = [59263.1412958599976, 59263.1578715299984];
station = "*G1-J2*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

yocoLogInfo, "Ignore bad baseline A0-G1";
startend = [59263.1412958599976, 59263.1578715299984];
station = "*A0-G1*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

