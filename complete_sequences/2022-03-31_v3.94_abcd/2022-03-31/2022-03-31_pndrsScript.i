yocoLogInfo, "Split night to isolate SCI-CAL sequences";
cc = [59669.32763888889];
oiFitsSplitNight, oiWave, oiVis2, oiVis, oiT3, tsplit=cc;yocoLogInfo, "Ignore bad baseline G1-J2";
startend = [59669.3074772399996, 59669.3256421900005];
station = "*G1-J2*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

yocoLogInfo, "Ignore bad baseline G1-J2";
startend = [59669.3373553499987, 59669.3539914900030];
station = "*G1-J2*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

