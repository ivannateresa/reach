yocoLogInfo, "Split night to isolate SCI-CAL sequences";
cc = [58816.05121527778];
oiFitsSplitNight, oiWave, oiVis2, oiVis, oiT3, tsplit=cc;yocoLogInfo, "Ignore bad baseline A0-G1";
startend = [58815.9987541800001, 58816.0176672099988];
station = "*A0-G1*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

yocoLogInfo, "Ignore bad baseline G1-J2";
startend = [58815.9987541800001, 58816.0176672099988];
station = "*G1-J2*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

yocoLogInfo, "Ignore bad baseline A0-G1";
startend = [58816.0300828300024, 58816.0484427999982];
station = "*A0-G1*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

yocoLogInfo, "Ignore bad baseline G1-J2";
startend = [58816.0300828300024, 58816.0484427999982];
station = "*G1-J2*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

yocoLogInfo, "Ignore bad baseline J2-J3";
startend = [58816.0300828300024, 58816.0484427999982];
station = "*J2-J3*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

