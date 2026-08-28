yocoLogInfo, "Split night to isolate SCI-CAL sequences";
cc = [58814.0408912037];
oiFitsSplitNight, oiWave, oiVis2, oiVis, oiT3, tsplit=cc;yocoLogInfo, "Ignore bad baseline A0-G1";
startend = [58814.0192506599997, 58814.0382075000016];
station = "*A0-G1*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

yocoLogInfo, "Ignore bad baseline G1-J2";
startend = [58814.0192506599997, 58814.0382075000016];
station = "*G1-J2*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

