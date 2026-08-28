yocoLogInfo, "Ignore bad baseline A0-G1";
startend = [58813.2037847000029, 58813.2292999499987];
station = "*A0-G1*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

yocoLogInfo, "Ignore bad baseline G1-J2";
startend = [58813.2037847000029, 58813.2292999499987];
station = "*G1-J2*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

