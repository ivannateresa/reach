yocoLogInfo, "Ignore bad baseline A0-G1";
startend = [58812.1882022599966, 58812.2257177400024];
station = "*A0-G1*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

yocoLogInfo, "Ignore bad baseline G1-J2";
startend = [58812.1882022599966, 58812.2257177400024];
station = "*G1-J2*";
oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis, base=station, tlimit=startend;

