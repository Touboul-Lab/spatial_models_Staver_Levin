function val = circ_dist(x, y, L)
    ydrift = y-x;
    ydrift2 = abs((L-abs(ydrift)));
    ydrift(abs(ydrift) > L/2) = ydrift2(abs(ydrift) > L/2);
    val = ydrift;
end