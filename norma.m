function n = norma()
    % 0.6; % = 0.8*0.75
    SC_BIAS = 1;
    x2 = 706.0;
    y2 = 198.0;
    x1 = 627.0;
    y1 = 100.0;
    n = SC_BIAS*norm([x2 y2] - [x1 y1]);
end