function [a,b,l,k] = inthalving(f,a,b,t,e)
    if nargin < 5
        e = 0.1;
    end

    xm = (a+b)/2;
    l = b-a;
    k = 0;

    while abs(l) > e
        x1 = a + (l/4);
        x2 = b - (l/4);

        if f(x1,t) < f(xm,t)
            b = xm;
            xm = x1;
        elseif f(x2,t) < f(xm,t)
            a = xm;
            xm = x2;
        else
            a = x1;
            b = x2;
        end

        l = b-a;
        k = k+1;
    end
end