fn = @myfunction;

x10 = (-3:3)';
vals10 = arrayfun(@(t) fn(t,10), x10);
Sol10 = [x10 vals10]

x7=zeros(5,1);
vals7 = arrayfun(@(t) fn(t,7), x7);
Sol7 = [x7 vals7]