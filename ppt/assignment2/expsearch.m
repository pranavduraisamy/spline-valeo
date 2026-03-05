function [a,b,k] = expsearch(f, x_init, d, t, max_iter)
    if nargin < 4
        max_iter = 100;
    end
    k = 0;
    if f(x_init - abs(d),t) >= f(x_init,t) && f(x_init,t) >= f(x_init + abs(d),t)
        d = abs(d);
    elseif f(x_init - abs(d),t) <= f(x_init,t) && f(x_init,t) <= f(x_init + abs(d),t)
        d = -abs(d);
    else
        disp("choose diff x_init or d")
        return
    end
    x_k = x_init;
    for i = 1:max_iter
        x_k1 = x_k + (2^k) * d;

        if f(x_k1,t) < f(x_k,t)
            k = k + 1;
            x_k = x_k1;
        else
            x_prev = x_k - (2^(k-1)) * d;
            a = min(x_prev, x_k1);
            b = max(x_prev, x_k1);
            return
        end
    end
    disp("max iters")
    a = min(x_prev, x_k1);
    b = max(x_prev, x_k1);
end