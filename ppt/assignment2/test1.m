sample_f = @sample_f;
[a_es,b_es,k_es] = expsearch(sample_f, 0, 2, 0, 100)
[a_ih,b_ih,l_ih,k_ih] = inthalving(sample_f, a_es, b_es, 0, 0.1)