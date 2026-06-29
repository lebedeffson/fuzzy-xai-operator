from fuzzyxai import IntervalFS, HesitantFS, NeutrosophicFS

def test_interval_reduction_loss_positive():
    fs = IntervalFS(lambda x: 0.4, lambda x: 0.8)
    _, delta = fs.reduce_to_f0()
    assert abs(delta - 0.2) < 1e-9

def test_hesitant_reduction_loss():
    fs = HesitantFS(lambda x: [0.2, 0.8])
    _, delta = fs.reduce_to_f0()
    assert abs(delta - 0.3) < 1e-9

def test_neutrosophic_reduction_loss():
    fs = NeutrosophicFS(lambda x: 0.78, lambda x: 0.20, lambda x: 0.64)
    _, delta = fs.reduce_to_f0()
    assert 0.20 <= delta <= 0.22
