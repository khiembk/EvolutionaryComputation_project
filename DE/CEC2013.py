import numpy as np

def shift_func(x, shift):
    return x - shift

def rotate_func(x, rotation_matrix):
    return np.dot(rotation_matrix, x)

def asy_func(x, beta):
    idx = np.where(x > 0)
    x[idx] = x[idx] ** (1.0 + beta * np.arange(len(x))[idx] / (len(x) - 1) * np.sqrt(x[idx]))
    return x

def osz_func(x):
    x_osz = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi != 0:
            log_abs = np.log(abs(xi))
            c1, c2 = (10, 7.9) if xi > 0 else (5.5, 3.1)
            sx = np.sign(xi)
            x_osz[i] = sx * np.exp(log_abs + 0.049 * (np.sin(c1 * log_abs) + np.sin(c2 * log_abs)))
        else:
            x_osz[i] = xi
    return x_osz

def sphere_func(x, shift=None, rotate=None):
    if shift is not None:
        x = shift_func(x, shift)
    if rotate is not None:
        x = rotate_func(x, rotate)
    return np.sum(x ** 2)

def rastrigin_func(x, shift=None, rotate=None):
    if shift is not None:
        x = shift_func(x, shift)
    if rotate is not None:
        x = rotate_func(x, rotate)
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def bent_cigar_func(x, shift=None, rotate=None):
    """
    Bent Cigar Function
    f(x) = x[0]^2 + 10^6 * sum(x[1:]^2)
    """
    if shift is not None:
        x = shift_func(x, shift)
    if rotate is not None:
        x = rotate_func(x, rotate)
    return x[0]**2 + 10**6 * np.sum(x[1:]**2)

def discus_func(x, shift=None, rotate=None):
    """
    Discus Function (High Conditioned Ellipsoid)
    f(x) = 10^6 * x[0]^2 + sum(x[1:]^2)
    """
    if shift is not None:
        x = shift_func(x, shift)
    if rotate is not None:
        x = rotate_func(x, rotate)
    return 10**6 * x[0]**2 + np.sum(x[1:]**2)

def dif_powers_func(x, shift=None, rotate=None):
    """
    Different Powers Function
    f(x) = sum(|x_i|^(2 + 4*i/(n-1)))
    """
    if shift is not None:
        x = shift_func(x, shift)
    if rotate is not None:
        x = rotate_func(x, rotate)
    powers = 2 + 4 * np.arange(len(x)) / (len(x) - 1)
    return np.sum(np.abs(x)**powers)

def rosenbrock_func(x, shift=None, rotate=None):
    """
    Rosenbrock Function
    f(x) = sum(100 * (x[i+1] - x[i]^2)^2 + (x[i] - 1)^2)
    """
    if shift is not None:
        x = shift_func(x, shift) * 2.048 / 100  # Shrink search range
    if rotate is not None:
        x = rotate_func(x, rotate)
    x = x + 1  # Shift to origin
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def schaffer_F7_func(x, shift=None, rotate=None):
    """
    Schwefel's F7 Function
    f(x) = sum((sqrt(x[i]^2 + x[i+1]^2))^0.5 * sin(50 * (sqrt(x[i]^2 + x[i+1]^2))^0.2)^2)
    """
    if shift is not None:
        x = shift_func(x, shift)
    if rotate is not None:
        x = rotate_func(x, rotate)
    z = np.sqrt(x[:-1]**2 + x[1:]**2)
    sin_term = np.sin(50 * z**0.2)**2
    return np.sum(z**0.5 * sin_term)**2 / (len(x) - 1)**2

def ackley_func(x, shift=None, rotate=None):
    """
    Ackley's Function
    f(x) = -20 * exp(-0.2 * sqrt(mean(x^2))) - exp(mean(cos(2*pi*x))) + 20 + e
    """
    if shift is not None:
        x = shift_func(x, shift)
    if rotate is not None:
        x = rotate_func(x, rotate)
    a, b = 20, 0.2
    sum_sq = np.mean(x**2)
    cos_sum = np.mean(np.cos(2 * np.pi * x))
    return -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(cos_sum) + a + np.e

def weierstrass_func(x, shift=None, rotate=None, a=0.5, b=3.0, k_max=20):
    """
    Weierstrass Function
    f(x) = sum(sum(a^k * cos(2*pi*b^k*(x_i + 0.5)))) - sum(a^k * cos(2*pi*b^k*0.5))
    """
    if shift is not None:
        x = shift_func(x, shift) * 0.5 / 100  # Shrink search range
    if rotate is not None:
        x = rotate_func(x, rotate)
    k = np.arange(k_max + 1)
    ak = a**k
    bk = b**k
    sum1 = np.sum([ak[j] * np.cos(2 * np.pi * bk[j] * (x + 0.5)) for j in range(k_max + 1)], axis=0)
    sum2 = np.sum([ak[j] * np.cos(2 * np.pi * bk[j] * 0.5) for j in range(k_max + 1)])
    return np.sum(sum1) - len(x) * sum2

def griewank_func(x, shift=None, rotate=None):
    """
    Griewank's Function
    f(x) = sum(x_i^2 / 4000) - prod(cos(x_i / sqrt(i))) + 1
    """
    if shift is not None:
        x = shift_func(x, shift) * 600 / 100  # Shrink search range
    if rotate is not None:
        x = rotate_func(x, rotate)
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_term - prod_term + 1

def step_rastrigin_func(x, shift=None, rotate=None):
    """
    Step Rastrigin Function
    - Adds a floor transformation to Rastrigin's function.
    """
    if shift is not None:
        x = shift_func(x, shift)
    if rotate is not None:
        x = rotate_func(x, rotate)
    x = np.where(np.abs(x) > 0.5, np.floor(2 * x + 0.5) / 2, x)  # Floor transformation
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)
def ellips_func(x, shift=None, rotate=None):
    """
    High Conditioned Ellipsoid Function
    f(x) = sum(10^(6 * (i-1)/(n-1)) * x_i^2)
    """
    if shift is not None:
        x = shift_func(x, shift)
    if rotate is not None:
        x = rotate_func(x, rotate)

    n = len(x)
    coefficients = 10**(6 * np.arange(n) / (n - 1))  # Scaling factors
    return np.sum(coefficients * x**2)

def schwefel_func(x, shift=None, rotate=None):
    """
    Schwefel Function
    f(x) = 418.9829 * len(x) - sum(x_i * sin(sqrt(abs(x_i))))
    """
    if shift is not None:
        x = shift_func(x, shift) * 1000 / 100  # Shrink search range
    if rotate is not None:
        x = rotate_func(x, rotate)
    z = x + 420.9687462275036  # Shift
    return 418.9829 * len(x) - np.sum(z * np.sin(np.sqrt(np.abs(z))))

def katsuura_func(x, shift=None, rotate=None):
    """
    Katsuura Function
    f(x) = prod(1 + i * sum(|2^j * x_i - floor(2^j * x_i + 0.5)|/2^j))^(10/(len(x)^1.2)) - 1
    """
    if shift is not None:
        x = shift_func(x, shift) * 5 / 100
    if rotate is not None:
        x = rotate_func(x, rotate)
    x = x * 100  # Scale x
    temp = 1 + np.arange(1, len(x) + 1) * np.sum(
        [np.abs(2**j * x - np.floor(2**j * x + 0.5)) / (2**j) for j in range(1, 33)], axis=0
    )
    result = np.prod(temp**(10 / len(x)**1.2)) - 1
    return result

def bi_rastrigin_func(x, shift=None, rotate=None, mu0=2.5, d=1.0):
    """
    Lunacek Bi-Rastrigin Function
    """
    s = 1 - 1 / (2 * np.sqrt(len(x) + 20) - 8.2)
    mu1 = -np.sqrt((mu0**2 - d) / s)
    if shift is not None:
        x = shift_func(x, shift) * 10 / 100
    z = np.where(x < 0, -2 * x, 2 * x)  # Adjust for shift
    z_shifted = rotate_func(z, rotate) if rotate is not None else z
    term1 = np.sum((z_shifted - mu0)**2)
    term2 = d * len(x) + s * np.sum((z_shifted - mu1)**2)
    term3 = 10 * (len(x) - np.sum(np.cos(2 * np.pi * z_shifted)))
    return min(term1, term2) + term3

def grie_rosen_func(x, shift=None, rotate=None):
    """
    Griewank-Rosenbrock Function
    """
    if shift is not None:
        x = shift_func(x, shift) * 5 / 100
    if rotate is not None:
        x = rotate_func(x, rotate)
    x = x + 1  # Shift to origin
    term = (x[:-1]**2 - x[1:])**2 + (x[:-1] - 1)**2
    term = 100 * term + 1
    return np.sum(term**2 / 4000 - np.cos(term) + 1)

def escaffer6_func(x, shift=None, rotate=None):
    """
    Expanded Scaffer's F6 Function
    f(x) = sum(0.5 + (sin^2(sqrt(x_i^2 + x_{i+1}^2)) - 0.5) / (1 + 0.001 * (x_i^2 + x_{i+1}^2))^2)
    """
    if shift is not None:
        x = shift_func(x, shift)
    if rotate is not None:
        x = rotate_func(x, rotate)
    z = np.sqrt(x[:-1]**2 + x[1:]**2)
    sin_term = np.sin(z)**2
    denominator = (1 + 0.001 * z**2)**2
    return np.sum(0.5 + (sin_term - 0.5) / denominator)

def cf_cal(x, weights, biases, fits):
    """
    Calculate the composition function value using weighted contributions of component functions.
    """
    
    weights = np.array(weights)
    weights = weights.astype(float)
    fits = np.array(fits)
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        weights[:] = 1 / len(weights)
    else:
        weights /= weight_sum
    return np.sum(weights * (fits + biases))

def cf01(x, shift, rotate, weights, biases):
    fits = [
        rosenbrock_func(x, shift, rotate),
        dif_powers_func(x, shift, rotate),
        bent_cigar_func(x, shift, rotate),
        discus_func(x, shift, rotate),
        sphere_func(x, shift, rotate),
    ]
    return cf_cal(x, weights, biases, fits)

def cf02(x, shift, rotate, weights, biases):
    fits = [
        schwefel_func(x, shift, rotate),
        schwefel_func(x, shift, rotate),
        schwefel_func(x, shift, rotate),
    ]
    return cf_cal(x, weights, biases, fits)

def cf03(x, shift, rotate, weights, biases):
    fits = [
        schwefel_func(x, shift, rotate),
        schwefel_func(x, shift, rotate),
        schwefel_func(x, shift, rotate),
    ]
    return cf_cal(x, weights, biases, fits)

def cf04(x, shift, rotate, weights, biases):
    fits = [
        schwefel_func(x, shift, rotate),
        rastrigin_func(x, shift, rotate),
        weierstrass_func(x, shift, rotate),
    ]
    return cf_cal(x, weights, biases, fits)

def cf05(x, shift, rotate, weights, biases):
    fits = [
        schwefel_func(x, shift, rotate),
        rastrigin_func(x, shift, rotate),
        weierstrass_func(x, shift, rotate),
    ]
    return cf_cal(x, weights, biases, fits)

def cf06(x, shift, rotate, weights, biases):
    fits = [
        schwefel_func(x, shift, rotate),
        rastrigin_func(x, shift, rotate),
        ellips_func(x, shift, rotate),
        weierstrass_func(x, shift, rotate),
        griewank_func(x, shift, rotate),
    ]
    return cf_cal(x, weights, biases, fits)

def cf07(x, shift, rotate, weights, biases):
    fits = [
        griewank_func(x, shift, rotate),
        rastrigin_func(x, shift, rotate),
        schwefel_func(x, shift, rotate),
        weierstrass_func(x, shift, rotate),
        sphere_func(x, shift, rotate),
    ]
    return cf_cal(x, weights, biases, fits)

def cf08(x, shift, rotate, weights, biases):
    fits = [
        grie_rosen_func(x, shift, rotate),
        schaffer_F7_func(x, shift, rotate),
        schwefel_func(x, shift, rotate),
        escaffer6_func(x, shift, rotate),
        sphere_func(x, shift, rotate),
    ]
    return cf_cal(x, weights, biases, fits)


class BenchmarkFunctions:
    def __init__(self, dimension, cf_num=10, shift_data=None, rotation_matrices=None):
        self.dimension = dimension
        self.cf_num = cf_num
        self.OShift = np.zeros((cf_num, dimension)) if shift_data is None else shift_data
        self.M = np.eye(dimension) if rotation_matrices is None else rotation_matrices

    def test_func(self, x, func_num,shift=None, rotate=None):
        weights = [10, 20, 30, 40, 50]  # Example weights
        biases = [0, 100, 200, 300, 400]  #
        if func_num == 1:
            return sphere_func(x, self.OShift[0], self.M)
        elif func_num == 2:
            return rastrigin_func(x, self.OShift[1], self.M)
        elif func_num == 3:
            return bent_cigar_func(x, shift, rotate) - 1200.0
        elif func_num == 4:
            return discus_func(x, shift, rotate) - 1100.0
        elif func_num == 5:
            return dif_powers_func(x, shift, rotate) - 1000.0
        elif func_num == 6:
            return rosenbrock_func(x, shift, rotate) - 900.0
        elif func_num == 7:
            return schaffer_F7_func(x, shift, rotate) - 800.0
        elif func_num == 8:
            return ackley_func(x, shift, rotate) - 700.0
        elif func_num == 9:
            return weierstrass_func(x, shift, rotate) - 600.0
        elif func_num == 10:
            return griewank_func(x, shift, rotate) - 500.0
        elif func_num == 11:
            return rastrigin_func(x, shift, rotate) - 400.0  
        elif func_num == 12:
            return rastrigin_func(x, shift, rotate) - 300.0
        elif func_num == 13:
            return step_rastrigin_func(x, shift, rotate) - 200.0
        elif func_num == 14:
            return schwefel_func(x, shift, rotate) - 100.0
        elif func_num == 15:
            return schwefel_func(x, shift, rotate) + 100.0
        elif func_num == 16:
            return katsuura_func(x, shift, rotate) + 200.0
        elif func_num == 17:
            return bi_rastrigin_func(x, shift, rotate, mu0=2.5, d=1.0) + 300.0
        elif func_num == 18:
            return bi_rastrigin_func(x, shift, rotate, mu0=2.5, d=1.0) + 400.0
        elif func_num == 19:
            return grie_rosen_func(x, shift, rotate) + 500.0
        elif func_num == 20:
            return escaffer6_func(x, shift, rotate) + 600.0
        elif func_num == 21:
            return cf01(x, shift, rotate, weights, biases) + 700.0
        elif func_num == 22:
            return cf02(x, shift, rotate, weights[:3], biases[:3]) + 800.0
        elif func_num == 23:
            return cf03(x, shift, rotate, weights[:3], biases[:3]) + 900.0
        elif func_num == 24:
            return cf04(x, shift, rotate, weights[:3], biases[:3]) + 1000.0
        elif func_num == 25:
            return cf05(x, shift, rotate, weights[:3], biases[:3]) + 1100.0
        elif func_num == 26:
            return cf06(x, shift, rotate, weights[:5], biases[:5]) + 1200.0
        elif func_num == 27:
            return cf07(x, shift, rotate, weights[:5], biases[:5]) + 1300.0
        elif func_num == 28:
            return cf08(x, shift, rotate, weights[:5], biases[:5]) + 1400.0   
        else:
            raise ValueError("Function number not recognized!")

def Test():
    dim = 10
    x = np.random.uniform(-100, 100, dim)
    shift = np.random.uniform(-100, 100, dim)

    rotation = np.eye(dim)  # Replace with actual rotation matrix
    bench = BenchmarkFunctions(dimension=dim, shift_data=shift, rotation_matrices=rotation)

    result = bench.test_func(x, 21, shift=shift, rotate=rotation)  # Case 21: CF01
    print(f"Result for Case 21: {result}")

if __name__ == "__main__":
    Test()    