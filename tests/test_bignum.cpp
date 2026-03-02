#include <cassert>
#include <iostream>

#define BIGNUM_NO_MAIN
#include "../src/BigNum.cpp"

int main() {
    using namespace mersenne;

    assert(is_prime_exponent(2));
    assert(is_prime_exponent(3));
    assert(!is_prime_exponent(1));
    assert(!is_prime_exponent(9));

    assert(lucas_lehmer(2, false));
    assert(lucas_lehmer(3, false));
    assert(lucas_lehmer(5, false));
    assert(lucas_lehmer(11, false) == false);

    const unsigned cores = runtime::detect_available_cores();
    assert(cores >= 1u);

    std::cout << "All tests passed\n";
    return 0;
}
