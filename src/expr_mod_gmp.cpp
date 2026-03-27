#include <gmp.h>

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

// Computes: Mod[Ceiling[N[Exp[(2^n Log[2 + Sqrt[3]])]]], 2^(n+2)-1]
//
// Key identity: Ceiling[(2+√3)^(2^n)] = (2+√3)^(2^n) + (2-√3)^(2^n) = S(n)
// because (2+√3)(2-√3)=1 so (2-√3)^(2^n) ∈ (0,1) and the sum is an integer.
//
// S(n) satisfies the recurrence:  S(0) = 4,  S(k+1) = S(k)^2 - 2
//
// Result = S(n) mod (2^(n+2)-1), computed via n squarings with GMP.
// Memory: O(n) bits — feasible for any n. Time: O(n * M(n)) where M(n) is
// the cost of squaring an n-bit number (GMP uses FFT for large n).
//
// Usage: expr_mod_gmp <n>

// Efficient reduction mod (2^k - 1) using the identity:
//   x mod (2^k - 1) = (x mod 2^k) + (x >> k)   [repeat until < modulus]
// This avoids full division; at most two passes suffice after a squaring.
static void mod_mersenne(mpz_t s, mpz_ptr tmp, const mpz_t mod, unsigned long k) {
    // tmp = lo = s mod 2^k
    mpz_tdiv_r_2exp(tmp, s, k);
    // s = hi = s >> k
    mpz_tdiv_q_2exp(s, s, k);
    // s = lo + hi
    mpz_add(s, s, tmp);
    // After one pass s < 2 * (2^k - 1), so at most one more correction needed.
    if (mpz_cmp(s, mod) >= 0) {
        mpz_sub(s, s, mod);
    }
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: expr_mod_gmp <n>\n";
            return 1;
        }

        const unsigned long n = std::stoul(argv[1]) - 2UL;
        const bool profile = (std::getenv("EXPR_PROFILE") != nullptr);
        using clock = std::chrono::steady_clock;
        using dsec = std::chrono::duration<double>;
        dsec t_mul{0}, t_reduce{0}, t_sub2{0};

        // modulus = 2^(n+2) - 1
        const unsigned long k = n + 2UL;
        mpz_t mod, s, tmp;
        mpz_init2(mod, k + 1UL);
        mpz_init2(s, 2UL * k + 64UL);
        mpz_init2(tmp, k + 64UL);
        mpz_set_ui(mod, 1);
        mpz_mul_2exp(mod, mod, k);
        mpz_sub_ui(mod, mod, 1UL);

        // S(0) = 4
        mpz_set_ui(s, 4);
        // Reduce S(0) mod (2^(n+2)-1) in case n=0 (mod=3, S(0)=4→1)
        if (mpz_cmp(s, mod) >= 0) mpz_sub(s, s, mod);

        // Iterate n times: S = S^2 - 2 mod (2^(n+2)-1)
        const unsigned long report_interval = (n >= 1000000UL) ? 100000UL : 0UL;
        unsigned long next_report = report_interval;
        for (unsigned long i = 0; i < n; ++i) {
            if (report_interval && i == next_report) {
                std::cerr << "  step " << i << " / " << n << "\r";
                next_report += report_interval;
            }

            const auto t0 = profile ? clock::now() : clock::time_point{};
            mpz_mul(s, s, s);        // s = s^2
            if (profile) t_mul += (clock::now() - t0);

            const auto t1 = profile ? clock::now() : clock::time_point{};
            mod_mersenne(s, tmp, mod, k);
            if (profile) t_reduce += (clock::now() - t1);

            // s = (s - 2) mod (2^k - 1)
            const auto t2 = profile ? clock::now() : clock::time_point{};
            if (mpz_cmp_ui(s, 2UL) >= 0) {
                mpz_sub_ui(s, s, 2UL);
            } else {
                // When s is 0 or 1, subtracting 2 wraps modulo (2^k-1).
                // (s - 2) mod M = s + (M - 2)
                mpz_add(s, s, mod);
                mpz_sub_ui(s, s, 2UL);
            }
            if (profile) t_sub2 += (clock::now() - t2);
        }
        if (report_interval) std::cerr << "\n";

        if (profile) {
            const double mul_s = t_mul.count();
            const double red_s = t_reduce.count();
            const double sub_s = t_sub2.count();
            const double tot_s = mul_s + red_s + sub_s;
            if (tot_s > 0.0) {
                std::cerr << "profile: mul=" << mul_s << "s (" << (100.0 * mul_s / tot_s)
                          << "%), reduce=" << red_s << "s (" << (100.0 * red_s / tot_s)
                          << "%), sub2=" << sub_s << "s (" << (100.0 * sub_s / tot_s)
                          << "%), tracked=" << tot_s << "s\n";
            }
        }

        mpz_out_str(stdout, 10, s);
        std::cout << "\n";

        mpz_clears(mod, s, tmp, nullptr);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
