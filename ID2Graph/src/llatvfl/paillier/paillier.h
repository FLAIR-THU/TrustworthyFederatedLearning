#pragma once
#include <unordered_map>
#include <random>
#include <cmath>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <cassert>
#include <cfloat>
#include <limits>
#include <boost/math/special_functions/round.hpp>
#include "../utils/mod_inverse.h"
#include "../utils/prime.h"
using namespace std;

struct PaillierPublicKey;
struct PaillierSecretKey;
struct PaillierCipherText;
struct PaillierKeyGenerator;

inline Bint positive_mod(Bint x, Bint y)
{
    Bint divisor = y;
    Bint dividend = x;
    Bint remainder;

    remainder = dividend % divisor;

    if (remainder < 0)
    {
        remainder += divisor;
    }

    return remainder;
}

inline Bint L(Bint u, Bint n)
{
    return (u - 1) / n;
}

struct PaillierPublicKey
{
    Bint n, n2, g, max_val;
    boost::random::uniform_int_distribution<Bint> distr;
    double precision;

    PaillierPublicKey(){};
    PaillierPublicKey(Bint n_, Bint g_, double precision_ = -1)
    {
        n = n_;
        n2 = n * n;
        g = g_;
        max_val = (n / Bint(3)) - Bint(1);
        distr = boost::random::uniform_int_distribution<Bint>(1, n - 1);
        precision = precision_;
    }

    PaillierPublicKey(Bint n_, Bint g_, Bint n2_, double precision_ = -1)
    {
        n = n_;
        n2 = n2_;
        g = g_;
        max_val = (n / Bint(3)) - Bint(1);
        distr = boost::random::uniform_int_distribution<Bint>(1, n - 1);
        precision = precision_;
    }

    void init_distribution()
    {
        distr = boost::random::uniform_int_distribution<Bint>(1, n - 1);
    }

    bool operator==(PaillierPublicKey pk2)
    {
        return (n == pk2.n) && (g == pk2.g);
    }

    bool operator!=(PaillierPublicKey pk2)
    {
        return (n != pk2.n) || (g != pk2.g);
    }

    Bint raw_encrypt(Bint m, Bint r);
    Bint raw_encrypt(Bint m);

    template <typename T>
    PaillierCipherText encrypt(T m);
};

struct PaillierSecretKey
{
    Bint p, q, n, n2, g, lam, mu;
    double precision = -1;

    PaillierSecretKey(){};
    PaillierSecretKey(Bint p_, Bint q_, Bint n_, Bint g_, double precision_ = -1)
    {
        p = p_;
        q = q_;
        n = n_;
        g = g_;

        n2 = n * n;
        lam = lcm(p - 1, q - 1);
        mu = boost::integer::mod_inverse(L(modpow(g, lam, n * n), n), n);
        precision = precision_;
    }

    PaillierSecretKey(Bint p_, Bint q_, Bint n_, Bint g_,
                      Bint n2_, Bint lam_, Bint mu_, double precision_ = -1)
    {
        p = p_;
        q = q_;
        n = n_;
        g = g_;

        n2 = n2_;
        lam = lam_;
        mu = mu_;
        precision = precision_;
    }

    template <typename T>
    T decrypt(PaillierCipherText pt);
};

template <typename T>
struct EncodedNumber
{
    int BASE = 16;
    double precision = -1;

    float LOG2_BASE = log2(BASE);

    PaillierPublicKey pk;
    Bint encoding;
    int exponent;

    EncodedNumber(PaillierPublicKey pk_, T scalar)
    {
        pk = pk_;
        encode(scalar, numeric_limits<int>::infinity());
    }

    EncodedNumber(PaillierPublicKey pk_, T scalar, double precision_, int max_exponent)
    {
        pk = pk_;
        precision = precision_;
        encode(scalar, max_exponent);
    }

    EncodedNumber(PaillierPublicKey pk_, Bint encoding_, int exponent_,
                  double precision_, int max_exponent)
    {
        pk = pk_;
        precision = precision_;
        encoding = encoding_;
        exponent = exponent_;
    }

    void encode(T scalar, int max_exponent)
    {
        if (precision <= 0)
        {
            if (floor(scalar) == scalar)
            {
                exponent = 0;
            }
            else
            {
                int bin_flt_exponent;
                frexp(scalar, &bin_flt_exponent);
                int bin_lsb_exponent = bin_flt_exponent - DBL_MANT_DIG;
                exponent = int(floor(bin_lsb_exponent / LOG2_BASE));
            }
        }
        else
        {
            exponent = int(floor(log(precision) / log(BASE)));
        }

        if (!isinf(max_exponent))
        {
            exponent = min(max_exponent, exponent);
        }

        Bint int_rep = (boost::math::round(Bfloat(scalar) * Bfloat(mp::pow(Bint(BASE), -1 * exponent)))).convert_to<Bint>();
        encoding = positive_mod(int_rep, pk.n);
    }

    T decode()
    {
        Bint mantissa;
        if (encoding <= pk.max_val)
        {
            mantissa = encoding;
        }
        else if (encoding >= (pk.n - pk.max_val))
        {
            mantissa = encoding - pk.n;
        }
        else
        {
            try
            {
                throw overflow_error("overflow detected");
            }
            catch (overflow_error e)
            {
                cerr << e.what() << endl;
            }
        }
        return T(Bfloat(mantissa) * mp::pow(Bfloat(BASE), exponent));
    }

    void decrease_exponent(int new_exponent)
    {
        if (new_exponent > exponent)
        {
            try
            {
                throw range_error("new exponent should be less than the current exponent");
            }
            catch (range_error e)
            {
                cerr << e.what() << endl;
            }
        }

        int factor = pow(BASE, exponent - new_exponent);
        encoding = encoding * factor % pk.n;
        exponent = new_exponent;
    }
};

struct PaillierCipherText
{
    PaillierPublicKey pk;
    Bint c;
    int exponent;
    double precision;

    int BASE = 16;

    PaillierCipherText(){};
    PaillierCipherText(PaillierPublicKey pk_, Bint c_, int exponent_, double precision_ = -1)
    {
        pk = pk_;
        c = c_;
        exponent = exponent_;
        precision = precision_;
    };

    PaillierCipherText decrease_exponent(int new_exponent)
    {
        if (new_exponent > exponent)
        {
            try
            {
                throw range_error("new exponent should be less than the current exponent");
            }
            catch (range_error e)
            {
                cerr << e.what() << endl;
            }
        }

        PaillierCipherText multiplied = *this * pow(BASE, exponent - new_exponent);
        multiplied.exponent = new_exponent;
        return multiplied;
    }

    PaillierCipherText operator+(PaillierCipherText other_ct)
    {
        if (other_ct.pk != pk)
        {
            try
            {
                throw runtime_error("public key does not match");
            }
            catch (runtime_error e)
            {
                cerr << e.what() << endl;
            }
        }

        PaillierCipherText temp_me = PaillierCipherText(pk, c, exponent);
        if (exponent > other_ct.exponent)
        {
            temp_me = decrease_exponent(other_ct.exponent);
        }
        else if (exponent < other_ct.exponent)
        {
            other_ct = other_ct.decrease_exponent(exponent);
        }

        return PaillierCipherText(pk, (temp_me.c * other_ct.c) % temp_me.pk.n2, temp_me.exponent);
    }

    template <typename T>
    PaillierCipherText _add_encoded(EncodedNumber<T> encoded)
    {
        PaillierCipherText temp_me = PaillierCipherText(pk, c, exponent);
        if (exponent > encoded.exponent)
        {
            temp_me = decrease_exponent(encoded.exponent);
        }
        else if (exponent < encoded.exponent)
        {
            encoded.decrease_exponent(exponent);
        }

        Bint encrypted_scalar = temp_me.pk.raw_encrypt(encoded.encoding, 1);
        return PaillierCipherText(pk, (temp_me.c * encrypted_scalar) % temp_me.pk.n2, temp_me.exponent);
    }

    PaillierCipherText operator+(int pt)
    {
        EncodedNumber<int> encoded = EncodedNumber<int>(pk, pt, precision, exponent);
        return _add_encoded(encoded);
    }

    PaillierCipherText operator+(long pt)
    {
        EncodedNumber<long> encoded = EncodedNumber<long>(pk, pt, precision, exponent);
        return _add_encoded(encoded);
    }

    PaillierCipherText operator+(float pt)
    {
        EncodedNumber<float> encoded = EncodedNumber<float>(pk, pt, precision, exponent);
        return _add_encoded(encoded);
    }

    PaillierCipherText operator+(double pt)
    {
        EncodedNumber<double> encoded = EncodedNumber<double>(pk, pt, precision, exponent);
        return _add_encoded(encoded);
    }

    Bint _mul(Bint encoded_pt)
    {
        Bint res;
        if (pk.n - pk.max_val <= encoded_pt)
        {
            Bint neg_c = boost::integer::mod_inverse(c, pk.n2);
            Bint neg_scalar = pk.n - encoded_pt;
            res = modpow(neg_c, neg_scalar, pk.n2);
        }
        else
        {
            res = modpow(c, encoded_pt, pk.n2);
        }
        return res;
    }

    PaillierCipherText operator*(int pt)
    {
        EncodedNumber<int> encoding = EncodedNumber<int>(pk, pt, precision, numeric_limits<int>::infinity());
        Bint mul_with_encoded_pt = _mul(encoding.encoding);
        int new_exponent = exponent + encoding.exponent;
        return PaillierCipherText(pk, mul_with_encoded_pt, new_exponent);
    }

    PaillierCipherText operator*(long pt)
    {
        EncodedNumber<long> encoding = EncodedNumber<long>(pk, pt, precision, numeric_limits<int>::infinity());
        Bint mul_with_encoded_pt = _mul(encoding.encoding);
        int new_exponent = exponent + encoding.exponent;
        return PaillierCipherText(pk, mul_with_encoded_pt, new_exponent);
    }

    PaillierCipherText operator*(float pt)
    {
        EncodedNumber<float> encoding = EncodedNumber<float>(pk, pt, precision, numeric_limits<int>::infinity());
        Bint mul_with_encoded_pt = _mul(encoding.encoding);
        int new_exponent = exponent + encoding.exponent;
        return PaillierCipherText(pk, mul_with_encoded_pt, new_exponent);
    }

    PaillierCipherText operator*(double pt)
    {
        EncodedNumber<double> encoding = EncodedNumber<double>(pk, pt, precision, numeric_limits<int>::infinity());
        Bint mul_with_encoded_pt = _mul(encoding.encoding);
        int new_exponent = exponent + encoding.exponent;
        return PaillierCipherText(pk, mul_with_encoded_pt, new_exponent);
    }

    void self_bliding()
    {
        boost::random::random_device rng;
        Bint r;
        r = pk.distr(rng);
        c = (c * modpow(r, pk.n, pk.n2)) % pk.n2;
    }
};

inline Bint PaillierPublicKey::raw_encrypt(Bint m, Bint r)
{
    Bint g2m_mod_n2;

    if (n - max_val <= m)
    {
        Bint neg_m = n - m;
        Bint neg_c = modpow(g, neg_m, n2);
        g2m_mod_n2 = boost::integer::mod_inverse(neg_c, n2);
    }
    else
    {
        g2m_mod_n2 = modpow(g, m, n2);
    }

    Bint c = (g2m_mod_n2 * modpow(r, n, n2)) % n2;
    return c;
}

inline Bint PaillierPublicKey::raw_encrypt(Bint m)
{
    boost::random::random_device rng;
    Bint r;
    while (true)
    {
        r = distr(rng);
        if (gcd(r, n) == 1)
        {
            break;
        }
    }
    return raw_encrypt(m, r);
}

template <typename T>
inline PaillierCipherText PaillierPublicKey::encrypt(T m)
{
    EncodedNumber<T> encoding = EncodedNumber<T>(*this, m, precision, numeric_limits<int>::infinity());
    Bint c = raw_encrypt(encoding.encoding);
    PaillierCipherText ciphertext = PaillierCipherText(*this, c, encoding.exponent);
    return ciphertext;
}

template <typename T>
inline T PaillierSecretKey::decrypt(PaillierCipherText ct)
{
    if (ct.c <= 0 || ct.c >= (n2))
    {
        try
        {
            throw range_error("pt.c should be (0, n^2)");
        }
        catch (range_error e)
        {
            cerr << e.what() << endl;
        }
    }

    Bint decrypted_encoding_val = (L(modpow(ct.c, lam, n2), n) * mu) % n;
    EncodedNumber<T> encoded = EncodedNumber<T>(PaillierPublicKey(n, g, n2),
                                                decrypted_encoding_val, ct.exponent,
                                                ct.precision, numeric_limits<int>::infinity());
    return encoded.decode();
}