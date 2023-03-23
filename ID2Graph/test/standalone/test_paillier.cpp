#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "llatvfl/paillier/paillier.h"
#include "llatvfl/paillier/keygenerator.h"
#include "llatvfl/paillier/serialization.h"
#include "gtest/gtest.h"
using namespace std;

TEST(paillier, PaillierUtilsTest)
{
    ASSERT_EQ(positive_mod(-13, 15), 2);
}

TEST(paillier, PaillierBaseTest)
{
    long long p = 19;
    long long q = 23;
    long long n = p * q;
    long long k = 4;
    long long g = (1 + k * n) % (n * n);

    ASSERT_EQ(L(g, n), k);

    PaillierPublicKey pk = PaillierPublicKey(n, g);
    PaillierSecretKey sk = PaillierSecretKey(p, q, n, g);
    // ASSERT_EQ(sk.lam, 198);

    PaillierCipherText ct_1 = pk.encrypt(3);
    ASSERT_EQ(sk.decrypt<int>(ct_1), 3);
    PaillierCipherText ct_2 = pk.encrypt(8);
    ASSERT_EQ(sk.decrypt<int>(ct_2), 8);

    ASSERT_TRUE(ct_1.pk == ct_2.pk);
    ASSERT_TRUE(!(ct_1.pk != ct_2.pk));

    PaillierCipherText ct_3 = ct_1 + ct_2;
    ASSERT_TRUE(ct_1.pk == ct_3.pk);
    ASSERT_EQ(sk.decrypt<int>(ct_3), 11);

    PaillierCipherText ct_4 = ct_1 + 0;
    ASSERT_TRUE(ct_1.pk == ct_4.pk);
    ASSERT_EQ(sk.decrypt<int>(ct_4), 3);

    PaillierCipherText ct_5 = ct_1 + 3;
    ASSERT_TRUE(ct_1.pk == ct_5.pk);
    ASSERT_EQ(sk.decrypt<int>(ct_5), 6);

    PaillierCipherText ct_6 = ct_1 * 0;
    ASSERT_TRUE(ct_1.pk == ct_6.pk);
    ASSERT_EQ(sk.decrypt<int>(ct_6), 0);

    PaillierCipherText ct_7 = ct_1 * 1;
    ASSERT_TRUE(ct_1.pk == ct_7.pk);
    ASSERT_EQ(sk.decrypt<int>(ct_7), 3);

    PaillierCipherText ct_8 = ct_1 * 3;
    ASSERT_TRUE(ct_1.pk == ct_8.pk);
    ASSERT_EQ(sk.decrypt<int>(ct_8), 9);
}

TEST(paillier, PaillierAdvancedIntegerTest)
{
    PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
    pair<PaillierPublicKey, PaillierSecretKey> keypair = keygenerator.generate_keypair();
    PaillierPublicKey pk = keypair.first;
    PaillierSecretKey sk = keypair.second;

    PaillierCipherText ct_1 = pk.encrypt(1);
    ASSERT_EQ(sk.decrypt<int>(ct_1), 1);
    PaillierCipherText ct_2 = pk.encrypt(2);
    ASSERT_EQ(sk.decrypt<int>(ct_2), 2);
    PaillierCipherText ct_3 = pk.encrypt(123456);
    ASSERT_EQ(sk.decrypt<int>(ct_3), 123456);
    PaillierCipherText ct_4 = ct_3 * 2;
    ASSERT_EQ(sk.decrypt<int>(ct_4), 246912);
    PaillierCipherText ct_5 = ct_4 + ct_2;
    ASSERT_EQ(sk.decrypt<int>(ct_5), 246914);

    long large_positive = 9223372036854775807;
    PaillierCipherText ct_lp = pk.encrypt(large_positive);
    ASSERT_EQ(sk.decrypt<long>(ct_lp), large_positive);
    long large_negative = -9223372036854775807;
    PaillierCipherText ct_ln = pk.encrypt(large_negative);
    ASSERT_EQ(sk.decrypt<long>(ct_ln), large_negative);

    PaillierCipherText ct_6 = pk.encrypt<int>(-15);
    ASSERT_EQ(sk.decrypt<int>(ct_6), -15);
    PaillierCipherText ct_7 = pk.encrypt<int>(1);
    PaillierCipherText ct_8 = ct_6 + ct_7;
    ASSERT_EQ(sk.decrypt<int>(ct_8), -14);
    PaillierCipherText ct_9 = pk.encrypt<int>(-1);
    PaillierCipherText ct_10 = ct_6 + ct_9;
    ASSERT_EQ(sk.decrypt<int>(ct_10), -16);
    PaillierCipherText ct_11 = ct_6 * -1;
    ASSERT_EQ(sk.decrypt<int>(ct_11), 15);
}

TEST(paillier, PaillierAdvancedFloatTest)
{
    PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
    pair<PaillierPublicKey, PaillierSecretKey> keypair = keygenerator.generate_keypair();
    PaillierPublicKey pk = keypair.first;
    PaillierSecretKey sk = keypair.second;

    PaillierCipherText ct_1 = pk.encrypt(0.005743);
    ASSERT_NEAR(sk.decrypt<float>(ct_1), 0.005743, 1e-6);
    PaillierCipherText ct_2 = pk.encrypt(-0.005743);
    ASSERT_NEAR(sk.decrypt<float>(ct_2), -0.005743, 1e-6);

    PaillierCipherText ct_3 = pk.encrypt(15.5);
    PaillierCipherText ct_4 = pk.encrypt(0.3);
    PaillierCipherText ct_5 = ct_3 + ct_4;
    ASSERT_NEAR(sk.decrypt<float>(ct_5), 15.8, 1e-6);
    PaillierCipherText ct_6 = ct_4 * 0.5;
    ASSERT_NEAR(sk.decrypt<float>(ct_6), 0.15, 1e-6);
}

TEST(paillier, PaillierEncodingTest)
{
    PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
    pair<PaillierPublicKey, PaillierSecretKey> keypair_1 = keygenerator.generate_keypair();
    PaillierPublicKey pk_1 = keypair_1.first;

    EncodedNumber<int> enc_1 = EncodedNumber<int>(pk_1, 15);
    ASSERT_EQ(0, enc_1.exponent);
    ASSERT_EQ(15, enc_1.encoding);

    EncodedNumber<int> enc_2 = EncodedNumber<int>(pk_1, -15);
    ASSERT_EQ(0, enc_2.exponent);
    ASSERT_EQ(positive_mod(-15, pk_1.n), enc_2.encoding);

    EncodedNumber<float> enc_3 = EncodedNumber<float>(pk_1, 15.1);
    ASSERT_NEAR(15.1, pow(enc_3.BASE, enc_3.exponent) * float(enc_3.encoding), 1e-6);
}

TEST(paillier, PaillierDecodingTest)
{
    PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
    pair<PaillierPublicKey, PaillierSecretKey> keypair_1 = keygenerator.generate_keypair();
    PaillierPublicKey pk_1 = keypair_1.first;

    EncodedNumber<int> enc_1 = EncodedNumber<int>(pk_1, 15);
    ASSERT_EQ(0, enc_1.exponent);
    ASSERT_EQ(15, enc_1.decode());

    EncodedNumber<int> enc_2 = EncodedNumber<int>(pk_1, -15);
    ASSERT_EQ(0, enc_2.exponent);
    ASSERT_EQ(-15, enc_2.decode());

    long large_positive = 9223372036854775807;
    EncodedNumber<long> enc_3 = EncodedNumber<long>(pk_1, large_positive);
    ASSERT_EQ(0, enc_3.exponent);
    ASSERT_EQ(large_positive, enc_3.decode());

    long large_negative = -9223372036854775807;
    EncodedNumber<long> enc_4 = EncodedNumber<long>(pk_1, large_negative);
    ASSERT_EQ(0, enc_4.exponent);
    ASSERT_EQ(large_negative, enc_4.decode());

    EncodedNumber<float> enc_5 = EncodedNumber<float>(pk_1, 15.1);
    ASSERT_NEAR(15.1, enc_5.decode(), 1e-6);

    EncodedNumber<float> enc_6 = EncodedNumber<float>(pk_1, -15.1);
    ASSERT_NEAR(-15.1, enc_6.decode(), 1e-6);

    double large_postive_double = 123456.123456;
    EncodedNumber<double> enc_7 = EncodedNumber<double>(pk_1, large_postive_double);
    ASSERT_NEAR(large_postive_double, enc_7.decode(), 1e-6);
}

TEST(paillier, PaillierDecreaseExponentTest)
{
    PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
    pair<PaillierPublicKey, PaillierSecretKey> keypair_1 = keygenerator.generate_keypair();
    PaillierPublicKey pk_1 = keypair_1.first;

    EncodedNumber<float> enc_1 = EncodedNumber<float>(pk_1, 3.14);
    int new_exponent = enc_1.exponent - 3;
    enc_1.decrease_exponent(new_exponent);
    ASSERT_EQ(enc_1.exponent, new_exponent);
    ASSERT_NEAR(3.14, enc_1.decode(), 1e-6);

    EncodedNumber<float> enc_2 = EncodedNumber<float>(pk_1, -3.14);
    int new_exponent_2 = enc_2.exponent - 3;
    enc_2.decrease_exponent(new_exponent_2);
    ASSERT_EQ(enc_2.exponent, new_exponent_2);
    ASSERT_NEAR(-3.14, enc_2.decode(), 1e-6);
}

TEST(paillier, PaillierSelfBlindingTest)
{
    PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
    pair<PaillierPublicKey, PaillierSecretKey> keypair_1 = keygenerator.generate_keypair();
    PaillierPublicKey pk_1 = keypair_1.first;
    PaillierSecretKey sk_1 = keypair_1.second;

    PaillierCipherText ct_1 = pk_1.encrypt(0.005743);
    PaillierCipherText ct_2 = ct_1 * 0;
    ASSERT_EQ(ct_2.c, 1);
    ct_2.self_bliding();
    ASSERT_TRUE(!(ct_2.c == 1));
}

TEST(paillier, PaillierSerializationTest)
{
    PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
    pair<PaillierPublicKey, PaillierSecretKey> keypair_1 = keygenerator.generate_keypair();
    PaillierPublicKey pk_1 = keypair_1.first;
    PaillierSecretKey sk_1 = keypair_1.second;

    ostringstream ss_1;
    boost::archive::text_oarchive ar_1(ss_1);
    ar_1 << boost::serialization::make_nvp("PaillierPublicKey", pk_1);

    istringstream ss_2(ss_1.str());
    boost::archive::text_iarchive ar_2(ss_2);
    PaillierPublicKey pk_2;
    ar_2 >> boost::serialization::make_nvp("PaillierPublicKey", pk_2);

    ASSERT_TRUE(pk_1 == pk_2);
    ASSERT_EQ(pk_1.n2, pk_2.n2);
    ASSERT_EQ(pk_1.max_val, pk_2.max_val);

    PaillierCipherText ct_1 = pk_1.encrypt(42);

    ostringstream ss_3;
    boost::archive::text_oarchive ar_3(ss_3);
    ar_3 << boost::serialization::make_nvp("PaillierCipherText", ct_1);

    istringstream ss_4(ss_3.str());
    boost::archive::text_iarchive ar_4(ss_4);
    PaillierCipherText ct_2;
    ar_4 >> boost::serialization::make_nvp("PaillierCipherText", ct_2);

    ASSERT_EQ(sk_1.decrypt<int>(ct_2), 42);
}