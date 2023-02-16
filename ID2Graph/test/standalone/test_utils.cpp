#include <vector>
#include <cassert>
#include <iostream>
#include <random>
#include "llatvfl/utils/utils.h"
#include "llatvfl/utils/prime.h"
#include "llatvfl/paillier/paillier.h"
#include "gtest/gtest.h"
using namespace std;

TEST(Utils, SoftmaxTest)
{
    vector<float> in = {1, 2, 3, 4, 5};
    vector<float> out = softmax(in);
    vector<float> ans = {0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865};
    for (int i = 0; i < ans.size(); i++)
    {
        ASSERT_NEAR(out[i], ans[i], 1e-7);
    }
}

TEST(Utils, QuantileTest)
{
    vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    vector<float> quartiles = Quantile<float>(in, {0.25, 0.5, 0.75});
    vector<float> test_quartiles = {3.25, 6, 8.75};
    for (int i = 0; i < quartiles.size(); i++)
    {
        ASSERT_EQ(quartiles[i], test_quartiles[i]);
    }
}

TEST(Utils, NumPartiesTest)
{
    vector<int> num_parties_per_process = get_num_parties_per_process(3, 8);
    vector<int> test_num_parties_per_process = {3, 3, 2};
    for (int i = 0; i < test_num_parties_per_process.size(); i++)
    {
        ASSERT_EQ(num_parties_per_process[i], test_num_parties_per_process[i]);
    }
}

TEST(utils, GCDTest)
{
    ASSERT_EQ(gcd(12, 42), 6);
    ASSERT_EQ(gcd(42, 12), 6);
}

TEST(utils, LCMTest)
{
    ASSERT_EQ(lcm(3, 4), 12);
    ASSERT_EQ(lcm(4, 3), 12);
}

TEST(utils, ModPowTest)
{
    ASSERT_EQ(modpow(17, 20, 17345), 13896);
    ASSERT_EQ(modpow(23, 19, 1), 0);
}

TEST(utils, MillerRabinPrimalityTest)
{
    ASSERT_TRUE(miller_rabin_primality_test(2));
    ASSERT_TRUE(miller_rabin_primality_test(3));
    ASSERT_TRUE(miller_rabin_primality_test(5));
    ASSERT_TRUE(miller_rabin_primality_test(1223));
    ASSERT_TRUE(miller_rabin_primality_test(9973));
    ASSERT_TRUE(miller_rabin_primality_test(99991));
    ASSERT_TRUE(miller_rabin_primality_test(524287));
    ASSERT_TRUE(miller_rabin_primality_test(2147483647));
    ASSERT_TRUE(miller_rabin_primality_test(200560490131));
    ASSERT_TRUE(miller_rabin_primality_test(92709568269121));
    ASSERT_TRUE(miller_rabin_primality_test(7365373222715531));
    ASSERT_TRUE(miller_rabin_primality_test(9007199254740881));
    ASSERT_TRUE(miller_rabin_primality_test(Bint("344582637415136393718840238387440763934159885455689809106778719839424629825829696027081285824319293")));
    ASSERT_TRUE(miller_rabin_primality_test(Bint("4083447434264164562204785541649619447013268836904761731606934449524722956971748845773058539600568258686945511880311921143786847069247176376531541055084570359915967511312326077113399980012291443")));
    ASSERT_TRUE(miller_rabin_primality_test(Bint("840554482393696190041092230614058763326557764052521405232692238232194511845692732698786219095908816002445579054477802598217867630615800128684969271165417812064109969684853213643966022686114404219528348019927092721912501193475916694377763127948350365189535938635081933110703492840209274233259659297047")));
    ASSERT_TRUE(!miller_rabin_primality_test(0));
    ASSERT_TRUE(!miller_rabin_primality_test(1));
    ASSERT_TRUE(!miller_rabin_primality_test(99991 * 9973));
    ASSERT_TRUE(!miller_rabin_primality_test(1234567892));
    ASSERT_TRUE(!miller_rabin_primality_test(12345678900));
    ASSERT_TRUE(!miller_rabin_primality_test(75361));
    ASSERT_TRUE(!miller_rabin_primality_test(512461));
    ASSERT_TRUE(!miller_rabin_primality_test(1565912117761));
    ASSERT_TRUE(!miller_rabin_primality_test(927868565729827));
    ASSERT_TRUE(!miller_rabin_primality_test(2543736331508089));
    ASSERT_TRUE(!miller_rabin_primality_test(8635844967113809));
    ASSERT_TRUE(!miller_rabin_primality_test(Bint("114381625757888867669235779976146612010218296721242362562561842935706935245733897830597123563958705058989075147599290026879543541")));
}

TEST(utils, GenerateProbablyPrimeTest)
{
    Bint p = generate_probably_prime(3);
    ASSERT_TRUE(p == 5 || p == 7);

    Bint q = generate_probably_prime(4);
    ASSERT_TRUE(q == 11 || q == 13);
}
