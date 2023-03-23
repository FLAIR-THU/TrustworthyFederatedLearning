#pragma once
#include <iostream>
#include <limits>
#include <map>
#include <vector>
#include "community.h"
using namespace std;

struct Louvain
{
    int max_itr;
    float precision;
    int nbp;
    int seed;
    int verbose;

    Community community;
    // Graph g;

    Louvain(int nbp_ = 100, int max_itr_ = 100, float precision_ = 0.000001,
            int seed_ = 42, int verbose_ = -1)
    {
        max_itr = max_itr_;
        precision = precision_;
        nbp = nbp_;
        seed = seed_;
        verbose = verbose_;
    }

    void reseed(int seed_)
    {
        seed = seed_;
    }

    void fit(Graph &gc)
    {
        bool improvement = true;

        // g = gc;
        community = Community(&gc, nbp, precision, seed);
        float mod = community.modularity(), new_mod;

        for (int i = 0; i < max_itr; i++)
        {
            improvement = community.one_level();
            new_mod = community.modularity();
            gc = community.partition2graph_binary();
            community = Community(&gc, nbp, precision, seed);
            mod = new_mod;

            if (verbose > 0 && i % verbose == 0)
            {
                cout << i << ": " << mod << endl;
            }

            if (!improvement || gc.nodes.size() == 2)
            {
                break;
            }

            if (i == max_itr - 1)
            {
                cout << "\033[31moptimization did not converge\033[0m" << endl;
            }
        }
    }
};