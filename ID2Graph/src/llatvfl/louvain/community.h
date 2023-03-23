#pragma once
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>
#include "graph.h"
using namespace std;

struct Community
{
    vector<float> neigh_weight;
    vector<unsigned int> neigh_pos;
    unsigned int neigh_last;
    Graph *g;                   // network to compute communities for
    int num_nodes;              // nummber of nodes in the network and num_nodes of all vectors
    vector<int> node2community; // community to which each node belongs
    vector<float> in;           // in[c] = total weights within c-th commynity (\sum_{(i, j) \in c} w_(i, j))
    vector<float> tot;          // tot[c]
    // used to compute the modularity participation of each community

    // number of pass for one level computation
    // if -1, compute as many pass as needed to increase modularity
    int maximum_nb_pass_done;

    // a new pass is computed if the last one has generated an increase
    // greater than min_modularity
    // if 0. even a minor increase is enough to go for one more pass
    float min_modularity;

    mt19937 gen;
    uniform_real_distribution<> uniform_dist_0_to_1;

    Community(){};
    Community(Graph *gc, int nbp, float minm, int seed = 42)
    {
        g = gc;
        num_nodes = g->num_nodes;

        neigh_weight.resize(num_nodes, -1);
        neigh_pos.resize(num_nodes);
        neigh_last = 0;

        node2community.resize(num_nodes);
        in.resize(num_nodes);
        tot.resize(num_nodes);

        for (int i = 0; i < num_nodes; i++)
        {
            node2community[i] = i;
            in[i] = g->get_num_selfloops(i);
            tot[i] = g->get_weighted_degree(i);
        }

        maximum_nb_pass_done = nbp;
        min_modularity = minm;

        gen = mt19937(seed);
        srand(seed);
        uniform_dist_0_to_1 = uniform_real_distribution<>(0, 1);
    }

    /**
     * @brief Remove the node from its current community with which it has weights_from_node_to_comm links
     *
     * @param node
     * @param comm
     * @param weights_from_node_to_comm
     */
    void remove(int node, int comm, float weights_from_node_to_comm)
    {
        tot[comm] -= g->get_weighted_degree(node);
        in[comm] -= 2 * weights_from_node_to_comm + g->get_num_selfloops(node);
        node2community[node] = -1;
    }

    /**
     * @brief Insert the node in comm with which it shares weights_from_node_to_comm links
     *
     * @param node
     * @param comm
     * @param weights_from_node_to_comm
     */
    void insert(int node, int comm, float weights_from_node_to_comm)
    {
        tot[comm] += g->get_weighted_degree(node);
        in[comm] += 2 * weights_from_node_to_comm + g->get_num_selfloops(node);
        node2community[node] = comm;
    }

    /**
     * @brief Calculate the current modularity.
     *        Note that each community consists of only ond node due to contraction.
     *
     * @return float
     */
    float modularity()
    {
        float q = 0.;
        float inv_m2 = 1 / (float)g->total_weight;

        for (int i = 0; i < num_nodes; i++)
        {
            if (tot[i] > 0)
                q += inv_m2 * ((float)in[i] - (inv_m2 * (float)tot[i] * (float)tot[i]));
            // q += (float)in[i] / m2 - ((float)tot[i] / m2) * ((float)tot[i] / m2);
        }

        return q;
    }

    /**
     * @brief Compute the gain of modularity if node where inserted in comm
     *
     * @param node
     * @param comm
     * @param weights_from_node_to_comm
     * @param w_degree
     * @return float
     */
    float modularity_gain(int node, int comm, float weights_from_node_to_comm, float w_degree)
    {
        // given that node has weights_from_node_to_comm links to comm.  The formula is:
        // [(In(comm)+2d(node,comm))/2m - ((tot(comm)+deg(node))/2m)^2]-
        // [In(comm)/2m - (tot(comm)/2m)^2 - (deg(node)/2m)^2]
        // where In(comm)    = number or total weights of half-links strictly inside comm
        //       Tot(comm)   = number or total weights of half-links inside or outside comm (sum(degrees))
        //       d(node,com) = number or total weights of links from node to comm
        //       deg(node)   = node degree
        //       m           = number or wegihts of all links
        // ignore const (1/2m)
        return ((float)weights_from_node_to_comm -
                (float)tot[comm] * (float)w_degree / (float)g->total_weight);
    }

    // compute the set of neighboring communities of node
    // for each community, gives the number of links from node to comm
    void compute_neigh_comms(unsigned int node)
    {
        // initialize neigh_weight up to the previous neigh_last
        for (unsigned int i = 0; i < neigh_last; i++)
            neigh_weight[neigh_pos[i]] = -1;
        neigh_last = 0;

        unsigned int degree = g->get_num_neighbors(node);
        neigh_pos[0] = node2community[node];
        neigh_weight[neigh_pos[0]] = 0;
        neigh_last = 1;

        pair<vector<unsigned int>::iterator, vector<float>::iterator> pointer2neigh_weight = g->get_neighbors(node);
        for (unsigned int i = 0; i < degree; i++)
        {
            unsigned int neigh = *(pointer2neigh_weight.first + i);
            unsigned int neigh_comm = node2community[neigh];
            float neigh_w = (g->weights.size() == 0) ? 1. : *(pointer2neigh_weight.second + i);
            if (neigh != node)
            {
                if (neigh_weight[neigh_comm] == -1)
                {
                    neigh_weight[neigh_comm] = 0.;
                    neigh_pos[neigh_last++] = neigh_comm;
                }
                neigh_weight[neigh_comm] += neigh_w;
            }
        }
    }

    Graph partition2graph_binary()
    {
        // Renumber communities
        // renumber[c] = the new index of the community whose current index is `c`
        vector<int> renumber(num_nodes, -1);

        // calculate the size of each community
        for (int node = 0; node < num_nodes; node++)
        {
            renumber[node2community[node]]++;
        }

        // increment `final` if c-th commynity has one or more nodes (renumber[c] != -1)
        int final = 0;
        for (int i = 0; i < num_nodes; i++)
            if (renumber[i] != -1)
                renumber[i] = final++;

        // Compute communities
        vector<vector<int>> comm2nodes(final);  // community id to current node idx
        vector<vector<int>> communities(final); // current node idx to original data records
        for (int node = 0; node < num_nodes; node++)
        {
            // TODO add node handling
            vector<int> &comm = communities[renumber[node2community[node]]];
            comm.insert(comm.end(), g->nodes[node].begin(), g->nodes[node].end());
            comm2nodes[renumber[node2community[node]]].push_back(node);
        }

        Graph g2(communities);

        // the size of contracted graph is equal to the number of communities
        g2.num_nodes = comm2nodes.size();
        g2.degrees.resize(comm2nodes.size());

        int comm_deg = comm2nodes.size();
        for (int comm = 0; comm < comm_deg; comm++)
        {
            unordered_map<int, float> m;
            unordered_map<int, float>::iterator it;

            int comm_size = comm2nodes[comm].size();
            for (int node = 0; node < comm_size; node++)
            {
                pair<vector<unsigned int>::iterator, vector<float>::iterator> p = g->get_neighbors(comm2nodes[comm][node]);
                int degree = g->get_num_neighbors(comm2nodes[comm][node]);
                for (int i = 0; i < degree; i++)
                {
                    int neigh = *(p.first + i);
                    int neigh_comm = renumber[node2community[neigh]];
                    float neigh_weight_i = (g->weights.size() == 0) ? 1. : *(p.second + i);

                    it = m.find(neigh_comm);
                    if (it == m.end())
                        m.insert(make_pair(neigh_comm, neigh_weight_i));
                    else
                        it->second += neigh_weight_i;
                }
            }

            unsigned int m_size = m.size();
            g2.degrees[comm] = (comm == 0) ? m_size : g2.degrees[comm - 1] + m_size;
            g2.num_links += m_size;

            g2.links.reserve(m_size);
            g2.weights.reserve(m_size);
            for (it = m.begin(); it != m.end(); it++)
            {
                g2.total_weight += it->second;
                g2.links.push_back(it->first);
                g2.weights.push_back(it->second);
            }
        }

        return g2;
    }

    bool one_level()
    {
        bool improvement = false;
        int nb_moves;
        int nb_pass_done = 0;
        float new_mod = modularity();
        float cur_mod = new_mod;

        vector<int> random_indicies_to_num_nodes(num_nodes);
        iota(random_indicies_to_num_nodes.begin(), random_indicies_to_num_nodes.end(), 0);

        // repeat while
        //   there is an improvement of modularity
        //   or there is an improvement of modularity greater than a given epsilon
        //   or a predefined number of pass have been done
        do
        {
            cur_mod = new_mod;
            nb_moves = 0;
            nb_pass_done++;

            // for each node: remove the node from its community and insert it in the best community
            shuffle(random_indicies_to_num_nodes.begin(), random_indicies_to_num_nodes.end(), gen);

            for (int node_tmp = 0; node_tmp < num_nodes; node_tmp++)
            {
                int node = random_indicies_to_num_nodes[node_tmp];
                int node_comm = node2community[node];
                float w_degree = g->get_weighted_degree(node);

                // computation of all neighboring communities of current node
                compute_neigh_comms(node);
                // remove node from its current community
                remove(node, node_comm, neigh_weight[node_comm]);

                // compute the nearest community for node
                // default choice for future insertion is the former community
                int best_comm = node_comm;
                float best_nblinks = 0.;
                float best_increase = 0.;

                for (unsigned int i = 0; i < neigh_last; i++)
                {
                    float increase = modularity_gain(node, neigh_pos[i], neigh_weight[neigh_pos[i]], w_degree);
                    if (increase > best_increase)
                    {
                        best_comm = neigh_pos[i];
                        best_nblinks = neigh_weight[neigh_pos[i]];
                        best_increase = increase;
                    }
                }

                // insert node in the nearest community
                insert(node, best_comm, best_nblinks);

                if (best_comm != node_comm)
                    nb_moves++;
            }

            float total_tot = 0;
            float total_in = 0;
            unsigned int tot_size = tot.size();
            for (unsigned int i = 0; i < tot_size; i++)
            {
                total_tot += tot[i];
                total_in += in[i];
            }

            new_mod = modularity();
            if (nb_moves > 0)
                improvement = true;

        } while ((nb_moves > 0) &&
                 (nb_pass_done < maximum_nb_pass_done));

        return improvement;
    }
};