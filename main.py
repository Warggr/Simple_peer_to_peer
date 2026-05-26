import numpy as np
import networkx as nx
import pickle
from GNE_part_info import PrimalDualPartialInfo
from GameDefinition import AggregativePartialInfo, bmm3
from SimpleP2PSetup import SimpleP2PSetup
from jaxtyping import Float, Array
import time
import logging
import copy
import math
import jax.numpy as jnp


def gaussian(x, alpha, r):
    return 1. / (math.sqrt(alpha ** math.pi)) * np.exp(-alpha * np.power((x - r), 2.))


def generate_load_profile(N,T, variance, *, seed=None) -> Float[np.ndarray, ""]:
    loads = np.zeros((N,T,1))
    for i in range(N):
        peak_time = min(max(0.1*np.random.randn(), -1),1)
        steepness = 1/(max(0.3*np.random.randn() ,.1))
        x = np.linspace(-1, 1, num=T)
        nominal_load = gaussian(x, steepness, peak_time) + 1
        loads[i,:,0] = nominal_load + variance*np.random.randn(T)
    return loads

def generate_gen_profile(N,T, variance) -> Float[np.ndarray, ""]:
    gen_profile = np.zeros((N,T,1))
    nominal_profile = np.matmul((np.random.rand(N,1))+0.5, np.ones((1,T)) )
    gen_profile[:,:,0] = nominal_profile + variance*np.random.randn(N,T)
    return gen_profile

def get_graph(
    N_agents: int,
    n_neighbors=2,
    seed: int | None = None,
):
    comm_graph = nx.random_regular_graph(n_neighbors, N_agents, seed=seed)
    while not nx.is_connected(comm_graph):
        n_neighbors = n_neighbors+1
        comm_graph = nx.random_regular_graph(n_neighbors, N_agents, seed=seed)
    # add self loops
    for i in comm_graph.nodes:
        comm_graph.add_edge(i,i)
    # Make graph stochastic WARNING: THIS IS ALSO DOUBLY STOCHASTIC ONLY BECAUSE WE ARE USING A REGULAR GRAPH
    return n_neighbors, nx.stochastic_graph(comm_graph.to_directed()).to_undirected()

def get_game(
    loads, x_pr_setpoint, T,
    N_agents,
    comm_graph = None,
    n_neighbors=2,  # for simplicity, each agent has the same number of neighbours. This is only used to create the communication graph (but i's not needed otherwise)
    c_mg = 10,
    c_pr = 10,
    c_tr = 1,
    c_regul = 0.1,
    seed: int | None = None,
):
    if comm_graph is None:
        n_neighbors, comm_graph = get_graph(N_agents, n_neighbors=n_neighbors, seed=seed)
    return SimpleP2PSetup(N_agents, n_neighbors, comm_graph, c_mg, c_pr, c_tr, c_regul, T, x_pr_setpoint, loads)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int, default=1)
    parser.add_argument('output_file', help='Output file or job ID. If this is an integer, will be interpreted as job ID and the output filename will be `saved_test_result_{job_id}.pkl', default='')
    parser.add_argument('--use-test-game', action='store_true')
    args = parser.parse_args()
    seed = args.seed
    use_test_game = args.use_test_game
    try:
        job_id = int(args.output_file)
        args.output_file = f"saved_test_result_{job_id}.pkl"
    except ValueError:
        pass
    logging.basicConfig(filename='log.txt', filemode='w',level=logging.DEBUG)
    if use_test_game:
        print("WARNING: test game will be used.")
        logging.info("WARNING: test game will be used.")
    print("Random seed set to  " + str(seed))
    logging.info("Random seed set to  " + str(seed))
    np.random.seed(seed)
    N_it_per_residual_computation = 10
    N_agents = 6
    N_random_tests = 1

    # parameters
    T = 24*4
    N_iter = 100000
    N_iter_per_timestep = [1, 100, 1000]

    # Create load/gen profiles
    # x_pr_setpoint = torch.ones(N_agents, T,1)
    # loads = 5*torch.ones(N_agents, T,1)
    x_pr_setpoint = generate_gen_profile(N_agents, T, 0)
    loads = generate_load_profile(N_agents,T,0.01)

    for test in range(N_random_tests):
        ##########################################
        #        Test case creation              #
        ##########################################
        n_neighbors, comm_graph = get_graph(N_agents, seed=seed)
        game_params = get_game(loads=loads, x_pr_setpoint=x_pr_setpoint, T=T, N_agents=N_agents, comm_graph=comm_graph, n_neighbors=n_neighbors)

        print("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
        logging.info("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
        ##########################################
        #             Game inizialization        #
        ##########################################
        game = AggregativePartialInfo(N_agents, comm_graph, game_params.Q, game_params.q, game_params.C, game_params.D,\
                                      game_params.A_eq_local_const, game_params.b_eq_local_const, \
                                      game_params.A_eq_shared_const, game_params.b_eq_shared_const, game_params.A_sel_positive_vars)
        x_0 = jnp.zeros((game.N_agents, game.n_opt_variables)) + \
            bmm3(game_params.A_sel_positive_vars, jnp.ones((game.N_agents, game.n_opt_variables, 1))).squeeze(2)
        if test == 0:
            print("The game has " + str(game.N_agents) + " agents; " + str(game.n_opt_variables) + " opt. variables per agent; " \
                  + " local eq. constraints; " + str(game.n_shared_eq_constr) + " shared eq. constraints" )
            logging.info("The game has " + str(game.N_agents) + " agents; " + str(game.n_opt_variables) + " opt. variables per agent; " \
                  + str(game.A_eq_loc.shape[1]) + " local eq. constraints; " + str(game.n_shared_eq_constr) + " shared eq. constraints" )
            ##########################################
            #   Variables storage inizialization     #
            ##########################################
            # pFB-Tichonov
            x_store = np.zeros((N_random_tests, game.N_agents, game.n_opt_variables))
            dual_share_store = np.zeros((N_random_tests, game.N_agents, game.n_shared_eq_constr))
            dual_loc_store = np.zeros((N_random_tests, game.N_agents, game.n_loc_eq_constr))
            aux_store = np.zeros((N_random_tests, game.N_agents, game.n_shared_eq_constr))
            res_est_store = np.zeros((N_random_tests, game.N_agents, game.n_shared_eq_constr))
            sigma_est_store = np.zeros((N_random_tests, game.N_agents, game.n_agg_variables))
            residual_store = np.zeros((N_random_tests, (N_iter // N_it_per_residual_computation)))
            local_constr_viol = np.zeros((N_random_tests, 1))
            shared_const_viol = np.zeros((N_random_tests, 1))

        #######################################
        #          GNE seeking                #
        #######################################
        # alg. initialization
        alg = PrimalDualPartialInfo(game)
        # The theoretically-sound stepsize is too small!
        # alg.set_stepsize_using_Lip_const(safety_margin=.9)
        index_storage = 0
        avg_time_per_it = 0
        for k in range(N_iter):
            if k % N_it_per_residual_computation == 0:
                # Save performance metrics
                state, r, c, const_viol_sh, const_viol_loc, dist_ref  = alg.get_state()
                residual_store[test, index_storage] = r
                print("Iteration " + str(k) + " Residual: " + str(r) + " Average time: " + str(avg_time_per_it))
                logging.info("Iteration " + str(k) + " Residual: " + str(r) +" Average time: " + str(avg_time_per_it))
                index_storage = index_storage + 1
            #  Algorithm run
            start_time = time.time()
            alg.run_once()
            end_time = time.time()
            avg_time_per_it = (avg_time_per_it * k + (end_time - start_time)) / (k + 1)

        # Store final variables
        state, r, c, const_viol_sh, const_viol_loc, dist_ref = alg.get_state()
        x_store[test, :, :] = state.x.squeeze(2)
        dual_share_store[test, :, :] = state.dual.squeeze(2)
        dual_loc_store[test,:,:] = state.dual_loc.squeeze(2)
        aux_store[test, :, :] = state.aux.squeeze(2)
        sigma_est_store[test,:,:] = state.agg.squeeze(2)
        res_est_store[test,:,:] = state.res.squeeze(2)
        local_constr_viol[test] = const_viol_loc
        shared_const_viol[test] = const_viol_sh

        ############################
        #        time-var. test    #
        ############################
        for index_K in range(len(N_iter_per_timestep)):
            for t in range(T):
                print("Initializing time-step" + str(t) + " out of " + str(T))
                logging.info("Initializing time-step" + str(t) + " out of " + str(T))
                game_old = copy.deepcopy(game)
                game_params = get_game(jnp.expand_dims(loads[:,t], 1), jnp.expand_dims(x_pr_setpoint[:,t], 1), 1, N_agents, comm_graph=comm_graph, n_neighbors=n_neighbors)
                game = AggregativePartialInfo(N_agents, comm_graph, game_params.Q, game_params.q, game_params.C,
                                              game_params.D, \
                                              game_params.A_eq_local_const, game_params.b_eq_local_const, \
                                              game_params.A_eq_shared_const, game_params.b_eq_shared_const, game_params.A_sel_positive_vars)
                # initialize the variables to be stored over the simulation period
                if test==0 and t==0 and index_K==0:
                    n = game.n_opt_variables  # For simplicity every agent has the same n. of variables
                    s = game.n_agg_variables
                    m = game.n_shared_eq_constr
                    m_loc = game.n_loc_eq_constr
                    x_tvar = np.zeros((N_random_tests, len(N_iter_per_timestep), T, N_agents, n))
                    agg_tvar = np.zeros((N_random_tests, len(N_iter_per_timestep),T, N_agents, s))
                    res_tvar = np.zeros((N_random_tests, len(N_iter_per_timestep),T, N_agents, m))
                    dual_tvar = np.zeros((N_random_tests, len(N_iter_per_timestep),  T, N_agents, m))
                    aux_tvar = np.zeros((N_random_tests, len(N_iter_per_timestep), T, N_agents, m))
                    dual_loc_tvar = np.zeros((N_random_tests, len(N_iter_per_timestep), T, N_agents, m_loc))
                    # Performance metrics
                    shared_const_viol_tvar = np.zeros((N_random_tests,len(N_iter_per_timestep),T))
                    loc_const_viol_tvar = np.zeros((N_random_tests,len(N_iter_per_timestep), T))
                    distance_from_optimal_tvar = np.zeros((N_random_tests,len(N_iter_per_timestep), T))
                if t==0:
                    x_tvar[test,index_K, 0, :, :] = jnp.zeros((game.N_agents, game.n_opt_variables)) + \
                                                    bmm3(game_params.A_sel_positive_vars, jnp.ones((game.N_agents, game.n_opt_variables, 1))).squeeze(2)
                    alg = PrimalDualPartialInfo(game, x_0=jnp.expand_dims(x_tvar[test,index_K, 0, :, :], 2))
                else:
                    # alg. re-initialization
                    x_init = jnp.expand_dims(x_tvar[test, index_K,t-1, :, :], 2)
                    agg_init = jnp.expand_dims(agg_tvar[test,index_K,t-1,:,:], 2) - game_old.S(x_init) + game.S(x_init)
                    res_init = jnp.expand_dims(res_tvar[test,index_K,t-1,:,:], 2) - game_old.b_eq_shared + game.b_eq_shared
                    dual_init = jnp.expand_dims(dual_tvar[test,index_K,t-1,:,:], 2)
                    aux_init = jnp.expand_dims(aux_tvar[test,index_K,t-1,:,:], 2)
                    dual_loc_init = jnp.expand_dims(dual_loc_tvar[test,index_K,t-1,:,:], 2)
                    alg = PrimalDualPartialInfo(game, x_0=x_init, agg_0=agg_init, res_0=res_init, dual_0=dual_init, aux_0=aux_init, dual_loc_0=dual_loc_init)
                for k in range(N_iter_per_timestep[index_K]):
                    #  Algorithm run
                    alg.run_once()

                # Compute P-distance with respect to pre-computed GNE
                x_ref = jnp.expand_dims(x_store[test, :, t*n:(t+1)*n], 2)
                # d_ref = dual_share_store[test, :, t*m:(t+1)*m].unsqueeze(2)
                # d_loc_ref = dual_loc_store[test, :, t*m_loc:(t+1)*m_loc].unsqueeze(2)
                # d_ref_avg = torch.mean(d_ref, dim=0)
                # x_ref = torch.reshape(x_ref, (x_ref.size(0) * x_ref.size(1), 1))
                # d_loc_ref = torch.reshape(d_loc_ref, (d_loc_ref.size(0) * d_loc_ref.size(1), 1))
                # omega_ref = torch.row_stack((x_ref, d_ref_avg, d_loc_ref))
                state, r, c, const_viol_sh, const_viol_loc, dist_ref = alg.get_state(x_ref)
                # store computed decision variables (THESE ARE ALSO USED FOR THE RE-INITIALIZATION)
                x_tvar[test, index_K,t, : ,:] = state.x.squeeze(2)
                agg_tvar[test, index_K,t, : ,:] = state.agg.squeeze(2)
                res_tvar[test, index_K,t, : ,:] = state.res.squeeze(2)
                dual_tvar[test,index_K, t, : ,:] = state.dual.squeeze(2)
                aux_tvar[test, index_K,t, : ,:] = state.aux.squeeze(2)
                dual_loc_tvar[test, index_K,t,:,:] = state.dual_loc.squeeze(2)
                # Store performance variables
                loc_const_viol_tvar[test,index_K, t] = const_viol_loc
                shared_const_viol_tvar[test,index_K, t] = const_viol_sh
                distance_from_optimal_tvar[test,index_K,t] = dist_ref
                print("Timestep " + str(t) + " Distance from ref.: " + str(dist_ref), " Constr. violation: " + str(const_viol_sh + const_viol_loc))
                logging.info("Timestep " + str(t) + " Distance from ref.: " + str(dist_ref))

    print("Saving results...")
    logging.info("Saving results...")
    f = open(args.output_file, 'wb')
    pickle.dump([ x_store, residual_store, dual_share_store, dual_loc_store,
                  local_constr_viol, shared_const_viol,
                  loc_const_viol_tvar, shared_const_viol_tvar,
                  distance_from_optimal_tvar, game_params.edge_to_index, N_iter_per_timestep ], f)
    f.close()
    print("Saved")
    logging.info("Saved, job done")


