import jax.numpy as jnp
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxtyping import Float, Array


class SimpleP2PSetup:
    def __init__(self, N_agents, n_neighbours, comm_graph, c_mg, c_pr, c_tr, c_regul, T, x_pr_setpoint, loads):
        # Each agent opt. variables: number of neighbours (from which to buy energy) + 1 (energy from main grid) + 1 (energy produced)
        # We assume regular graph for simplicity (each agent has the same number of neighbours)
        # c_mg = linear price factor on main grid electricity,
        # c_pr = quadratic penalty on deviation from generator setpoint
        # c_tr = cost/reward of trading
        # loads \in N*T*1 = vector of loads over time per each agent
        if comm_graph.is_directed():
            raise ValueError("[SimpleP2PSetup]: the graph must be undirected")
        n_x_per_t = n_neighbours + 2
        self.N_agents = N_agents
        self.Q,self.q, self.C, self.D = self.define_cost_functions(n_x_per_t, T, c_mg, c_pr, c_tr, c_regul, x_pr_setpoint)
        self.A_eq_local_const, self.b_eq_local_const = self.define_local_constraints(n_x_per_t,  T,loads)
        # edge_to_index maps the graph edge to the index in the constraints
        self.A_eq_shared_const, self.b_eq_shared_const, self.edge_to_index = self.define_shared_constraints(n_x_per_t, comm_graph, T)
        self.A_sel_positive_vars = self.define_positive_variables(n_x_per_t, T)

    def define_cost_functions(
        self, n_x_per_t, T, c_mg, c_pr, c_tr, c_regul,
        x_pr_setpoint: Float[Array, ""],
    ):
        N_agents = self.N_agents
        n_opt_var = n_x_per_t
        Q_single_t = c_regul * jnp.eye(n_opt_var)
        Q_single_t = Q_single_t.at[1, 1].set(c_pr)
        Q_all_t = jnp.kron(jnp.eye(T), Q_single_t)
        Q = jnp.tile(jnp.expand_dims(Q_all_t, 0), (N_agents,1,1))

        q_single_t = np.ones((n_opt_var, 1))
        q_single_t[0, 0] = 0
        q_single_t[1, 0] = 0
        q_single_t = q_single_t*c_tr
        q_all_t = np.kron(np.ones((T,1)), q_single_t)
        q = np.tile(q_all_t, (N_agents,1,1))

        # TODO: create reference for generators
        for i in range(N_agents):
            for t in range(T):
                q[i,1+t*n_opt_var,0] = -c_pr*x_pr_setpoint[i,t,0]
        q = jnp.array(q)

        C_single_t = jnp.zeros((1, n_opt_var))\
            .at[0, 0].set(c_mg)
        C_all_t = jnp.kron(jnp.eye(T), C_single_t)
        C = jnp.tile(C_all_t, (N_agents,1,1))

        D_single_t = jnp.zeros((1, n_opt_var))\
            .at[0, 0].set(1)
        D_all_t = jnp.kron(jnp.eye(T), D_single_t)
        D = jnp.tile(D_all_t, (N_agents, 1, 1))

        return Q, q, C, D

    def define_local_constraints(self, n_x_per_t, T, loads):
        n_local_const_eq = 1 # power balance constraint
        N_agents=self.N_agents
        A_eq_loc_const_single_t = jnp.ones((n_local_const_eq, n_x_per_t))
        A_eq_loc_const_all_t = jnp.kron(jnp.eye(T), A_eq_loc_const_single_t)
        A_eq = jnp.tile(A_eq_loc_const_all_t, (N_agents,1,1))
        b_eq = loads

        return A_eq, b_eq

    def define_positive_variables(self, n_x_per_t, T): #Constraint some variables to be positive. WARNING these are "soft" constraints
        A_sel_single_t = (jnp.zeros((n_x_per_t, n_x_per_t))
            .at[0, 0].set(1) #main grid power must be positive
            .at[1, 1].set(1)  # main grid power must be positive
        )
        A_sel_all_t = jnp.kron(jnp.eye(T), A_sel_single_t)
        A_sel_pos = jnp.tile(A_sel_all_t, (self.N_agents, 1, 1))
        return  A_sel_pos

    def define_shared_constraints(self, n_x_per_t, graph, T) -> tuple[
        Float[Array, "N T*n_eq_constr T*n_x_per_t"],
        Float[Array, "N T*n_eq_constr 1"],
        dict[int, int],
    ]:
        N = self.N_agents
        n_opt = n_x_per_t
        n_edg_without_self_loops = len(graph.edges) - N
        n_eq_constr = n_edg_without_self_loops # 1 reciprocity const. per edge (each edge is counted only once because graph is undirected)
        A_eq_shared_single_t = np.zeros((N, n_eq_constr, n_x_per_t))

        # Ugly but it should work
        edge_to_index = {}
        i=0
        for edge in graph.edges:
            if edge[0]!=edge[1]:
                edge_to_index.update({edge: i})
                i=i+1
        for i in range(N):
            j=0
            for neigh in graph.neighbors(i):
                if neigh!=i:
                    edge = (i,neigh) if (i,neigh) in edge_to_index.keys() else (neigh,i)
                    A_eq_shared_single_t[i, edge_to_index[edge],j+2]=1
                    j=j+1
        A_eq_shared = np.zeros((N, T*n_eq_constr, T*n_x_per_t))
        for i in range(N):
            A_eq_shared[i,:,:] = np.kron(np.eye(T), A_eq_shared_single_t[i,:,:])

        b_eq_shared = np.zeros((N, T*n_eq_constr, 1))

        return A_eq_shared, b_eq_shared, edge_to_index
