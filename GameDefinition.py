import networkx
import networkx as nx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float, Array


def bmm(M: Float[Array, "N a n_x"], x: Float[Array, "N n_x"]) -> Float[Array, "N a"]:
    return jnp.einsum('nax,nx->na', M, x)

def bmm3(M: Float[Array, "N a n_x"], x: Float[Array, "N n_x 1"]) -> Float[Array, "N a 1"]:
    return jnp.einsum('nax,nxi->nai', M, x)

class AggregativePartialInfo:
    def __init__(
        self,
        N,
        communication_graph,
        Q: Float[Array, "N n_x n_x"],
        q: Float[Array, "N n_x 1"],
        C: Float[Array, "N n_s n_x"],
        D: Float[Array, "N n_s n_x"],
        A_loc: Float[Array, "N n_s n_x"],
        b_loc: Float[Array, "N n_s 1"],
        A_shared: Float[Array, "N n_m n_x"],
        b_shared: Float[Array, "N n_m 1"],
        A_sel_positive_vars: Float[Array, "N n_x n_x"],
        gamma_barr=10,
        test=False,
    ):
        r"""
        Define distributed aggregative game where each agent has the same number of opt. variables.

        Args:
            N: Number of agents.
            Q: Q[i,:,:] is the matrix that define the (quadratic) local cost
            q: q[i,:] is the affine part of the local cost
            C: C[i,:,:] is the matrix that define the local contribution to the aggregation
                The aggregative variable is sigma = \sum (1/N) C_i x_i
            D: D[i,:,:] is the matrix that define the influence of the aggregation to the agent
            A_shared: A_shared[i,:,:] defines the local contribution to the shared eq. constraints
            b_shared: b_shared[i,:] is the affine part of the shared eq. constraints
        #### WARNING: D_iC_i should be symmetric!!
        # The game is in the form:
        # \sum .5 x_i' Q_i x_i + q_i'x_i + (1/N)(D_i x_i)'Cx
        # s.t. \sum_i A_shared_i x_i = \sum_i b_shared_i
        """
        if test:
            N, n_opt_var, Q, c, Q_sel, c_sel, A_shared, b_shared, \
                A_eq_loc, A_ineq_loc, b_eq_loc, b_ineq_loc, communication_graph = self.setToTestGameSetup()
        self.N_agents = N
        self.n_opt_variables = Q.shape[1]
        self.n_agg_variables = C.shape[1]
        # Local constraints
        self.A_eq_loc = A_loc
        self.b_eq_loc = b_loc
        self.n_loc_eq_constr = self.A_eq_loc.shape[1]
        # Shared constraints
        self.A_eq_shared= A_shared
        self.b_eq_shared = b_shared
        self.n_shared_eq_constr = self.A_eq_shared.shape[1]
        # Selection matrix for variables that need be positive
        self.A_sel_positive_vars = A_sel_positive_vars
        self.gamma_barr = gamma_barr
        # Define the (nonlinear) game mapping as a torch custom activation function
        self.F = self.GameMapping(Q, q, C, D, A_sel_positive_vars, gamma_barr)
        self.J = self.GameCost(Q, q, C, D)
        # Define the consensus operator
        # self.K = self.Consensus(communication_graph, self.n_shared_eq_constr)
        # Define the adjacency operator
        self.W = self.Adjacency(communication_graph)
        # Define the operator which computes the locally-estimated aggregation
        self.S = self.Aggregation(C)

    class GameCost:
        def __init__(
            self,
            Q: Float[Array, "N n_x n_x"],
            q: Float[Array, "N n_x 1"],
            C: Float[Array, "N n_s n_x"], 
            D: Float[Array, "N n_s n_x"],
        ):
            self.Q = Q
            self.q = q
            self.C = C
            self.D = D
            self.N = Q.shape[0]

        def __call__(self, x: Float[Array, "N n_x 1"]) -> Float[Array, "N 1 1"]:
            N = self.N

            Cx: Float[Array, "N n_s 1"] = bmm3(self.C, x)

            agg = jnp.sum(Cx, axis=0, keepdims=True)
            agg: Float[Array, "N n_s 1"] = jnp.repeat(agg, N, axis=0)

            term1 = jnp.matmul(jnp.swapaxes(x,1,2),
                              0.5 * bmm3(self.Q, x) + self.q)

            term2 = (1/N) * bmm3(
                jnp.swapaxes(bmm3(self.D, x), 1, 2),
                agg
            )

            return term1 + term2

    class GameMapping:
        def __init__(
            self,
            Q: Float[Array, "N n_x n_x"],
            q: Float[Array, "N n_x 1"],
            C: Float[Array, "N n_s n_x"], 
            D: Float[Array, "N n_s n_x"],
            A_sel_positive_vars: Float[Array, "N n_x n_x"],
            gamma_barr,
        ):
            self.Q = Q
            self.q = q
            self.C = C
            self.D = D
            self.N = Q.shape[0]
            self.n_x = Q.shape[1]
            self.A_sel_positive_vars = A_sel_positive_vars
            self.gamma_barr = gamma_barr

        def __call__(self, x: Float[Array, "N n_x 1"], agg=None) -> Float[Array, "N n_x 1"]:
            """
            Args:
                agg: allows to provide the estimated aggregation (Partial information)
            """
            N = self.N

            if agg is None:
                Cx = bmm3(self.C, x)
                agg = jnp.mean(Cx, axis=0, keepdims=True)
                agg = jnp.repeat(agg, N, axis=0)

            # barrier
            Ax = bmm3(self.A_sel_positive_vars, x)

            #Force positive variables via barrier function. #TODO: clean this up!
            barrier = jnp.maximum(
                -1.0 / Ax,
                -self.gamma_barr * jnp.ones_like(x)
            )

            # F = Qx + q + (1/N)*(D_i'Cx + C_i'*D_i*x_i)
            term1 = bmm3(self.Q, x) + self.q

            term2 = (1/N) * (
                jnp.matmul(jnp.swapaxes(self.D,1,2), agg)
                + jnp.matmul(
                    jnp.swapaxes(self.C,1,2),
                    bmm3(self.D, x)
                )
            )

            return barrier + term1 + term2

        def get_strMon_Lip_constants(self):
            """Return strong monotonicity and Lipschitz constant."""
            # Define the matrix that defines the pseudogradient mapping
            # F = Mx +m, where M = diag(Q_i) + diag(C_i'D_i) + col(D_i'C)
            N = self.Q.shape[0]
            n_x = self.Q.shape[2]

            diagonal_elements = self.Q + (1/N)*jnp.matmul(
                jnp.swapaxes(self.C,1,2), self.D
            )

            blocks = [diagonal_elements[i] for i in range(N)]
            Q_mat = jsp.linalg.block_diag(*blocks)

            for i in range(N):
                for j in range(N):
                    Q_mat = Q_mat.at[
                        i*n_x:(i+1)*n_x,
                        j*n_x:(j+1)*n_x
                    ].add(
                        jnp.matmul(
                            self.D[i].T,
                            self.C[j]
                        )
                    )

            S = jnp.linalg.svd(Q_mat, compute_uv=False)

            return jnp.min(S), jnp.max(S)

    class Consensus:
        def __init__(self, communication_graph, N_dual_variables):
            super().__init__()
            # Convert Laplacian matrix to sparse tensor
            L = networkx.laplacian_matrix(communication_graph)
            self.L: Float[Array, "n_x n_x 1 1"] = jnp.array(L.tocsr())[:, :, jnp.newaxis, jnp.newaxis]
            # TODO: understand why sparse does not work
            # self.L = L_torch.to_sparse_coo()

        def forward(self, x: Float["N n_x"]):
            n_x = x.shape[1]
            L_expanded = jnp.kron(jnp.eye(n_x)[:, :, jnp.newaxis, jnp.newaxis], self.L)
            return jnp.sum(jnp.matmul(L_expanded, x), dim=1) # This applies the laplacian matrix to each of the dual variables

    class Adjacency:
        def __init__(self, communication_graph):
            W = nx.adjacency_matrix(communication_graph).toarray()
            W = jnp.array(W)
            self.W = W[..., None, None]  # (N,N,1,1)

        def __call__(self, x: Float[Array, "N n_x"]):
            n_x = x.shape[1]

            W_expanded = jnp.kron(
                jnp.eye(n_x)[None,None,:,:],
                self.W
            )

            return jnp.sum(jnp.matmul(W_expanded, x), axis=1)

    class Aggregation:
        def __init__(self, C):
            self.C = C

        def __call__(self, x: Float[Array, "N n_x 1"]) -> Float[Array, "N n_s 1"]:
            return bmm3(self.C, x)

    def setToTestGameSetup(self):
        raise NotImplementedError("[GameAggregativePartInfo:setToTestGameSetup] Test game not implemented")

    def get_strMon_Lip_constants_eq_constraints(self) -> tuple[float, float]:
        """
        Returns:
            mu_A: 
            L_A: 
        """
        A = self.A_eq_shared
        A_square = bmm3(A, jnp.transpose(A, (0, 2, 1)))
        mu_A = jnp.min(jnp.linalg.eigvals(A_square).real)
        L_A = jnp.sqrt(jnp.max(jnp.linalg.eigvals(A_square).real))
        return mu_A, L_A
