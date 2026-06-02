from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, Generic

import jax
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from operators.backwardStep import BackwardStep
from GameDefinition import AggregativePartialInfo, bmm, bmm3

if TYPE_CHECKING:
    from typing import Tuple
    from jaxtyping import Float, Array


Problem = TypeVar('Problem')
State = TypeVar('State')


class Solver(ABC, Generic[Problem, State]):
    @abstractmethod
    def get_state(self, ref_point=None) -> Tuple[State, float, float, float, float, float | None]:
        """
        Returns:
            state: algorithm state
            residual: self-explanatory
            cost: self-explanatory
            constr_viol_sh: ???
            constr_viol_loc: ???
            dist_ref: ???
        """

    @abstractmethod
    def run_once(self) -> None:
        """
        Runs one iteration of the algorithm. Call `self.get_state` afterwards to get the current state.
        """

@jax.tree_util.register_dataclass
@dataclass
class PrimalDualPartialInfoState:
    x: Float[Array, ""]
    dual: Float[Array, ""]
    dual_loc: Float[Array, ""]
    aux: Float[Array, ""]
    res: Float[Array, ""]
    agg: Float[Array, ""]


def transpose(A, i, j):
    axes = list(range(len(A.shape)))
    axes[i], axes[j] = axes[j], axes[i]
    return jnp.transpose(A, axes)


class PrimalDualPartialInfo(Solver[AggregativePartialInfo, PrimalDualPartialInfoState]):
    """
    For partial information aggregative games with only shared equality constr.
    """
    def __init__(self, game, x_0=None, agg_0=None, res_0=None, dual_0=None, aux_0=None, dual_loc_0=None, stepsize=0.01):
        self.game = game
        # self.P = self.set_stepsize_using_Lip_const(safety_margin)
        self.stepsize = stepsize
        self.N = game.N_agents
        self.P, self.nu = self.compute_P_matrix()
        n = game.n_opt_variables # For simplicity every agent has the same n. of variables
        s = game.n_agg_variables
        m = game.n_shared_eq_constr
        m_loc = game.n_loc_eq_constr
        if x_0 is None:
            x_0 = jnp.zeros((self.N, n, 1))
        if agg_0 is None:
            agg_0 = self.game.S(x_0)
        if res_0 is None:
            res_0 = bmm3(self.game.A_eq_shared, x_0) - self.game.b_eq_shared
        if aux_0 is None:
            aux_0 = jnp.zeros((self.N,m, 1))
        if dual_0 is None:
            dual_0 = aux_0
        if dual_loc_0 is None:
            dual_loc_0 = jnp.zeros((self.N,m_loc, 1))
        self.state = PrimalDualPartialInfoState(
            x=x_0,
            agg=agg_0,
            res=res_0,
            aux=aux_0,
            dual=dual_0,
            dual_loc=dual_loc_0,
        )
        # These are used to store the previous iteration value (needed for residual computation)
        self.state_last = self.state

    @jax.jit(static_argnums=(0,))
    def _update(self, old_state: PrimalDualPartialInfoState) -> PrimalDualPartialInfoState:
        x = old_state.x
        agg = old_state.agg
        res = old_state.res
        dual = old_state.dual
        dual_loc = old_state.dual_loc
        aux = old_state.aux
        A_i = self.game.A_eq_shared
        b_i = self.game.b_eq_shared
        A_i_loc = self.game.A_eq_loc
        b_i_loc = self.game.b_eq_loc
        F = self.game.F(x,agg * self.N)
        x_new = x - self.stepsize * (F + bmm3(transpose(A_i, 1,2), old_state.dual) + bmm3(transpose(A_i_loc, 1,2), old_state.dual_loc))
        dual_loc_new = dual_loc + self.stepsize * (bmm3(A_i_loc, x) - b_i_loc)
        aux_new = aux + self.stepsize * self.N * res
        # the function game.W applies the incidence matrix, the function game.S computes the aggregation
        agg_new = self.game.W(agg) + self.game.S(x_new) - self.game.S(x)
        res_new = self.game.W(res) + bmm3(A_i,x_new-x)
        dual_new = self.game.W(dual) + aux_new - aux
        return PrimalDualPartialInfoState(
            x=x_new,
            aux=aux_new,
            agg=agg_new,
            res=res_new,
            dual=dual_new,
            dual_loc=dual_loc_new,
        )

    def run_once(self):
        self.state_last = self.state
        self.state = self._update(self.state)

    def get_state(self, ref_point=None) -> tuple[
        PrimalDualPartialInfoState,
        float, float, float, float, float | None
    ]:
        residual,  constr_viol_sh, constr_viol_loc = self.compute_residual()
        cost = self.game.J(self.state.x)
        if ref_point is not None:
            dist_ref = self.compute_distance_from_ref(ref_point)
        else:
            dist_ref=None
        return self.state, residual, cost, constr_viol_sh, constr_viol_loc, dist_ref

    def compute_distance_from_ref(self, ref_x):
        x = self.state.x
        # d_avg = jnp.mean(self.dual, axis=0)
        # d_loc = self.dual_loc
        # x = jnp.reshape(x, (x.shape[0] * x.shape[1], 1))
        # d_loc = jnp.reshape(d_loc, (d_loc.shape[0] * d_loc.shape[1], 1))
        # omega_1 = jnp.hstack((x, d_avg, d_loc))
        # dist_ref = jnp.matmul(jnp.matmul(jnp.transpose(omega_1-ref_point,0,1), jnp.from_numpy(self.P)), omega_1-ref_point)
        dist_ref = jnp.linalg.norm(x-ref_x)
        return dist_ref

    def compute_residual(self) -> tuple[float, float, float]:
        # As the game is strongly monotone, the convergence is checked by x_{t+1} - x_t.
        # A_i = self.game.A_eq_shared
        # b_i = self.game.b_eq_shared
        # x = self.x
        # x_res, status = self.game.F(x) - jnp.matmul(jnp.transpose(A_i, 1, 2), self.dual)
        # d_res = bmm3(A_i, self.x) - b_i
        # residual = np.sqrt( ((x_res).norm())**2 + ((d_res).norm())**2 )

        P = self.P
        x = self.state.x
        d_avg = jnp.mean(self.state.dual, axis=0)
        A_sh = self.game.A_eq_shared
        b_sh = jnp.sum(self.game.b_eq_shared, axis=0)
        A_i_loc = self.game.A_eq_loc
        b_i_loc = self.game.b_eq_loc
        # reshape everything in a column vector
        res_x = self.game.F(x) + jnp.matmul(transpose(A_sh, 1,2), d_avg) + bmm3(transpose(A_i_loc, 1,2), self.state.dual_loc)
        res_d_sh = jnp.sum(bmm3(A_sh, x), axis=0) - jnp.sum(b_sh, axis=0)
        res_d_loc = bmm3(A_i_loc, x)- b_i_loc
        res_x = jnp.reshape(res_x, (res_x.shape[0] * res_x.shape[1], 1))
        res_d_loc = jnp.reshape(res_d_loc, (res_d_loc.shape[0] * res_d_loc.shape[1], 1) )
        res_avg_track = jnp.linalg.norm(self.state.dual - d_avg*jnp.ones(self.state.dual.shape))**2
        res_res_track = jnp.linalg.norm(self.state.res - jnp.mean(self.state.res,axis=0) * jnp.ones(self.state.res.shape))**2
        res_agg_track = jnp.linalg.norm(self.state.agg - jnp.mean(self.state.agg, axis=0) * jnp.ones(self.state.agg.shape))**2

        omega_1_res = jnp.vstack((res_x, res_d_sh, res_d_loc))
        residual = .5*jnp.matmul(jnp.matmul(transpose(omega_1_res, 0,1), P), omega_1_res) \
                   + res_avg_track + res_res_track + res_agg_track
        constr_viol_sh = jnp.linalg.norm(res_d_sh )
        constr_viol_loc = jnp.sqrt(jnp.linalg.norm(res_d_loc )**2 + \
                          jnp.linalg.norm(jnp.minimum(bmm3(self.game.A_sel_positive_vars,x), jnp.zeros(x.shape) ))**2)
        return residual.item(), constr_viol_sh, constr_viol_loc



    def compute_P_matrix(self):
        mu_F, L_F = self.game.F.get_strMon_Lip_constants()
        n = self.game.n_opt_variables
        N = self.game.N_agents
        m_sh = self.game.n_shared_eq_constr
        m_loc = self.game.n_loc_eq_constr
        list_of_A_sh_i = [self.game.A_eq_shared[i, :, :] for i in range(N)]
        list_of_A_loc_i = [self.game.A_eq_loc[i,:,:] for i in range(N)]
        A = jnp.vstack( (jnp.column_stack(list_of_A_sh_i), jsp.linalg.block_diag(*list_of_A_loc_i)) )
        mu_A, L_A = self.game.get_strMon_Lip_constants_eq_constraints()
        nu = .5 * 4 * mu_F * mu_A / (L_F * L_F * L_A * L_A + 4 * mu_A * L_A * L_A)
        P = jnp.block([
            [jnp.eye(n * N), nu * A.T],
            [nu * A, jnp.eye(m_sh + m_loc*N)],
        ])
        return P, nu

    def set_stepsize_using_Lip_const(self, safety_margin=0.5):

        mu_F, L_F = self.game.F.get_strMon_Lip_constants()
        P = self.P
        nu = self.nu
        mu_A, L_A = self.game.get_strMon_Lip_constants_eq_constraints()
        M = np.matrix([ [ mu_F-nu*L_A*L_A,    -nu*L_A*L_F/2 ],
                        [ -nu * L_A * L_F/2,   nu*mu_A      ]])
        # compute str. monotonicity and lipschitz constant in P-norm
        mu_KKT_P = np.min(np.linalg.eigvals(M).real)/np.min(np.linalg.eigvals(P).real)
        L_KKT_P_square = (L_F+L_A)*np.max(np.linalg.eigvals(P).real)/np.min(np.linalg.eigvals(P).real)
        self.stepsize = safety_margin * 2 * mu_KKT_P/L_KKT_P_square
        return P
