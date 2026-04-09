
from xml.parsers.expat import model
import numpy as np
import gstools as gs
from fipy import CellVariable, Grid2D, ImplicitDiffusionTerm
from fipy.solvers import LinearLUSolver

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def gaussian_points(center, max_radius, num_points):
    """Generate gaussian-distributed points around (x0,y0) inside a circle."""
    x, y = [], []
    while len(x) < num_points:
        radius = np.random.normal(0.0, 0.5 * max_radius)
        if abs(radius) > max_radius:
            continue
        angle = np.random.uniform(0, 2 * np.pi)
        x.append(center[0] + radius * np.cos(angle))
        y.append(center[1] + radius * np.sin(angle))
    return np.asarray(x), np.asarray(y)


# ----------------------------------------------------------------------------
# Main solver
# ----------------------------------------------------------------------------

class ReactiveTransport_solver:
    """Reactive transport solver with Random Walk Method (RWM).

    Key upgrades vs. the original version:
      - Fast structured (staggered) bilinear interpolation on a fixed Grid2D
        instead of matplotlib Triangulation/LinearTriInterpolator.
      - Supports transient velocity fields (u_face, v_face can be updated
        every time step).
      - Absorbing outflow boundary at x > Lx: particles leave the domain.
      - Reaction A + B -> C implemented using labels (no fixed midpoint),
        robust when particles are pruned after leaving the domain.
      - CFL time step uses dt = CFL * dx / max_speed (dimensionally correct).

    Notes:
      - Velocity interpolation is face-centered (staggered grid):
            u on vertical faces, v on horizontal faces.
      - Ito drift correction is kept:
            u' = u + dDxx/dx + dDxy/dy
            v' = v + dDyy/dy + dDxy/dx
    """

    def __init__(self, seed: int = 1234):
        # Flow solver settings
        self.tol = 1e-12
        self.iterations = 10000
        # RNG for reproducibility
        self.rng = np.random.default_rng(seed)
        # Internal flags
        self._maps_ready = False

    # ---------------------------------------------------------------------
    # Domain / properties
    # ---------------------------------------------------------------------

    def set_domain_size_discretization(self, Nx, Ny, grid_size):
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.dx = float(grid_size)
        self.Lx, self.Ly = self.Nx * self.dx, self.Ny * self.dx

    def set_permeability_porosity(self, permeability, porosity):
        self.k = np.asarray(permeability)
        self.phi = float(porosity)

    def set_permeability_logk(self, mean_log10k, sigma_ln_k, correlation_length, random_seed):
        """
        Generate correlated random field of log10(k) consistent with the model:

            k = k_m * exp( -sigma_k^2/2 + sigma_k * r(x) )

        where:
            - mean_log10k = log10(k_m)
            - sigma_ln_k  = sigma_k in the equation (std dev of ln(k))
            - correlation_length = L in the Gaussian model
        """

        x, y = range(self.Nx), range(self.Ny)

        # Input mean log10(k) into mean ln(k)
        ln10 = np.log(10.0)
        mean_lnk = mean_log10k * ln10

        # gstools expects *variance* in ln-space
        var_lnk = sigma_ln_k ** 2

        # Correlation length in grid units
        L_grid = float(correlation_length) / self.dx

        # Gaussian covariance model in ln(k)
        model = gs.Gaussian(
            dim=2,
            var=var_lnk,
            len_scale=[L_grid, L_grid]
        )

        # Generate field of ln(k)
        srf = gs.SRF(model, mean=mean_lnk, seed=int(random_seed))
        lnK = srf((x, y), mesh_type='structured')

        # Store and return log10(k) for consistency with the rest of your code
        self.logk = lnK / ln10
        return self.logk

    # ---------------------------------------------------------------------
    # Flow
    # ---------------------------------------------------------------------

    def compute_flow_field(self, dP_x, mu):
        """Solve steady-state pressure and compute face-centered velocities."""
        Dis_mesh = Grid2D(dx=self.dx, dy=self.dx, nx=self.Nx, ny=self.Ny)

        p = CellVariable(name='pressure', mesh=Dis_mesh, value=0.0)
        k = CellVariable(name='permeability', mesh=Dis_mesh, value=1.0)
        # FiPy uses 1D cell ordering; original code used self.k.T.flatten()
        k.numericValue[:] = np.asarray(self.k).T.flatten()

        # Boundary conditions: left high, right low
        p.constrain(0.0, Dis_mesh.facesRight)
        p.constrain(float(dP_x), Dis_mesh.facesLeft)

        k_face = k.harmonicFaceValue
        eq = ImplicitDiffusionTerm(coeff=k_face / float(mu))
        eq.solve(var=p, solver=LinearLUSolver(tolerance=self.tol, iterations=self.iterations, precon='jacobi'))

        # Mean velocity u = q/phi, q = -(k/mu) grad(p)
        u_face = -k_face / float(mu) * p.faceGrad.value[0] / self.phi
        v_face = -k_face / float(mu) * p.faceGrad.value[1] / self.phi

        # Store mesh
        self.mesh = Dis_mesh
        return p, Dis_mesh, u_face, v_face

    # ---------------------------------------------------------------------
    # Transport parameters
    # ---------------------------------------------------------------------

    def set_dispersivity(self, aL, aT, Dm):
        self.aL = float(aL)
        self.aT = float(aT)
        self.Dm = float(Dm)

    def set_time_steps(self, steps):
        self.timesteps = int(steps)

    def set_save_interval(self, interval):
        self.save_interval = int(interval)

    # ---------------------------------------------------------------------
    # Structured face-centered interpolation setup (fixed mesh)
    # ---------------------------------------------------------------------

    def init_face_index_maps(self, mesh=None):
        """Precompute mapping from FiPy face arrays to staggered grids.

        Call ONCE after the first mesh is available. Mesh is fixed afterwards.
        """
        if mesh is None:
            mesh = self.mesh
        xF = mesh.faceCenters.value[0]
        yF = mesh.faceCenters.value[1]
        dx = self.dx
        Nx, Ny = self.Nx, self.Ny

        # Vertical faces for u: x=i*dx, y=(j+0.5)*dx -> uX[j,i] with shape (Ny, Nx+1)
        ix_u = np.rint(xF / dx).astype(np.int32)
        jy_u = np.rint(yF / dx - 0.5).astype(np.int32)
        mask_u = (ix_u >= 0) & (ix_u <= Nx) & (jy_u >= 0) & (jy_u < Ny)
        self._u_face_ids = np.where(mask_u)[0].astype(np.intp)
        self._u_j = jy_u[mask_u].astype(np.intp)
        self._u_i = ix_u[mask_u].astype(np.intp)

        # Horizontal faces for v: x=(i+0.5)*dx, y=j*dx -> vY[j,i] with shape (Ny+1, Nx)
        ix_v = np.rint(xF / dx - 0.5).astype(np.int32)
        jy_v = np.rint(yF / dx).astype(np.int32)
        mask_v = (ix_v >= 0) & (ix_v < Nx) & (jy_v >= 0) & (jy_v <= Ny)
        self._v_face_ids = np.where(mask_v)[0].astype(np.intp)
        self._v_j = jy_v[mask_v].astype(np.intp)
        self._v_i = ix_v[mask_v].astype(np.intp)

        # Allocate staggered grids
        self.uX = np.zeros((Ny, Nx + 1), dtype=np.float32)
        self.vY = np.zeros((Ny + 1, Nx), dtype=np.float32)

        self._maps_ready = True

    def update_staggered_uv(self, u_face, v_face):
        """Update staggered velocity grids from FiPy face arrays (transient-safe)."""
        if not self._maps_ready:
            self.init_face_index_maps(self.mesh)

        self.uX.fill(0.0)
        self.vY.fill(0.0)

        uf = np.asarray(u_face, dtype=np.float32)
        vf = np.asarray(v_face, dtype=np.float32)

        self.uX[self._u_j, self._u_i] = uf[self._u_face_ids]
        self.vY[self._v_j, self._v_i] = vf[self._v_face_ids]

    # ---------------------------------------------------------------------
    # Fast bilinear interpolators
    # ---------------------------------------------------------------------

    @staticmethod
    def _bilinear_u(uX, x, y, Nx, Ny, dx):
        """Bilinear interpolation for u on vertical faces (Ny, Nx+1)."""
        fx = x / dx
        fy = y / dx - 0.5
        ix = np.floor(fx).astype(np.int32)
        iy = np.floor(fy).astype(np.int32)
        ix = np.clip(ix, 0, Nx - 1)      # ix+1 <= Nx
        iy = np.clip(iy, 0, Ny - 2)      # iy+1 <= Ny-1
        tx = fx - ix
        ty = fy - iy
        f00 = uX[iy, ix]
        f10 = uX[iy, ix + 1]
        f01 = uX[iy + 1, ix]
        f11 = uX[iy + 1, ix + 1]
        return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11

    @staticmethod
    def _bilinear_v(vY, x, y, Nx, Ny, dx):
        """Bilinear interpolation for v on horizontal faces (Ny+1, Nx)."""
        fx = x / dx - 0.5
        fy = y / dx
        ix = np.floor(fx).astype(np.int32)
        iy = np.floor(fy).astype(np.int32)
        ix = np.clip(ix, 0, Nx - 2)      # ix+1 <= Nx-1
        iy = np.clip(iy, 0, Ny - 1)      # iy+1 <= Ny
        tx = fx - ix
        ty = fy - iy
        f00 = vY[iy, ix]
        f10 = vY[iy, ix + 1]
        f01 = vY[iy + 1, ix]
        f11 = vY[iy + 1, ix + 1]
        return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11

    @staticmethod
    def _bilinear_cell(F, x, y, Nx, Ny, dx):
        """Bilinear interpolation for cell-centered field F (Ny, Nx)."""
        fx = x / dx - 0.5
        fy = y / dx - 0.5
        ix = np.floor(fx).astype(np.int32)
        iy = np.floor(fy).astype(np.int32)
        ix = np.clip(ix, 0, Nx - 2)
        iy = np.clip(iy, 0, Ny - 2)
        tx = fx - ix
        ty = fy - iy
        f00 = F[iy, ix]
        f10 = F[iy, ix + 1]
        f01 = F[iy + 1, ix]
        f11 = F[iy + 1, ix + 1]
        return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11

    # ---------------------------------------------------------------------
    # Dispersion gradients (Ito drift terms) from transient uX/vY
    # ---------------------------------------------------------------------

    def update_dispersion_gradients_from_faces(self):
        """Compute Dxx_x, Dyy_y, Dxy_x, Dxy_y on cell centers from uX/vY."""
        # Cell-centered velocities from staggered faces
        uC = 0.5 * (self.uX[:, :-1] + self.uX[:, 1:])  # (Ny, Nx)
        vC = 0.5 * (self.vY[:-1, :] + self.vY[1:, :])  # (Ny, Nx)

        umag = np.sqrt(uC * uC + vC * vC)
        umag = np.maximum(umag, 1e-20)

        Dxx = (self.aL * uC * uC + self.aT * vC * vC) / umag + self.Dm
        Dyy = (self.aT * uC * uC + self.aL * vC * vC) / umag + self.Dm
        Dxy = (self.aL - self.aT) * uC * vC / umag

        # gradients: returns (d/dy, d/dx)
        dDxx_dy, dDxx_dx = np.gradient(Dxx, self.dx, self.dx)
        dDyy_dy, dDyy_dx = np.gradient(Dyy, self.dx, self.dx)
        dDxy_dy, dDxy_dx = np.gradient(Dxy, self.dx, self.dx)

        self.Dxx_x = dDxx_dx.astype(np.float32)
        self.Dyy_y = dDyy_dy.astype(np.float32)
        self.Dxy_x = dDxy_dx.astype(np.float32)
        self.Dxy_y = dDxy_dy.astype(np.float32)

    # ---------------------------------------------------------------------
    # Particle initialization
    # ---------------------------------------------------------------------

    def set_initial_particle_position(
        self,
        A_lable, A_num, A_x, A_y,
        B_lable, B_num, B_x, B_y,
        C_lable, shape
    ):
        """Initialize A and B particles (and label C for products)."""
        self.color_A, self.color_B, self.color_C = A_lable, B_lable, C_lable

        A_num = int(A_num)
        B_num = int(B_num)

        if shape == "line":
            Ax = np.linspace(A_x[0], A_x[1], A_num)
            Ay = np.linspace(A_y[0], A_y[1], A_num)
            Bx = np.linspace(B_x[0], B_x[1], B_num)
            By = np.linspace(B_y[0], B_y[1], B_num)
        elif shape == "disc":
            Ax, Ay = gaussian_points([A_x[0], A_y[0]], 0.5 * (A_x[1] + A_y[1]), A_num)
            Bx, By = gaussian_points([B_x[0], B_y[0]], 0.5 * (B_x[1] + B_y[1]), B_num)
        else:
            raise ValueError("shape must be 'line' or 'disc'.")

        self.pos_x = np.concatenate([Ax, Bx]).astype(np.float64)
        self.pos_y = np.concatenate([Ay, By]).astype(np.float64)
        self.label = np.concatenate([
            np.repeat(self.color_A, A_num),
            np.repeat(self.color_B, B_num)
        ]).astype('<U2')

    # ---------------------------------------------------------------------
    # Random walk step (with Ito drift), reflections, and open outflow
    # ---------------------------------------------------------------------

    def randomwalk(self, pos_x, pos_y, dt):
        """Advance particles by one RWM step.

        Returns:
          pos_x, pos_y, alive
        where alive is a boolean mask for particles with x <= Lx (inside/outflow).
        """
        # Interpolate face-centered velocity at particle positions
        u = self._bilinear_u(self.uX, pos_x, pos_y, self.Nx, self.Ny, self.dx)
        v = self._bilinear_v(self.vY, pos_x, pos_y, self.Nx, self.Ny, self.dx)

        u_norm = np.sqrt(u * u + v * v)
        u_norm = np.maximum(u_norm, 1e-20)

        # Ito drift correction terms (computed on cell centers, interpolated)
        Dxx_x = self._bilinear_cell(self.Dxx_x, pos_x, pos_y, self.Nx, self.Ny, self.dx)
        Dyy_y = self._bilinear_cell(self.Dyy_y, pos_x, pos_y, self.Nx, self.Ny, self.dx)
        Dxy_x = self._bilinear_cell(self.Dxy_x, pos_x, pos_y, self.Nx, self.Ny, self.dx)
        Dxy_y = self._bilinear_cell(self.Dxy_y, pos_x, pos_y, self.Nx, self.Ny, self.dx)

        u_prime = u + Dxx_x + Dxy_y
        v_prime = v + Dyy_y + Dxy_x

        n = pos_x.size
        Z1 = self.rng.standard_normal(n)
        Z2 = self.rng.standard_normal(n)

        BL = np.sqrt(2.0 * dt * (self.aL * u_norm + self.Dm)) / u_norm
        BT = np.sqrt(2.0 * dt * (self.aT * u_norm + self.Dm)) / u_norm

        pos_x = pos_x + u_prime * dt + Z1 * BL * u - Z2 * BT * v
        pos_y = pos_y + v_prime * dt + Z1 * BL * v + Z2 * BT * u

        # Reflecting boundaries: bottom/top/left
        idx = pos_y < 0.0
        pos_y[idx] = -pos_y[idx]

        idx = pos_y > self.Ly
        pos_y[idx] = 2.0 * self.Ly - pos_y[idx]

        idx = pos_x < 0.0
        pos_x[idx] = -pos_x[idx]

        # Absorbing outflow at right boundary (particles leave domain)
        alive = pos_x <= self.Lx
        return pos_x, pos_y, alive

    # ---------------------------------------------------------------------
    # Reaction A + B -> C (label-based; robust after pruning)
    # ---------------------------------------------------------------------

    def reaction(self, pos_x=None, pos_y=None, label=None):
        """Perform reaction A + B -> C within each grid cell (particle-count conserving).

        Project rule in each cell:
          - If there are m particles of A and n particles of B, then nr = min(m, n)
            reactions occur.
          - After reaction: (m-nr) A remain, (n-nr) B remain, and nr particles of C exist.

        Particle implementation (consistent with the rule above):
          - For each of the nr reactions, we keep exactly ONE of the two reacted
            particles as product C, and REMOVE the other reacted particle.
          - The kept particle is chosen randomly from A or B (50/50), so the product
            location is randomly taken from the reacted A/B locations.

        Notes:
          - This avoids the (incorrect for A+B->C) behavior of converting BOTH the reacted
            A and B particles into C (which would create 2C per reaction event).
          - Args are kept for backward compatibility; this method updates self.pos_x,
            self.pos_y, self.label.
        """

        # Operate on current particle state (robust after outflow pruning)
        label = self.label
        pos_x = self.pos_x
        pos_y = self.pos_y

        idxA = (label == self.color_A)
        idxB = (label == self.color_B)
        if (not idxA.any()) or (not idxB.any()):
            return

        # Cell id for each particle
        ix = np.floor(pos_x / self.dx).astype(np.int32)
        iy = np.floor(pos_y / self.dx).astype(np.int32)
        ix = np.clip(ix, 0, self.Nx - 1)
        iy = np.clip(iy, 0, self.Ny - 1)
        cell_id = ix + self.Nx * iy

        # Reactant indices and their cell ids
        A_ids = cell_id[idxA]
        B_ids = cell_id[idxB]
        A_idx = np.nonzero(idxA)[0]
        B_idx = np.nonzero(idxB)[0]

        # Sort by cell for grouping
        A_sort = np.argsort(A_ids)
        B_sort = np.argsort(B_ids)
        A_ids_s = A_ids[A_sort]
        B_ids_s = B_ids[B_sort]
        A_idx_s = A_idx[A_sort]
        B_idx_s = B_idx[B_sort]

        A_cells, A_counts = np.unique(A_ids_s, return_counts=True)
        B_cells, B_counts = np.unique(B_ids_s, return_counts=True)
        A_starts = np.cumsum(np.r_[0, A_counts[:-1]])
        B_starts = np.cumsum(np.r_[0, B_counts[:-1]])

        inter_cells, ia, ib = np.intersect1d(A_cells, B_cells, return_indices=True)
        if inter_cells.size == 0:
            return

        drop_blocks = []

        for a_i, b_i in zip(ia, ib):
            na = int(A_counts[a_i])
            nb = int(B_counts[b_i])
            nr = na if na < nb else nb
            if nr <= 0:
                continue

            a0 = int(A_starts[a_i])
            b0 = int(B_starts[b_i])
            a_pool = A_idx_s[a0:a0 + na]
            b_pool = B_idx_s[b0:b0 + nb]

            # choose reacting subsets
            choose_a = self.rng.choice(a_pool, size=nr, replace=False)
            choose_b = self.rng.choice(b_pool, size=nr, replace=False)

            # keep one particle as C, drop the other
            keep_from_A = self.rng.random(nr) < 0.5
            keep_idx = np.where(keep_from_A, choose_a, choose_b)
            drop_idx = np.where(keep_from_A, choose_b, choose_a)

            label[keep_idx] = self.color_C
            drop_blocks.append(drop_idx)

        if drop_blocks:
            drop_idx_all = np.concatenate(drop_blocks)
            mask = np.ones(label.shape[0], dtype=bool)
            mask[drop_idx_all] = False
            self.pos_x = pos_x[mask]
            self.pos_y = pos_y[mask]
            self.label = label[mask]


    def breakthrough_counts(self, x_min=199.0, x_max=201.0, y_min=0.0, y_max=100.0):
        """Count particles A/B/C inside observation zone."""
        in_obs = (
            (self.pos_x >= x_min) & (self.pos_x <= x_max) &
            (self.pos_y >= y_min) & (self.pos_y <= y_max)
        )
        A = int(np.sum(in_obs & (self.label == self.color_A)))
        B = int(np.sum(in_obs & (self.label == self.color_B)))
        C = int(np.sum(in_obs & (self.label == self.color_C)))
        return A, B, C

    # ---------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------

    def ReactiveRandomWalk(self, u_face, v_face, CFL=0.1, velocity_updater=None):
        """Run reactive random walk.

        Args:
          u_face, v_face: initial FiPy face velocity arrays (steady or first transient step)
          CFL: CFL coefficient for dt
          velocity_updater: optional callable(step:int, t:float) -> (u_face, v_face)
            If provided, velocities are updated every step (transient flow).

        Returns:
          pos_x_list, pos_y_list, label_list, t_list, btc_list
          where lists contain snapshots every save_interval.

        Notes:
          - Particle counts can decrease due to outflow; lists contain arrays of varying length.
          - Reaction is applied only for particles remaining in-domain.
        """
        if not hasattr(self, 'mesh'):
            raise RuntimeError("Mesh not found. Call compute_flow_field(...) at least once to set self.mesh.")

        if not self._maps_ready:
            self.init_face_index_maps(self.mesh)

        # Initial field update
        self.update_staggered_uv(u_face, v_face)
        self.update_dispersion_gradients_from_faces()

        # dt based on max speed (dimensionally correct)
        speed_max = float(np.max(np.sqrt(np.asarray(u_face) ** 2 + np.asarray(v_face) ** 2)))
        speed_max = max(speed_max, 1e-20)
        self.dt = float(CFL) * self.dx / speed_max

        # time axis
        self.t_space = np.arange(1, self.timesteps + 1) * self.dt

        # storage (lists, because particle number changes)
        pos_x_list = [self.pos_x.copy()]
        pos_y_list = [self.pos_y.copy()]
        label_list = [self.label.copy()]
        t_list = [0.0]
        btc_list = [self.breakthrough_counts()]

        for step, t in enumerate(self.t_space, start=1):
            # update velocities if transient
            if velocity_updater is not None:
                u_face, v_face = velocity_updater(step, t)
                self.update_staggered_uv(u_face, v_face)
                self.update_dispersion_gradients_from_faces()

                # optional: update dt with transient max speed
                speed_max = float(np.max(np.sqrt(np.asarray(u_face) ** 2 + np.asarray(v_face) ** 2)))
                speed_max = max(speed_max, 1e-20)
                self.dt = float(CFL) * self.dx / speed_max

            # transport step
            self.pos_x, self.pos_y, alive = self.randomwalk(self.pos_x, self.pos_y, self.dt)

            # prune exited particles (absorbing outflow)
            if not np.all(alive):
                self.pos_x = self.pos_x[alive]
                self.pos_y = self.pos_y[alive]
                self.label = self.label[alive]

            # reaction inside domain
            self.reaction(self.pos_x, self.pos_y, self.label)

            # save snapshots
            if step % self.save_interval == 0:
                pos_x_list.append(self.pos_x.copy())
                pos_y_list.append(self.pos_y.copy())
                label_list.append(self.label.copy())
                t_list.append(float(t))
                btc_list.append(self.breakthrough_counts())

            # early stop if all particles have left the domain
            if self.pos_x.size == 0:
                break

        return pos_x_list, pos_y_list, label_list, t_list, btc_list
