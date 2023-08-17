import numpy as np

from mpi4py import MPI
from petsc4py import PETSc
from tqdm import trange
from dolfinx import fem, mesh, io, plot
import ufl



def get_sol(i):
    # Define temporal parameters
    t = 0 # Start time
    T = 1.0 # Final time
    num_steps = 101     
    dt = T / num_steps # time step size

    # Define mesh
    nx, ny = 20, 20
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-2, -2]), np.array([2, 2])], 
                                [nx, ny], mesh.CellType.triangle)    
    V = fem.FunctionSpace(domain, ("Lagrange", 3))
    # Create initial condition
    def initial_condition(x, a=1+i/99*0.5):
        return 1+np.exp(-a*(x[0]**2+x[1]**2))
    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.interpolate(initial_condition)
    # Create boundary condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(PETSc.ScalarType(1), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

    # Define solution variable, and interpolate initial solution for visualization in Paraview
    uh = fem.Function(V)
    uh.name = "uh"
    uh.interpolate(initial_condition)
    u_tot=np.zeros((num_steps+1,*np.array(uh.vector).shape))
    u_tot[0]=np.array(uh.vector)


    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    f = fem.Constant(domain, PETSc.ScalarType(0))
    a = u * v * ufl.dx + dt*ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx 
    L = (u_n + dt * f) * v * ufl.dx
    bilinear_form = fem.form(a)
    linear_form = fem.form(L)
    A = fem.petsc.assemble_matrix(bilinear_form, bcs=[bc])
    A.assemble()
    b = fem.petsc.create_vector(linear_form)
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    for i in range(num_steps):
        t += dt

        # Update the right hand side reusing the initial vector
        with b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(b, linear_form)
        
        # Apply Dirichlet boundary condition to the vector
        fem.petsc.apply_lifting(b, [bilinear_form], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, [bc])

        # Solve linear problem
        solver.solve(b, uh.vector)
        uh.x.scatter_forward()

        # Update solution at previous time step (u_n)
        u_n.x.array[:] = uh.x.array

        # Write solution to file
        u_tot[i+1]=np.array(uh.vector)
        # Update plot
    return u_tot
print("ciao")
tmp=get_sol(0)
print("ciao")
inputs=np.zeros((100,tmp.shape[0],2))
all=np.zeros((100,*tmp.shape))
for i in trange(100):
    all[i]=get_sol(i)
    for k in range(tmp.shape[0]): 
        inputs[i,k,0]=k/(tmp.shape[0]-1)
        inputs[i,k,1]=1+i/99*0.5
np.save("outputs.npy",all)
np.save("inputs.npy",inputs)