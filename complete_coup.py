# Climate model

import time
from functions_barycentric_mesh import *

U = 10 # characteristic velocity
L = 10**6 # characteristic length
f = 10**-5 # f = 2*omega*sin(phi) coriolis parameter. This corresponds to 5-10 degrees of latitude

nu_a = 10**6 # kinematic viscosity of air at 25 degrees celcius is roughly equal to 10^-5
D_a = 10**6 # diffusion coefficient of air is roughly equal to 10^-5
# characteristic time = 10**5 (1 day)
Molar_mass = 10**-2 # Average molar mass of air = 29*10^-3
R = 10 # Gas constant = 8.314
THETA = U*f*L*Molar_mass/R # Temperature is nondimensionalized using this values
# typ_temp = 300 # Typical temperature value in Kelvin
# dim_temp = typ_temp/THETA # Typical nondimensionalized temperature value
Rossby = U/(f*L)
Reynolds_a = (L*U)/nu_a
Peclet_a = (L*U)/D_a

nu_o = 10**5 # kinematic viscosity of ocean surface water at 25 degrees celcius is 10^-6
D_o = 10**4 # diffusion coefficient of ocean surface water is roughly equal to 10^-7
rho_o = 10**3 # ocean density
P = f*rho_o*U*L # characteristic pressure = 10**2 (=f*rho*U*L)
Reynolds_o = (L*U)/nu_o
Peclet_o = (L*U)/D_o


N = 50

bmh = PeriodicUnitSquareBaryMeshHierarchy(N,0)
M =  bmh[-1]

V_1 = VectorFunctionSpace(M, "CG", 2)
V_1_out = VectorFunctionSpace(M, "CG", 1)
V_2 = FunctionSpace(M, "CG", 1)
V_3 = FunctionSpace(M, "DG", 1)

Z = V_1*V_2*V_1*V_3*V_2

w = Function(Z)
ua,Ta,uo,p,To = split(w)
va,phi,vo,q,psi = TestFunctions(Z)
ua_ = Function(V_1)
uo_ = Function(V_1)

Dt = 0.1*(1/N) # CFL condition
half = Constant(0.5)

x,y = SpatialCoordinate(M)

# dimensionless constants for atmosphere
Ro_a = Constant(Rossby) # Rossby number
Re_a = Constant(Reynolds_a) # Reynolds number
Pe_a = Constant(Peclet_a) # Peclet number

# dimensionless constants for ocean
Ro_o = Constant(Rossby) # Rossby number
Re_o = Constant(Reynolds_o) # Reynolds number
Pe_o = Constant(Peclet_o) # Peclet number

# cone = 1.0 - min_value(sqrt(pow(x-0.5, 2) + pow(y-0.5, 2))/0.15, 1.0)
bell = 0.5*(1+cos(math.pi*min_value(sqrt(pow(x-0.5, 2) + pow(y-0.5, 2))/0.25, 1.0)))
i_uo = project(as_vector([Constant(0),Constant(0)]), V_1)
To_ = Function(V_2).interpolate(3000 + 50*bell)
p_= Function(V_3).interpolate(Constant(0))

# i_ua = project(as_vector([(sin(pi*y))**2, 0]), V_1)
i_ua = project(as_vector([Constant(0),Constant(0)]), V_1)
Ta_ = Function(V_2).interpolate(Constant(3000))

gamma = -Constant(1.0)
sigma = -Constant(1.0)
ua_.assign(i_ua)
uo_.assign(i_uo)


F = ( inner(ua-ua_,va)
    + Dt*half*(1/Ro_a)*(-(ua[1]+ua_[1])*va[0] +(ua[0]+ua_[0])*va[1])
    + Dt *half *(1/Re_a)*inner((nabla_grad(ua)+nabla_grad(ua_)), nabla_grad(va))
    + Dt*half*(inner(dot(ua, nabla_grad(ua)), va) + inner(dot(ua_, nabla_grad(ua_)), va))
    + Dt*half*(1/Ro_a)*inner((grad(Ta)+grad(Ta_)),va)
    - Dt*gamma*half*(Ta - To + Ta_ - To_)* phi
    + (Ta -Ta_)*phi + Dt*half*(inner(ua_,grad(Ta_)) + inner(ua,grad(Ta)))*phi
    + Dt*half*(1/Pe_a)*inner((grad(Ta)+grad(Ta_)),grad(phi))
    + inner(uo-uo_,vo)
    + Dt*half*(1/Ro_o)*(-(uo[1]+uo_[1])*vo[0] +(uo[0]+uo_[0])*vo[1])
    + Dt *half *(1/Re_o)*inner((nabla_grad(uo)+nabla_grad(uo_)), nabla_grad(vo))
    + Dt*half*(inner(dot(uo, nabla_grad(uo)), vo) + inner(dot(uo_, nabla_grad(uo_)), vo))
    - Dt*(1/Ro_o)*p*div(vo) + Dt*div(uo)*q
    - Dt*half*sigma*inner((uo_ - ua_ + uo -ua),vo)
    - Dt*sigma*inner(project(as_vector([Constant(assemble(ua_[0]*dx)),Constant(assemble(ua_[1]*dx))]), V_1),vo)
    + (To -To_)*psi + Dt*half*(inner(uo_,grad(To_))+inner(uo,grad(To)))*psi
    + Dt*half*(1/Pe_o)*inner((grad(To)+grad(To_)),grad(psi)) )*dx

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), Z.sub(1), Z.sub(2),VectorSpaceBasis(constant=True), Z.sub(4)])

p_.rename("Ocean_pressure")
To_.rename("Ocean_temperature")
Ta_.rename("Atm_temperature")

outfile = File("CC_1.pvd")
outfile.write(project(i_ua,V_1_out, name= "atm_velocity"),Ta_,project(i_uo,V_1_out, name= "ocean_velocity"),p_,To_)

t = Dt
iter = 1
end = 0.2
freq = 5 # printing results after every freq solves
t_step = freq*Dt  # printing time step

#---------------------------------#
lines = ["CLIMATE MODEL (Equations are non-dimensionalized with respect to atmosphere characteristic values) \n",
        "Characteristic length (in meters), L = "+ str(L) + "\n", "Characteristic time (in seconds) = "+ str(L/U) + "\n",
        "Characteristic velocity (in meters/second), U = "+ str(U) + "\n","Characteristic pressure (in Pascals), P = "+ str(P) + "\n",
        "Characteristic temperature (in Kelvin) = "+ str(THETA) + "\n","Coriolis parameter, f = "+ str(f) + "\n",
        "Kinematic viscosity of air (assumption) = "+str(nu_a)+"\n", "Thermal difffusivity of air (assumption) = "+str(D_a)+"\n",
        "Kinematic viscosity of ocean (assumption) = "+str(nu_o)+"\n", "Thermal difffusivity of ocean (assumption) = "+str(D_o)+"\n",
        "Coupling coefficients, sigma = "+str(-1.0)+", gamma = "+str(-1.0)+"\n",
        "Rossby number = "+str(format(Rossby,"1.1e")) + "\n",
        "Reynolds number (atmosphere) = "+ str(format(Reynolds_a,"1.1e")) + "\n",
        "Peclet number (atmosphere) = "+str(format(Peclet_a,"1.1e")) + "\n",
        "Reynolds number (ocean) = "+ str(format(Reynolds_o,"1.1e")) + "\n",
        "Peclet number (ocean) = "+str(format(Peclet_o,"1.1e")) + "\n",
        "Domain: 1x1 \n", "Mesh element size: " + str(1/N) + "\n",
        "Time step: "+ str(Dt)+ "\n", "End time: "+str(end)+"\n",
        "Initial ocean velocity: [Constant(0),Constant(0)] \n",
        "Initial ocean temperature: 3000 + 50*bell \n",
        "Initial atmospheric velocity: [Constant(0), Constant(0)] \n",
        "Initial atmospheric temperature: Constant(3000) \n",
        "Printing results at every "+str(t_step)+" time steps (we call it t_step)"+ "\n"]
with open('CC_1.txt','w') as file:
    file.writelines(lines)
#---------------------------------#
current_time = time.strftime("%H:%M:%S", time.localtime())
print("Local time at the start of simulation:",current_time)
start_time = time.time()

div_ua_at_t_0 = sqrt(assemble(((div(ua_))**2)*dx))
print("Divergence_check_atmosphere, L-2_norm_of_div(ua)_at_t=0:", div_ua_at_t_0)
div_uo_at_t_0 = sqrt(assemble(((div(uo_))**2)*dx))
print("Divergence_check_ocean, L-2_norm_of_div(uo)_at_t=0:", div_uo_at_t_0)
with open('CC_1.txt','a') as file:
    file.write("L-2 norm of atm velocity at t=0: "+str(div_ua_at_t_0)+"\n")
    file.write("L-2 norm of ocean velocity at t=0: "+str(div_uo_at_t_0)+"\n")

while (round(t,4)<=end):

    solve(F==0, w)

    ua,Ta,uo,p,To= w.split()

    if iter%freq==0:
        if iter==freq:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 # running time for one time step (t_step)
            print("Approx. running time for one t_step: %.2f minutes" %execution_time)
            total_execution_time = (end/t_step)*execution_time
            print("Approx. total running time: %.2f minutes:" %total_execution_time)
            print("Approx total running time: %.2f hours:"%(total_execution_time/60))
            with open('CC_1.txt','a') as file:
                file.write("Approx. running time for one t_step (in minutes): "+str(round(execution_time,2))+"\n")
                # file.write("Approx. total running time (in minutes): "+str(round(total_execution_time,2))+"\n")
                # file.write("Approx. total running time (in hours): "+str(round(total_execution_time/60,2))+"\n")
        print("t=", round(t,4))
        div_ua_at_t = sqrt(assemble(((div(ua))**2)*dx))
        print("Divergence_check_atmosphere, L-2_norm_of_div(ua)_at_this_time:", div_ua_at_t)
        div_uo_at_t = sqrt(assemble(((div(uo))**2)*dx))
        print("Divergence_check_ocean, L-2_norm_of_div(uo)_at_this_time:", div_uo_at_t)
        with open('CC_1.txt','a') as file:
            file.write("L-2 norm of atm velocity at t= "+str(round(t,4))+" is "+str(div_ua_at_t)+"\n")
            file.write("L-2 norm of ocean velocity at t= "+str(round(t,4))+" is "+str(div_uo_at_t)+"\n")
        p.rename("Ocean_pressure")
        To.rename("Ocean_temperature")
        Ta.rename("Atm_temperature")
        outfile.write(project(ua,V_1_out, name= "atm_velocity"),Ta,project(uo,V_1_out, name= "ocean_velocity"),p,To)

    ua_.assign(ua)
    Ta_.assign(Ta)
    uo_.assign(uo)
    To_.assign(To)
    # ua_avg = project(as_vector([assemble(ua[0]*dx),assemble(ua[1]*dx)]), V_1)

    t += Dt
    iter +=1

# the solution diverges after t=0.11. Maybe mesh needs to be more refined.
