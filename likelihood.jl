using ForwardDiff, LinearAlgebra

struct TimingData{T<:AbstractFloat}
    N::Vector{Int64}          # Number of transits for each body
    nt::Vector{Vector{Int64}} # Transit numbers
    tt::Vector{Vector{T}}     # Transit times for each body
    tterr::Vector{Vector{T}}  # Transit time errors
    BJD::T                    # Timing offset
end

function chi_squared(model::Vector{T}, data::Vector{T}, errors::Vector{T}) where T<:Real
    res = (data .- model) ./ errors
    return dot(res,res)
end

function chi_squared(model::T, data::T, error::T) where T<:Real
    res = (data - model) / error
    return res * res
end

function ∇chi_squared(model::Vector{T}, dmodel::Matrix{T}, data::Vector{T}, errors::Vector{T}, dll::Vector{T}) where T<:Real
    res = (data .- model) ./ (errors .* errors)
    # Iterate over each model parameter
    for i in eachindex(dmodel[:,1])
        dll[i] -= @views 2.0 * dot(dmodel[i,:], res)
    end
end

function ∇chi_squared(model::T, dmodel::Vector{T}, data::T, error::T, dll::Vector{T}) where T<:Real
    res = (data - model) / error
    for i in eachindex(dmodel)
        dll[i] -= @views 2.0 .* dmodel[:] * res / error
    end
    return
end

# Hessian of nll wrt model, data, error
chi_squared(p) = chi_squared(p[1], p[2], p[3])
hess_chi_squared = p -> ForwardDiff.hessian(chi_squared, p)

# Bounds for optimizer
function get_upper_open(elements)
    m,P,t0,ecosϖ,esinϖ,I,Ω = unpack_elements(elements)
    N = length(m)
    ms = 1.1 .* m[:]
    Ps = 1.2 .* P[:]
    t0s = t0[:] .+ (1.1 .* Ps[:])
    ecs = ones(N) .* 0.5
    ess = ones(N) .* 0.5
    Is = ones(N) .* (π/2 + π/8)
    Ωs = ones(N) .* 0.1
    return [ms...,Ps...,t0s...,ecs...,ess...,Is...,Ωs...]
end

function get_lower_open(elements)
    m,P,t0,ecosϖ,esinϖ,I,Ω = unpack_elements(elements)
    N = length(m)
    ms = ones(N) .* 1e-6
    Ps = ones(N) .* 0.9
    t0s = t0[:] .- (2.0 .* Ps[:])
    ecs = -ones(N) .* 0.5
    ess = -ones(N) .* 0.5
    Is = ones(N) .* (π/2 - π/8)
    Ωs = ones(N) .* -0.1
    return [ms...,Ps...,t0s...,ecs...,ess...,Is...,Ωs...]
end  

unpack_elements(el) = [el[:,i] for i in 1:7]

function logP(lnlike::Function, elements::Matrix{T}, H::Vector{Int64}, data::TimingData, t0::T, 
    intr::Integrator; parameter_mask=nothing, grad::Bool=true) where T <: Real
    
    ## Check things here ##
    if parameter_mask == nothing
        # Assume all planet parameters, and fully-nested orbits
        parameter_mask = zeros(Bool, size(elements))
        parameter_mask[2:end,:] .= true
    end

    # Setup and run simulation
    ic = ElementsIC(t0,H,elements)
    s = State(ic)
    tt = TransitTiming(intr.tmax, ic)
    intr(s,tt,grad=grad)

    # calculate -loglikelihood function
    params = elements[parameter_mask][:]
    if @generated
        nll = :(0.0)
        for i in 1:length(data.N)
            nll = :($nll + lnlike(tt.tt[$i + 1, data.nt[$i]],data.tt[$i],data.tterr[$i],grad)) 
        end
        return :($nll)
    else
        nll = 0.0
        for i in 1:length(data.N)
            nll += lnlike(tt.tt[i+1,data.nt[i]],data.tt[i],data.tterr[i],grad) 
        end
        return nll
    end
end

function logP(lnlike::Function, dlnlike::Function, elements::Matrix{T}, H::Vector{Int64}, 
    data::TimingData, t0::T, intr::Integrator; parameter_mask=nothing, grad=true) where T <: Real
    
    ## Check things here ##
    if parameter_mask == nothing
        # Assume all planet parameters, and fully-nested orbits
        parameter_mask = zeros(Bool, size(elements))
        parameter_mask[2:end,:] .= true
    end
    
    # Check eccentricity
    e = zeros(length(elements[2:end,1]))
    e = sqrt.(elements[:,4].^2 .+ elements[:,5].^2)
    if any(e .> 1.0) || any(e .< 0.0)
        if grad; return 1e30, ones(length(elements[2:end,:][:])) * 1e30; end
        return 1e30
    end
    
    # Setup and run simulation
    ic = ElementsIC(t0, H, elements)
    s = State(ic)
    tt = TransitTiming(intr.tmax, ic)
    intr(s, tt)

    # calculate -loglikelihood function
    params = elements[parameter_mask][:]
    N_params = sum(parameter_mask)
    ll = 0.0
    for i in 1:length(data.N)
        ll += lnlike(tt.tt[i+1, data.nt[i]], data.tt[i], data.tterr[i])
    end

    # Now gradients 
    dll = zeros(T, N_params)
    N = length(data.N); n = sum(parameter_mask[:,1])
    inds = [7,1,2,3,4,5,6]
    for i in 1:N # <- Make this a list of planet indices..
        dmodel = zeros(N_params, length(data.tt[i])) # Find a way to allocate this outside loop...
        for k in 1:data.N[i]
            dmodel[:,k] = tt.dtdelements[i+1, data.nt[i][k], inds, :]'[parameter_mask][:]
        end
        dlnlike(tt.tt[i+1, data.nt[i]], dmodel, data.tt[i], data.tterr[i], dll)
    end
    
    # Add priors here
    return ll, dll
end

function calc_hessian(hess_func::Function, elements::Matrix{T}, H::Vector{Int64}, 
    data::TimingData, t0::T, intr::Integrator; parameter_mask=nothing) where T<:Real
    
    ## Check things here ##
    if parameter_mask == nothing
        # Assume all planet parameters, and fully-nested orbits
        parameter_mask = zeros(Bool, size(elements))
        parameter_mask[2:end,:] .= true
    end

    # Setup and run simulation
    ic = ElementsIC(t0, H, elements)
    s = State(ic)
    tt = TransitTiming(intr.tmax, ic)
    intr(s, tt)
    
    # Calculate hessian
    inds = [7,1,2,3,4,5,6]
    N_params = sum(parameter_mask); N = length(data.N)
    hessian = zeros(T, N_params, N_params)
    for i in 1:N
        dmodel = zeros(N_params, length(data.tt[i]))
        hmodel = zeros(length(data.tt[i]))
        for k in 1:data.N[i]
            dmodel[:,k] .= tt.dtdelements[i+1, data.nt[i][k], inds, :]'[parameter_mask][:]
        end
        for k in 1:N_params, l in 1:N_params
            for j in 1:length(data.tt[i])
                hmodel[j] = hess_func([tt.tt[i+1, data.nt[i][j]], data.tt[i][j], data.tterr[i][j]])[1]
            end
            hessian[k,l] -= sum(hmodel .* dmodel[k,:] .* dmodel[l,:] ./ (data.tterr[i].^2))
        end
    end
    return 2.0 .* hessian
end

    
