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

function ∇chi_squared(model::Vector{T}, dmodel::Matrix{T}, data::Vector{T}, errors::Vector{T}, dnll::Vector{T}) where T<:Real
    res = (data .- model) ./ errors
    # Iterate over each model parameter
    for i in eachindex(dmodel[:,1])
        dnll[i] += -2.0 * dot(dmodel[i,:] ./ errors, res)
    end
end

# Bounds for optimizer
function get_upper_open(elements)
    m,P,t0,ecosϖ,esinϖ,I,Ω = unpack_elements(elements)
    N = length(m)
    ms = 2.0 .* m[:]
    Ps = 5.0 .* P[:]
    t0s = t0[:] .+ (1.1 .* Ps[:])
    ecs = ones(N)
    ess = ones(N)
    Is = ones(N) .* π
    Is[1] = (π/2 + π/6)
    Ωs = ones(N) .* π
    return [ms...,Ps...,t0s...,ecs...,ess...,Is...,Ωs...]
end

function get_lower_open(elements)
    m,P,t0,ecosϖ,esinϖ,I,Ω = unpack_elements(elements)
    N = length(m)
    ms = ones(N) .* 1e-10
    Ps = ones(N) .* 0.5
    t0s = t0[:] .- (2.0 .* Ps[:])
    ecs = -ones(N)
    ess = -ones(N)
    Is = zeros(N)
    Is[1] = (π/2 - π/6)
    Ωs = ones(N) .* -π
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
    data::TimingData, t0::T, intr::Integrator; parameter_mask=nothing) where T <: Real
    
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

    # calculate -loglikelihood function
    params = elements[parameter_mask][:]
    N_params = sum(parameter_mask)
    nll = 0.0
    for i in 1:length(data.N)
        nll += lnlike(tt.tt[i+1, data.nt[i]], data.tt[i], data.tterr[i])
    end

    # Now gradients 
    dnll = zeros(T, N_params)
    N = length(data.N); n = sum(parameter_mask[:,1])
    for i in 1:N # <- Make this a list of planet indices..
        dmodel = zeros(N_params, length(data.tt[i]))
        for k in 1:data.N[i]
            dmodel[n+1:end,k] = tt.dtdelements[i+1, k, 1:6, :]'[parameter_mask[:,2:end]][:]
            dmodel[1:n,k] = tt.dtdelements[i+1, k, 7, :]'[parameter_mask[:,1]][:]
        end
        dlnlike(tt.tt[i+1, data.nt[i]], dmodel, data.tt[i], data.tterr[i], dnll)
    end

    # Add priors here

    return nll, dnll
end


    
