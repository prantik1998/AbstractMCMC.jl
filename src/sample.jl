# Default implementations of `sample`.
const PROGRESS = Ref(true)

"""
    setprogress!(progress::Bool)

Enable progress logging globally if `progress` is `true`, and disable it otherwise.
"""
function setprogress!(progress::Bool)
    @info "progress logging is $(progress ? "enabled" : "disabled") globally"
    PROGRESS[] = progress
    return progress
end

function StatsBase.sample(
    model::AbstractModel,
    sampler::AbstractSampler,
    arg;
    kwargs...
)
    return StatsBase.sample(Random.GLOBAL_RNG, model, sampler, arg; kwargs...)
end

"""
    sample([rng, ]model, sampler, N; kwargs...)

Return `N` samples from the `model` with the Markov chain Monte Carlo `sampler`.
"""
function StatsBase.sample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    N::Integer;
    kwargs...
)
    return mcmcsample(rng, model, sampler, N; kwargs...)
end

"""
    sample([rng, ]model, sampler, isdone; kwargs...)

Sample from the `model` with the Markov chain Monte Carlo `sampler` until a
convergence criterion `isdone` returns `true`, and return the samples.

The function `isdone` has the signature
```julia
isdone(rng, model, sampler, samples, iteration; kwargs...)
```
and should return `true` when sampling should end, and `false` otherwise.
"""
function StatsBase.sample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    isdone;
    kwargs...
)
    return mcmcsample(rng, model, sampler, isdone; kwargs...)
end

function StatsBase.sample(
    model::AbstractModel,
    sampler::AbstractSampler,
    parallel::AbstractMCMCEnsemble,
    N::Integer,
    nchains::Integer;
    kwargs...
)
    return StatsBase.sample(Random.GLOBAL_RNG, model, sampler, parallel, N, nchains;
                            kwargs...)
end

"""
    sample([rng, ]model, sampler, parallel, N, nchains; kwargs...)

Sample `nchains` Monte Carlo Markov chains from the `model` with the `sampler` in parallel
using the `parallel` algorithm, and combine them into a single chain.
"""
function StatsBase.sample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    parallel::AbstractMCMCEnsemble,
    N::Integer,
    nchains::Integer;
    kwargs...
)
    return mcmcsample(rng, model, sampler, parallel, N, nchains; kwargs...)
end

# Default implementations of regular and parallel sampling.

function mcmcsample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    N::Integer;
    progress = PROGRESS[],
    progressname = "Sampling",
    callback = nothing,
    discard_initial = 0,
    thinning = 1,
    chain_type::Type=Any,
    kwargs...
)
    # Check the number of requested samples.
    N > 0 || error("the number of samples must be ≥ 1")
    Ntotal = thinning * (N - 1) + discard_initial + 1

    # Start the timer
    start = time()
    local state

    @ifwithprogresslogger progress name=progressname begin
        # Determine threshold values for progress logging
        # (one update per 0.5% of progress)
        if progress
            threshold = Ntotal ÷ 200
            next_update = threshold
        end

        # Obtain the initial sample and state.
        sample, state = step(rng, model, sampler; kwargs...)

        # Discard initial samples.
        for i in 1:(discard_initial - 1)
            # Update the progress bar.
            if progress && i >= next_update
                ProgressLogging.@logprogress i/Ntotal
                next_update = i + threshold
            end

            # Obtain the next sample and state.
            sample, state = step(rng, model, sampler, state; kwargs...)
        end

        # Run callback.
        callback === nothing || callback(rng, model, sampler, sample, state, 1; kwargs...)

        # Save the sample.
        samples = AbstractMCMC.samples(sample, model, sampler, N; kwargs...)
        samples = save!!(samples, sample, 1, model, sampler, N; kwargs...)

        # Update the progress bar.
        itotal = 1 + discard_initial
        if progress && itotal >= next_update
            ProgressLogging.@logprogress itotal / Ntotal
            next_update = itotal + threshold
        end

        # Step through the sampler.
        for i in 2:N
            # Discard thinned samples.
            for _ in 1:(thinning - 1)
                # Obtain the next sample and state.
                sample, state = step(rng, model, sampler, state; kwargs...)

                # Update progress bar.
                if progress && (itotal += 1) >= next_update
                    ProgressLogging.@logprogress itotal / Ntotal
                    next_update = itotal + threshold
                end
            end

            # Obtain the next sample and state.
            sample, state = step(rng, model, sampler, state; kwargs...)

            # Run callback.
            callback === nothing || callback(rng, model, sampler, sample, state, i; kwargs...)

            # Save the sample.
            samples = save!!(samples, sample, i, model, sampler, N; kwargs...)

            # Update the progress bar.
            if progress && (itotal += 1) >= next_update
                ProgressLogging.@logprogress itotal / Ntotal
                next_update = itotal + threshold
            end
        end
    end

    # Get the sample stop time.
    stop = time()
    duration = stop - start
    stats = SamplingStats(start, stop, duration)

    return bundle_samples(
        samples, 
        model, 
        sampler,
        state,
        chain_type;
        stats=stats,
        discard_initial=discard_initial,
        thinning=thinning,
        kwargs...
    )
end

function mcmcsample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    samplers::Vector{<:AbstractSampler},
    swap_every::Integer,
    N::Integer;
    progress = PROGRESS[],
    progressname = "Replica Exchange: Sampling ($(length(samplers)) replicas)",
    callback = nothing,
    discard_initial = 0,
    thinning = 1,
    chain_type::Type=Any,
    kwargs...
)
    """
    Replica exchange
    """
    println("Replica Exchange AbstractMCMC.mcmcsample")
    function swap_β(samplers::Vector{<:AbstractSampler}, states,start::Integer,total_swap_moves,accepted_swap_moves,samples_per_replica)
        L = length(samplers) - 1
        for sampler_id in start:2:L
            println("Attempting swap b/w $sampler_id, $(sampler_id+1)")
            T = typeof(states[sampler_id].z.ℓπ)
            for (name, typ) in zip(fieldnames(T), T.types)
                println("type of the fieldname $name is $typ")
            end
            println(states[sampler_id].z.ℓπ.value)
            logα = - (samplers[sampler_id].alg.β - samplers[sampler_id + 1].alg.β) * (states[sampler_id].z.ℓπ.value - states[sampler_id+1].z.ℓπ.value)
            if log(1-Random.rand(rng)) ≤ logα
                @set samples_per_replica[sampler_id][length(samples_per_replica[sampler_id])],samples_per_replica[sampler_id+1][[length(samples_per_replica[sampler_id+1])]] = samples_per_replica[sampler_id+1][[length(samples_per_replica[sampler_id+1])]],samples_per_replica[sampler_id][[length(samples_per_replica[sampler_id])]]
                @set states[sampler_id],states[sampler_id+1] = states[sampler_id+1],states[sampler_id]
                println("Successful swap b/w $sampler_id, $(sampler_id+1)")                
            else
                println("Failed swap b/w $sampler_id, $(sampler_id+1)")
            end
            total_swap_moves[sampler_id] += 1
        end
    end
    # Check the number of requested samples.
    N > 0 || error("the number of samples must be ≥ 1")
    Ntotal = thinning * (N - 1) + discard_initial + 1

    # Start the timer
    start = time()
    local states, state, itotal
    states, samples_per_replica = [], []
    L = length(samplers)
    total_swap_moves = zeros(L-1)
    accepted_swap_moves = zeros(L-1)
    βs = [sampler.alg.β for sampler in samplers]
    @ifwithprogresslogger progress name=progressname begin
        # Determine threshold values for progress logging
        # (one update per 0.5% of progress)
        if progress
            threshold = Ntotal ÷ 200
            next_update = threshold
        end
        
        for (sampler_id, sampler) in enumerate(samplers)
            # Obtain the initial sample and state.
            sample, state = step(rng, model, sampler; kwargs...)

            push!(states, state)

            # Discard initial samples.
            for i in 1:(discard_initial - 1)
                # Update the progress bar.
                if progress && i >= next_update && sampler_id == length(samplers)
                    ProgressLogging.@logprogress i/Ntotal
                    next_update = i + threshold
                end
    
                # Obtain the next sample and state.
                sample, states[sampler_id] = step(rng, model, sampler, states[sampler_id]; kwargs...)
            end
    
            # Run callback.
            callback === nothing || callback(rng, model, sampler, sample, states[sampler_id], 1; kwargs...)
    
            # Save the sample.
            samples = AbstractMCMC.samples(sample, model, sampler, N; kwargs...)
            samples = save!!(samples, sample, 1, model, sampler, N; kwargs...)
            push!(samples_per_replica, samples)
            # Update the progress bar if it is the last replica
            itotal = 1 + discard_initial
            println(itotal)
            if progress && itotal >= next_update && sampler_id == length(samplers)
                ProgressLogging.@logprogress itotal / Ntotal
                next_update = itotal + threshold
            end
        end
        println(itotal)
        # Step through the sampler.
        for i in 2:N
            # # Discard thinned samples.
            # for _ in 1:(thinning - 1)
            #     # Obtain the next sample and state.
            #     sample, state = step(rng, model, sampler, state; kwargs...)

            #     # Update progress bar.
            #     if progress && (itotal += 1) >= next_update
            #         ProgressLogging.@logprogress itotal / Ntotal
            #         next_update = itotal + threshold
            #     end
            # end

            if i % swap_every == 0
                if i % (2*swap_every) == 0
                    swap_β(samplers, states,2,total_swap_moves,accepted_swap_moves,samples_per_replica) # swap even indices
                else
                    swap_β(samplers, states,1,total_swap_moves,accepted_swap_moves,samples_per_replica) # swap odd indices
                end
            end
            for (sampler_id, sampler) in enumerate(samplers)
                # Load state of replica 'sampler_id'
                state = states[sampler_id]
                
                # Obtain the next sample and state for the replica
                sample, state = step(rng, model, sampler, state; kwargs...)

                # Run callback for the replica
                callback === nothing || callback(rng, model, sampler, sample, state, i; kwargs...)
                
                # Save state of replica 'sampler_id'
                states[sampler_id] = state

                # Save the sample for the replica
                samples_per_replica[sampler_id] = save!!(samples_per_replica[sampler_id], sample, i, model, sampler, N; kwargs...)

                # Update the progress bar if it is the last replica
                if progress && (itotal += 1) >= next_update && sampler_id == length(samplers)
                    ProgressLogging.@logprogress itotal / Ntotal
                    next_update = itotal + threshold
                end
            end
        end
    end

    # Get the sample stop time.
    stop = time()
    duration = stop - start
    stats = SamplingStats(start, stop, duration)

    bundled_samples = Dict()
    println(accepted_swap_moves./total_swap_moves)
    for β in βs
        for (sampler_id, sampler) in enumerate(samplers)
            if sampler.alg.β == β
                state = states[sampler_id]
                samples = samples_per_replica[sampler_id]
                bundled_samples[β] =  bundle_samples(
                    samples, 
                    model, 
                    sampler,
                    state,
                    chain_type;
                    stats=stats,
                    discard_initial=discard_initial,
                    thinning=thinning,
                    kwargs...
                )
            end
        end
    end
    return bundled_samples
end

function mcmcsample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    isdone;
    chain_type::Type=Any,
    progress = PROGRESS[],
    progressname = "Convergence sampling",
    callback = nothing,
    discard_initial = 0,
    thinning = 1,
    kwargs...
)

    # Start the timer
    start = time()
    local state

    @ifwithprogresslogger progress name=progressname begin
        # Obtain the initial sample and state.
        sample, state = step(rng, model, sampler; kwargs...)

        # Discard initial samples.
        for _ in 2:discard_initial
            # Obtain the next sample and state.
            sample, state = step(rng, model, sampler, state; kwargs...)
        end

        # Run callback.
        callback === nothing || callback(rng, model, sampler, sample, state, 1; kwargs...)

        # Save the sample.
        samples = AbstractMCMC.samples(sample, model, sampler; kwargs...)
        samples = save!!(samples, sample, 1, model, sampler; kwargs...)

        # Step through the sampler until stopping.
        i = 2

        while !isdone(rng, model, sampler, samples, state, i; progress=progress, kwargs...)
            # Discard thinned samples.
            for _ in 1:(thinning - 1)
                # Obtain the next sample and state.
                sample, state = step(rng, model, sampler, state; kwargs...)
            end

            # Obtain the next sample and state.
            sample, state = step(rng, model, sampler, state; kwargs...)

            # Run callback.
            callback === nothing || callback(rng, model, sampler, sample, state, i; kwargs...)

            # Save the sample.
            samples = save!!(samples, sample, i, model, sampler; kwargs...)

            # Increment iteration counter.
            i += 1
        end
    end

    # Get the sample stop time.
    stop = time()
    duration = stop - start
    stats = SamplingStats(start, stop, duration)

    # Wrap the samples up.
    return bundle_samples(
        samples, 
        model,
        sampler, 
        state, 
        chain_type; 
        stats=stats,
        discard_initial=discard_initial,
        thinning=thinning,
        kwargs...
    )
end

function mcmcsample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    ::MCMCThreads,
    N::Integer,
    nchains::Integer;
    progress = PROGRESS[],
    progressname = "Sampling ($(min(nchains, Threads.nthreads())) threads)",
    kwargs...
)
    # Check if actually multiple threads are used.
    if Threads.nthreads() == 1
        @warn "Only a single thread available: MCMC chains are not sampled in parallel"
    end

    # Check if the number of chains is larger than the number of samples
    if nchains > N
        @warn "Number of chains ($nchains) is greater than number of samples per chain ($N)"
    end

    # Copy the random number generator, model, and sample for each thread
    # NOTE: As of May 17, 2020, this relies on Julia's thread scheduling functionality
    #       that distributes a for loop into equal-sized blocks and allocates them
    #       to each thread. If this changes, we may need to rethink things here.
    interval = 1:min(nchains, Threads.nthreads())
    rngs = [deepcopy(rng) for _ in interval]
    models = [deepcopy(model) for _ in interval]
    samplers = [deepcopy(sampler) for _ in interval]

    # Create a seed for each chain using the provided random number generator.
    seeds = rand(rng, UInt, nchains)

    # Set up a chains vector.
    chains = Vector{Any}(undef, nchains)

    @ifwithprogresslogger progress name=progressname begin
        # Create a channel for progress logging.
        if progress
            channel = Channel{Bool}(length(interval))
        end

        Distributed.@sync begin
            if progress
                # Update the progress bar.
                Distributed.@async begin
                    # Determine threshold values for progress logging
                    # (one update per 0.5% of progress)
                    threshold = nchains ÷ 200
                    nextprogresschains = threshold

                    progresschains = 0
                    while take!(channel)
                        progresschains += 1
                        if progresschains >= nextprogresschains
                            ProgressLogging.@logprogress progresschains/nchains
                            nextprogresschains = progresschains + threshold
                        end
                    end
                end
            end

            Distributed.@async begin
                try
                    Threads.@threads for i in 1:nchains
                        # Obtain the ID of the current thread.
                        id = Threads.threadid()

                        # Seed the thread-specific random number generator with the pre-made seed.
                        subrng = rngs[id]
                        Random.seed!(subrng, seeds[i])

                        # Sample a chain and save it to the vector.
                        chains[i] = StatsBase.sample(subrng, models[id], samplers[id], N;
                                                     progress = false, kwargs...)

                        # Update the progress bar.
                        progress && put!(channel, true)
                    end
                finally
                    # Stop updating the progress bar.
                    progress && put!(channel, false)
                end
            end
        end
    end

    # Concatenate the chains together.
    return chainsstack(tighten_eltype(chains))
end

function mcmcsample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    ::MCMCDistributed,
    N::Integer,
    nchains::Integer;
    progress = PROGRESS[],
    progressname = "Sampling ($(Distributed.nworkers()) processes)",
    kwargs...
)
    # Check if actually multiple processes are used.
    if Distributed.nworkers() == 1
        @warn "Only a single process available: MCMC chains are not sampled in parallel"
    end

    # Check if the number of chains is larger than the number of samples
    if nchains > N
        @warn "Number of chains ($nchains) is greater than number of samples per chain ($N)"
    end

    # Create a seed for each chain using the provided random number generator.
    seeds = rand(rng, UInt, nchains)

    # Set up worker pool.
    pool = Distributed.CachingPool(Distributed.workers())

    local chains
    @ifwithprogresslogger progress name=progressname begin
        # Create a channel for progress logging.
        if progress
            channel = Distributed.RemoteChannel(() -> Channel{Bool}(Distributed.nworkers()))
        end

        Distributed.@sync begin
            if progress
                # Update the progress bar.
                Distributed.@async begin
                    # Determine threshold values for progress logging
                    # (one update per 0.5% of progress)
                    threshold = nchains ÷ 200
                    nextprogresschains = threshold

                    progresschains = 0
                    while take!(channel)
                        progresschains += 1
                        if progresschains >= nextprogresschains
                            ProgressLogging.@logprogress progresschains/nchains
                            nextprogresschains = progresschains + threshold
                        end
                    end
                end
            end

            Distributed.@async begin
                try
                    chains = Distributed.pmap(pool, seeds) do seed
                        # Seed a new random number generator with the pre-made seed.
                        Random.seed!(rng, seed)

                        # Sample a chain.
                        chain = StatsBase.sample(rng, model, sampler, N;
                                                 progress = false, kwargs...)

                        # Update the progress bar.
                        progress && put!(channel, true)

                        # Return the new chain.
                        return chain
                    end
                finally
                    # Stop updating the progress bar.
                    progress && put!(channel, false)
                end
            end
        end
    end

    # Concatenate the chains together.
    return chainsstack(tighten_eltype(chains))
end

function mcmcsample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    ::MCMCSerial,
    N::Integer,
    nchains::Integer;
    progressname = "Sampling",
    kwargs...
)
    # Check if the number of chains is larger than the number of samples
    if nchains > N
        @warn "Number of chains ($nchains) is greater than number of samples per chain ($N)"
    end

    # Sample the chains.
    chains = map(
        i -> StatsBase.sample(rng, model, sampler, N; progressname = string(progressname, " (Chain ", i, " of ", nchains, ")"),
        kwargs...),
        1:nchains
    )

    # Concatenate the chains together.
    return chainsstack(tighten_eltype(chains))
end

tighten_eltype(x) = x
tighten_eltype(x::Vector{Any}) = map(identity, x)
