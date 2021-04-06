"""
    SamplingStats

A struct that tracks sampling information. 

The fields available are:

- `start_time`: A `DateTime` indicating when sampling begun.
- `stop_time`: A `DateTime` indicating when sampling finished.
- `step_calls`: The number of times `step!` was called.
- `step_time`: The total number of seconds spent inside `step!` calls.
- `allocations`: The total number of bytes allocated by `step!`.
"""
struct SamplingStats
    start::Float64
    stop::Union{Float64, Missing}
    duration::Union{Float64, Missing}
    allocations::Int64
end