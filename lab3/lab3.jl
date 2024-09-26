using LinearAlgebra
using Random

## Генерация данных

function generate_matrix(n::Int64)::Matrix{Float64}
    return rand(-100:0.01:100, n, n)
end
function generate_matrix_delta(n::Int64)::Matrix{Float64}
    return rand(-10:0.01:10, n, n)
end

function generate_vector(n::Int64)::Vector{Float64}
    return rand(-100:0.01:100, n)
end
function generate_vector_delta(n::Int64)::Vector{Float64}
    return rand(-10:0.01:10, n)
end

## Нормы

function euclidean_norm(A)::Float64
    return sqrt(sum(abs2, A))
end

function uniform_norm(x::Vector{Float64})::Float64
    return maximum(abs.(x))
end

function uniform_norm(x::Matrix{Float64})::Float64
    return maximum(sum(abs, x, dims=2))
end

## Число обусловленности матрицы

function get_condition_number(A::Matrix{Float64}, _norm::Function)::Float64
    return _norm(inv(A)) * _norm(A)
end

## Коэффициент роста элементов матрицы

function gauss_growth_factor(A::Matrix{Float64})::Float64
    n = size(A, 1)
    max_initial = maximum(abs.(A))
    max_during = max_initial

    A_work = copy(A)

    for k in 1:n-1
        for i in k+1:n
            if A_work[k, k] != 0
                factor = A_work[i, k] / A_work[k, k]
                for j in k:n
                    A_work[i, j] -= factor * A_work[k, j]
                end
            end
        end
        max_during = max(max_during, maximum(abs.(A_work)))
    end
    
    return max_during / max_initial
end

## Оценки погрешностей

function error_rounding(
    A::Matrix{Float64},
    p::Int64,
    t::Int64,
    _norm::Function
)::Float64
    nu_A = get_condition_number(A, _norm)
    n = size(A, 1)
    g_A = gauss_growth_factor(A)
    return nu_A * n * g_A / p ^ t
end

function error_input_data(
    A::Matrix{Float64}, delta_A::Matrix{Float64},
    f::Vector{Float64}, delta_f::Vector{Float64},
    _norm::Function
)::Float64
    nu_A = get_condition_number(A, _norm)
    error_A = _norm(delta_A) / _norm(A)
    error_f = _norm(delta_f) / _norm(f)
    return nu_A * (error_A + error_f)
end

## Погрешность решения

function error_result(
    A::Matrix{Float64}, delta_A::Matrix{Float64},
    f::Vector{Float64}, delta_f::Vector{Float64},
    _norm::Function
)::Float64
    x = A \ f
    delta_x = (A + delta_A) \ (f + delta_f) - x
    return _norm(delta_x) / _norm(x)
end

## Тестирование

A = [
    100. 99.;
    99. 98.
]
delta_A = [
    0. 0.;
    0. 0.
]
f = [
    199.,
    197.
]
delta_f = [
    -.01,
    .01
]

print("Относительная ошибка входных данных: ", error_input_data(A, delta_A, f, delta_f, uniform_norm))
print("\nОтносительная ошибка результата: ", error_result(A, delta_A, f, delta_f, uniform_norm))

A = generate_matrix(5)
delta_A = generate_matrix_delta(5)
f = generate_vector(5)
delta_f = generate_vector_delta(5)

_norm = euclidean_norm

print("Относительная ошибка округления: ", error_rounding(A, 2, 11, _norm))
print("\nОтносительная ошибка входных данных: ", error_input_data(A, delta_A, f, delta_f, _norm))
print("\nОтносительная ошибка результата: ", error_result(A, delta_A, f, delta_f, _norm))