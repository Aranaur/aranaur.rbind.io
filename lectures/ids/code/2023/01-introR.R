# Арифметика

2 + 2 # додавання
5 - 3
3 * 5
25 / 5
3 ^ 4
3 ** 4
5 %% 3
17 %/% 5

# Пріоритети

2 + 3 * 4
(2 + 3) * 4

# Готові функції

9 ** 0.5
9 ** (1/2)
sqrt(9)

abs(5 - 10)

log(x = 10)
log(x = 10, base = 2)
log(10, 2)
log(base = 2, x = 10)

log(2, 10)

# Змінні

var_log = log(2, 10)
var_abs <- abs(5 - 10) # "alt -"
sqrt(9) -> var_sqrt

var_log + var_abs ** var_sqrt

# Оператори порівняння

var_log == var_abs
var_log != var_abs
var_log > var_abs
var_log < var_abs
var_log >= var_abs
var_log <= var_abs

# Типи даних

## Числа

is.numeric(var_abs)
is.integer(var_abs)
is.double(var_abs)

is.integer(10L)

is.integer(as.integer(10))

as.integer(is.double(var_log))

round(var_log, 0)

## Текст

var_text <- "10"
var_text2 <- 'Hello, R!'
var_text_true <- "TRUE"
var_name <- "Мар'яна"
var_name_2 <- 'Мар\'яна'

## Логічні

var_t <- TRUE
var_f <- FALSE

# Структури даних

## Вектор

vec_1 <- c(var_sqrt, 2, 5, 32, -10, var_abs)
vec_text <- c("Hello", "R", var_name)
vec_logic <- c(TRUE, var_t, FALSE)

vec_generator <- -5:5
vec_generator2 <- 3:-2

seq(from = 1, to = 10, by = 2)
seq(1, 10, 2)
seq(1, 10, length.out = 935)

rep(3, 5)
rep(1:3, 5)
rep(1:3, 1:3)

c("Hello!", 1, -5)
c(TRUE, 5, FALSE)

5 + TRUE

c("Hello", 10, TRUE)

1:3 + 2:4

1:3 + 1:6

1:3 + 1:5

c("hello!", "r") + c('buy!', 'python')

vec_1 * 5

vector_index <- -5:5

vector_index[9]

vector_index[c(1, 3, 5)]

vector_index[seq(1, length(vector_index), 4)]

vector_index[3:1]

vector_index[-1]

vector_index[c(-1, -2)]























