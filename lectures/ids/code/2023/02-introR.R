# Матриці

matrix(1:16, nrow = 4)

matrix(1:16, ncol = 4)

matrix(1:16, nrow = 4, ncol = 4)

matrix(1:16, nrow = 4, ncol = 4, byrow = TRUE)

my_matrix <- matrix(1:16, nrow = 4, ncol = 4,
       byrow = TRUE, dimnames = list(c("row_1", "row_2", "row_3", "row_4"),
                                     c("col_1", "col_2", "col_3", "col_4")))

my_matrix

my_matrix[1, 3]
my_matrix[1:2, 3:4]

my_matrix[1:2, 3:4] <- 5
my_matrix

my_matrix * 10
t(my_matrix)

# Масиви

array(1:16, dim = c(4, 2, 2)) * 10

# Списки

my_list <- list(vec = 1:5,
                hello = "Hello, R!",
                matrix = matrix(1:16, nrow = 4),
                array = array(1:16, dim = c(4, 2, 2)))

my_list

my_list[1]

my_list[[1]]
my_list[[2]]
my_list[[3]]
my_list[[4]]

my_list$hello

my_list[['vec']][4]

my_list$vec[4]

# data frame

ds_lec <- data.frame(
  name = c("Ігор", "Анастасія", "Дмитро", "Анастасія"),
  sex = factor(c("ч", "ж", "ч", "ж")),
  check = c(TRUE, TRUE, TRUE, FALSE),
  child = c(3, 0, 1, NA)
)

ds_lec

str(ds_lec)

str(ds_lec)
str(my_list)

str(mtcars)
str(iris)

iris

names(ds_lec)

head(iris, n = 10)
tail(iris, n = 20)

ds_lec[, 1]
ds_lec['name']
ds_lec[3, 'name']

ds_lec$child[1]

ds_lec[1, ]

# Фактор

race <- factor(
  c("людина", "ельф", "дворф", "орк", "гоблін", "людина", "ельф", "маг"),
  levels = c("гоблін", "орк", "дворф", "людина", "ельф"),
  ordered = TRUE
)

race

# Пакети

nrow(available.packages())

library(tidyverse)

# Обробка даних: dplyr

iris

## tibble

iris_tbl <- as_tibble(iris)
iris_tbl

class(iris_tbl)

head(iris_tbl)
str(iris_tbl)
glimpse(iris_tbl) # аналог str()

## Читання даних (.csv, .xlsx)

dino_tbl <- read_csv('https://raw.githubusercontent.com/Aranaur/aranaur.rbind.io/main/datasets/datasaurus/datasaurus.csv',
                     col_names = c("var1", "var2", "var3"),
                     skip = 1)

dino_tbl

read_csv2('https://raw.githubusercontent.com/Aranaur/aranaur.rbind.io/main/datasets/datasaurus/datasaurus.csv')

read_delim('https://raw.githubusercontent.com/Aranaur/aranaur.rbind.io/main/datasets/datasaurus/datasaurus.csv',
           delim = '',
           skip = 1,
           col_names = )

dino_local_tbl <- read_csv('data/datasaurus_datasaurus.csv')

library(readxl)

tourism_tbl <- read_excel('data/tourism.xlsx', sheet = 2)
tourism_tbl

excel_sheets('data/tourism.xlsx')

read_excel('data/tourism.xlsx', sheet = excel_sheets('data/tourism.xlsx')[2])

# -------------

files <- dir(path = 'data/', pattern = '^20')
dino_all <- read_csv(file = paste0('data/', files))
dino_all
dino_local_tbl

















