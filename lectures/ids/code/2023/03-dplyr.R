library(tidyverse)
library(readxl)

readxl::excel_sheets("data/tourism.xlsx")

tour <- read_excel("data/tourism.xlsx", sheet = 2)

tour

# Фільтрація

filter(tour, Trips > mean(Trips))

filter(tour, Trips > mean(Trips), Region == "Hobart and the South")

filter(tour, Region == 'Adelaide' | Region == 'Alice Springs' | Region == 'Canberra')

filter(tour, Region %in% c('Adelaide', 'Alice Springs', 'Canberra'))

# Зрізи

slice(tour, c(1, 5, 15))

slice(tour, seq(1, nrow(tour), 100))

slice(tour, -(1:100))

slice(tour, -c(1:100))

slice_head(tour, n = 100) # head()

slice_tail(tour, n = 100) # tail()

slice_max(tour, Trips, n = 10)

slice_min(tour, Trips, n = 10)

slice_sample(tour, n = 100)

slice_sample(tour, n = nrow(tour), replace = TRUE)

# Пайп %>%  --- Ctrl + Shift + M

slice_max(filter(tour, Region %in% c('Adelaide', 'Alice Springs', 'Canberra')), Trips, n = 3)

sort(sqrt(abs(sin(1:10))))

1:10 %>% 
  sin() %>% 
  abs() %>% 
  sqrt() %>% 
  sort()

filter(tour, Region %in% c('Adelaide', 'Alice Springs', 'Canberra')) %>% 
  slice_max(Trips, n = 3)

tour %>% 
  filter(Region %in% c('Adelaide', 'Alice Springs', 'Canberra')) %>% 
  slice_max(Trips, n = 3)

1:10 |> 
  sin() |> 
  abs() |>
  sqrt() |>
  sort()

# Сортування

tour %>% 
  filter(Region %in% c('Adelaide', 'Alice Springs', 'Canberra')) %>% 
  arrange(Trips)

tour %>% 
  filter(Region %in% c('Adelaide', 'Alice Springs', 'Canberra')) %>% 
  arrange(desc(Trips))

tour %>% 
  filter(Region %in% c('Adelaide', 'Alice Springs', 'Canberra')) %>% 
  arrange(-Trips)

tour %>% 
  filter(Region %in% c('Adelaide', 'Alice Springs', 'Canberra')) %>% 
  arrange(Trips, Quarter)

# Відбір змінник

tour_select <- tour %>% 
  select(Quarter, Trips)

tour_select

tour %>% 
  select(1, 3, 5)

tour %>% 
  select(State:Trips, Quarter)

tour %>% 
  select(-State)

tour %>% 
  select(!c(State, Purpose))

tour %>% 
  select(starts_with("Q"))

tour %>% 
  select(ends_with("e"))

tour %>% 
  select(contains("e"))

tour %>% 
  select(matches("[ra]t")) # rt або at

tour %>% 
  select(where(is.numeric))

tour %>% 
  select(where(is.character))

tour %>% 
  select(where(is.numeric), everything())

# Зміна позицій змінних

tour %>% 
  relocate(Region)

tour %>% 
  relocate(Trips, .after = Quarter)

tour %>% 
  relocate(Trips, .before = Quarter)

tour %>% 
  relocate(where(is.double), .after = Quarter)

# Зміна назв змінних

tour %>% 
  rename(Date = Quarter) %>% # Нова_назва = Стара_назва
  rename_with(tolower)

tour %>% 
  rename_with(tolower)

tour %>% 
  rename_with(toupper)

tour %>% 
  rename_with(tolower, ends_with('e'))

tour_rename <- tour %>% 
  rename('Date Quarter' = Quarter)

tour_rename

tour_rename %>% 
  rename_with(~ gsub(" ", "_", .x)) %>% 
  rename_with(tolower)

tour_rename %>% 
  set_names(names(.) %>% str_replace(" ", "_") %>% str_to_lower())

# Створення змінних

tour_tbl <- tour %>% 
  mutate(Quarter = as.Date(Quarter),
         Region = as.factor(Region),
         State = as.factor(State),
         Purpose = as.factor(Purpose),
         Trips = as.integer(Trips)) %>% 
  set_names(names(.) %>% str_replace(" ", "_") %>% str_to_lower())
  
tour_tbl %>% 
  mutate(trips_percent = trips / mean(trips))

tour_tbl %>% 
  transmute(quarter,
            trips,
            trips_percent = trips / mean(trips))

# Групування та агрегація

tour_tbl %>% 
  group_by(purpose) %>% 
  summarise(avg_trip = mean(trips)) %>% 
  arrange(desc(avg_trip))

tour_tbl %>% 
  group_by(purpose, state) %>% 
  summarise(avg_trip = mean(trips)) %>% 
  ungroup()

tour_tbl %>% 
  group_by(purpose, state) %>% 
  summarise(avg_trip = mean(trips),
            sd_trip = sd(trips),
            md_trip = median(trips)) %>% 
  ungroup()

set.seed(123)
c(sample(1:10, size = 10, replace = TRUE), 1000) %>% 
  mean()

set.seed(123)
c(sample(1:10, size = 10, replace = TRUE), 1000) %>% 
  median()

tour_tbl %>% 
  group_by(purpose) %>% 
  summarise(avg_trip = mean(trips),
            md_trip = median(trips))


tour_tbl %>% 
  mutate(year = year(quarter), .after = quarter) %>% 
  mutate(month = month(quarter), .after = year) %>% 
  group_by(year) %>% 
  summarise(avg_trip = mean(trips),
            md_trip = median(trips))

tour_tbl %>% 
  mutate(year = year(quarter), .after = quarter) %>% 
  mutate(month = month(quarter), .after = year) %>% 
  group_by(month) %>% 
  summarise(avg_trip = mean(trips),
            md_trip = median(trips))

tour_tbl %>% 
  mutate(year = year(quarter), .after = quarter) %>% 
  mutate(month = month(quarter), .after = year) %>% 
  group_by(year, month) %>% 
  summarise(avg_trip = mean(trips),
            md_trip = median(trips)) %>% 
  ungroup()

# Кількість

tour_tbl %>% 
  group_by(purpose) %>% 
  summarise(n = n()) %>% 
  arrange(-n)

tour_tbl %>% 
  count(purpose, sort = TRUE)

# Створення груп

summary(tour_tbl$trips)

tour_tbl %>% 
  mutate(trip_group = case_when(trips > 78 ~ 'h',
                                trips > 7 ~ 'm',
                                TRUE ~ 'l')) %>% 
  count(trip_group)































