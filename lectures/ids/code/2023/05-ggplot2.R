library(tidyverse)
library(palmerpenguins)

penguins

# basic

plot(penguins$bill_length_mm, penguins$bill_depth_mm)

hist(penguins$body_mass_g)

boxplot(body_mass_g ~ sex, data = penguins)

# ggplot2

ggplot(data = penguins,
       mapping = aes(x = bill_length_mm,
                     y = bill_depth_mm)) +
  geom_point()

penguins %>% 
  filter(body_mass_g >= mean(body_mass_g, na.rm = TRUE)) %>% 
  ggplot(aes(bill_length_mm, bill_depth_mm)) +
  geom_point()

penguins %>% 
  ggplot(aes(body_mass_g)) +
  geom_histogram(bins = 10)

penguins %>% 
  ggplot(aes(body_mass_g)) +
  geom_histogram(binwidth = 750)

penguins %>% 
  ggplot(aes(sex, body_mass_g)) +
  geom_boxplot()

# advance

penguins %>% 
  ggplot(aes(bill_length_mm, bill_depth_mm)) +
  geom_point(size = 3, shape = 22)

penguins %>% 
  ggplot(aes(bill_length_mm,
             bill_depth_mm,
             size = body_mass_g,
             shape = species,
             color = species
             )) +
  geom_point(alpha = 0.5) +
  expand_limits(x = 0, y = 0)

penguins %>% 
  ggplot(aes(bill_length_mm,
             bill_depth_mm,
             size = body_mass_g,
             shape = species,
             color = species
  )) +
  geom_point(alpha = 0.5) +
  scale_x_continuous(limits = c(40, 50)) +
  scale_y_continuous(limits = c(15, 20))

penguins %>% 
  ggplot(aes(bill_length_mm,
             bill_depth_mm,
             size = body_mass_g,
             color = species
  )) +
  geom_point(alpha = 0.5) +
  labs(
    x = "Довжина клюва, мм",
    y = "Ширина клюва, мм",
    title = "Характеристики типів пінгвінів",
    subtitle = "Види пінгвінів зі станції Палмер",
    caption = "Дані: palmerpenguins",
    size = "Масса, г",
    color = "Тип"
  ) +
  theme_minimal()
  
penguins %>% 
  ggplot(aes(bill_length_mm,
             bill_depth_mm,
             size = body_mass_g,
             color = species
  )) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    x = "Довжина клюва, мм",
    y = "Ширина клюва, мм",
    title = "Характеристики типів пінгвінів",
    subtitle = "Види пінгвінів зі станції Палмер",
    caption = "Дані: palmerpenguins",
    size = "Масса, г",
    color = "Тип"
  )
  
penguins %>% 
  ggplot() +
  geom_point(aes(bill_length_mm,
                 bill_depth_mm,
                 size = body_mass_g,
                 color = species),
             alpha = 0.5) +
  geom_smooth(aes(bill_length_mm,
                  bill_depth_mm),
              method = "lm",
              se = FALSE)
  
# bar plot

penguins %>% 
  count(species, sort = TRUE) %>% 
  ggplot() +
  geom_col(aes(x = species, y = n, fill = species))

penguins %>% 
  ggplot() +
  geom_bar(aes(x = species, fill = species)) +
  theme(legend.position = "none")

# lolipop

penguins %>% 
  count(species, sort = TRUE) %>% 
  mutate(species = factor(species, levels = species)) %>% 
  ggplot() +
  geom_segment(aes(x = species, xend = species,
                   y = 0, yend = n), linewidth = 1) +
  geom_point(aes(species, n, color = species), size = 4) +
  coord_flip()

# hist

penguins %>% 
  ggplot(aes(body_mass_g, fill = species)) +
  geom_histogram(bins = 15) +
  facet_wrap(~ species)

penguins %>% 
  ggplot(aes(body_mass_g, fill = species)) +
  geom_histogram(bins = 15) +
  facet_grid(species ~ ., scales = "free")

penguins %>% 
  ggplot(aes(body_mass_g, fill = species)) +
  geom_density() +
  facet_wrap(~ species)

penguins %>% 
  ggplot(aes(body_mass_g, fill = species)) +
  geom_density() +
  facet_grid(species ~ ., scales = "free")

# joyplot

library(ggridges)

penguins %>% 
  ggplot() +
  geom_density_ridges(aes(x=body_mass_g, y = species, fill = species),
                      color = NA,
                      scale = 1) +
  theme_minimal()

# plotly

library(plotly)

bar_plot <- penguins %>% 
  ggplot() +
  geom_bar(aes(x = species, fill = species)) +
  theme(legend.position = "none")

ggplotly(bar_plot)


library(gridExtra)

bar_plot

joyplot <- penguins %>% 
  ggplot() +
  geom_density_ridges(aes(x=body_mass_g, y = species, fill = species),
                      color = NA,
                      scale = 1) +
  theme_minimal()

main_plot <- penguins %>% 
  ggplot(aes(bill_length_mm,
             bill_depth_mm,
             size = body_mass_g,
             color = species
  )) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    x = "Довжина клюва, мм",
    y = "Ширина клюва, мм",
    title = "Характеристики типів пінгвінів",
    subtitle = "Види пінгвінів зі станції Палмер",
    caption = "Дані: palmerpenguins",
    size = "Масса, г",
    color = "Тип"
  )

grid.arrange(
  main_plot, bar_plot, joyplot, nrow = 3
)


























  
  
  
  
































