plot(mtcars$mpg, mtcars$disp)
hist(mtcars$mpg)

library(tidyverse)

cars_tbl <- as_tibble(mtcars)

mtcars %>% 
  ggplot(
    mapping = aes(x = mpg, y = disp)
  ) +
  geom_point()

mtcars %>% 
  ggplot(aes(mpg, disp)) +
  geom_point(size = 3, shape = "diamond")

mtcars %>% 
  ggplot() +
  geom_point(aes(mpg, disp)) +
  geom_line(aes(mpg, disp))

mtcars %>% 
  ggplot(aes(mpg, disp)) +
  geom_point() +
  geom_line(aes(mpg, hp))


mtcars %>% 
  ggplot(aes(mpg, disp)) +
  geom_point(size = 3) +
  geom_line(size = 1)


cars_tbl %>% 
  ggplot(aes(disp, mpg)) +
  geom_point(aes(color = as.factor(cyl)), size = 3)

cars_tbl %>% 
  ggplot(aes(disp, mpg, color = as.factor(cyl))) +
  geom_point(size = 3)

cars_tbl %>% 
  ggplot(aes(disp, mpg, shape = as.factor(cyl), color = as.factor(cyl))) +
  geom_point(size = 3) +
  scale_color_manual(values = c("#19a6b3","#f26c0d","#5e3894"))

cars_tbl %>% 
  ggplot(aes(disp, mpg, shape = as.factor(cyl), color = as.factor(cyl))) +
  geom_point(size = 3) +
  scale_color_brewer(palette = 'Set1')

cars_tbl %>% 
  ggplot(aes(disp, mpg, shape = as.factor(cyl), color = as.factor(cyl))) +
  geom_point(size = 3) +
  scale_color_brewer(palette = 'Set1') +
  theme_void() +
  theme(legend.position = 'none',
        panel.background = element_rect(fill = "lightblue"))


cars_tbl %>% 
  ggplot(aes(disp, mpg, shape = as.factor(cyl), color = as.factor(cyl))) +
  geom_point(size = 3) +
  scale_color_brewer(palette = 'Set1') +
  theme_classic() +
  labs(x = "Об'єм двигуна",
       y = "Споживання пального",
       title = "Залежність споживання пального від об'єму двигуна",
       subtitle = "за якийсь-період",
       caption = "Дані: mtcars") +
  scale_x_continuous(breaks = c(seq(80, 500, 50)),
                     limits = c(130, 380))


cars_tbl %>% 
  ggplot(aes(disp, mpg, shape = as.factor(cyl), color = as.factor(cyl))) +
  geom_point(size = 3) +
  scale_color_brewer(palette = 'Set1') +
  theme_classic() +
  labs(x = "Об'єм двигуна",
       y = "Споживання пального",
       title = "Залежність споживання пального від об'єму двигуна",
       subtitle = "за якийсь-період",
       caption = "Дані: mtcars") +
  coord_cartesian(xlim = c(0, 400), ylim = c(0, 35))

cars_tbl %>% 
  ggplot(aes(disp, mpg, shape = as.factor(cyl), color = as.factor(cyl))) +
  geom_point(size = 3) + 
  geom_smooth(method = 'loess', se = FALSE)

cars_tbl %>% 
  ggplot() +
  geom_bar(aes(as.factor(carb), fill = as.factor(carb))) +
  coord_flip()

cars_tbl %>% 
  ggplot(aes(mpg, fill = as.factor(vs))) +
  geom_histogram(bins = 5, alpha = 0.5) +
  facet_grid(vs ~ .)

library(plotly)


my_plot <- cars_tbl %>% 
  ggplot(aes(disp, mpg, shape = as.factor(cyl), color = as.factor(cyl))) +
  geom_point(size = 3) +
  scale_color_manual(values = c("#19a6b3","#f26c0d","#5e3894"))

my_plot

ggplotly(my_plot)



















